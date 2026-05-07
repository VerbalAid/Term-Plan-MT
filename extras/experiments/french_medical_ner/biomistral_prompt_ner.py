#!/usr/bin/env python3
"""LLM-based French medical term extraction (HF 4-bit, Ollama, or Unsloth merged LoRA) → segments JSONL."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cuda_ld_path import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger("biomistral_prompt_ner")

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_DEFAULT_UNSLOTH_MERGED = ROOT / "models" / "biomistral-ner-merged"
# Local LoRA dir is optional (not shipped); pass --unsloth-lora-path when you have one.
_DEFAULT_UNSLOTH_LORA = ROOT / "models" / "biomistral-ner-lora"


def _dir_has_hf_model_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        return True
    return any(path.glob("model-*.safetensors"))


def model_slug_from_hf_id(model_id: str) -> str:
    """Stable filename slug from a Hugging Face model id (last path segment, lower_snake)."""
    tail = model_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", tail).strip("_").lower()


def default_output_path(
    *,
    backend: str,
    hf_model: str,
    ollama_model: str,
    unsloth_path: Path,
) -> Path:
    if backend == "ollama":
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", ollama_model.strip()).strip("_").lower()
    elif backend == "unsloth":
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", unsloth_path.name).strip("_").lower()
        # Legacy default: merged BioMistral NER export → same filename as before.
        if "biomistral" in stem and ("merged" in stem or stem.endswith("ner_merged")):
            slug = "unsloth"
        else:
            slug = stem or "unsloth"
    else:
        slug = model_slug_from_hf_id(hf_model)
    return ROOT / "data" / "section48" / f"segments_ner_{slug}.jsonl"


def _require_min_free_gpu_mib(min_mib: int, *, what: str) -> None:
    """Fail fast when another process (or a zombie training run) leaves too little VRAM."""
    if not torch.cuda.is_available():
        return
    try:
        free_b, _tot = torch.cuda.mem_get_info()
        free_mib = int(free_b // (1024 * 1024))
    except Exception:
        return
    if free_mib < min_mib:
        raise SystemExit(
            f"Need about >= {min_mib} MiB free GPU RAM for {what}; only {free_mib} MiB free.\n"
            "Run:  nvidia-smi  — stop other Python / training jobs (note the PID using several GiB), "
            "or reboot.\n"
            "Then open a **new** shell and retry with:\n"
            "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )


def _load_causal_4bit(model_id_or_path: str | Path) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """4-bit load pattern aligned with ``src/systems/models.load_mistral_4bit``."""
    path_str = str(model_id_or_path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _require_min_free_gpu_mib(3072, what="4-bit causal LM load")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(path_str, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dm_raw = os.environ.get("BIOMISTRAL_DEVICE_MAP", os.environ.get("MISTRAL_DEVICE_MAP", "")).strip().lower()
    if dm_raw == "auto":
        device_map = "auto"
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        path_str,
        quantization_config=bnb,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    return tok, model


def _load_causal_4bit_with_lora(base_model_id: str, lora_dir: Path) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """4-bit base + on-disk LoRA (same inference path as merged, without exporting merged FP weights)."""
    from peft import PeftModel

    path_str = str(lora_dir.resolve())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _require_min_free_gpu_mib(3072, what="4-bit base + LoRA load")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dm_raw = os.environ.get("BIOMISTRAL_DEVICE_MAP", os.environ.get("MISTRAL_DEVICE_MAP", "")).strip().lower()
    if dm_raw == "auto":
        device_map = "auto"
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, path_str)
    return tok, model


def _first_balanced_json_array_slice(s: str) -> str | None:
    """Slice from first ``[`` through its matching ``]``, respecting JSON string quoting."""
    i = s.find("[")
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def _quoted_strings_after_bracket(s: str, *, max_terms: int = 384) -> list[str] | None:
    """When the model repeats tokens until ``max_new_tokens`` without closing ``]``, recover quoted strings."""
    lb = s.find("[")
    if lb < 0:
        return None
    chunk = s[lb : lb + 500_000]
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'"((?:\\.|[^"\\])*)"', chunk):
        if len(out) >= max_terms:
            break
        try:
            val = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(val, str):
            w = val.strip()
            if w and w not in seen:
                seen.add(w)
                out.append(w)
    return out if out else None


def _json_decode_candidates(text: str) -> list[str]:
    """Return strings to try with json.loads (full text, fenced block, bracket slice)."""
    t = text.strip()
    out: list[str] = []
    balanced = _first_balanced_json_array_slice(t)
    if balanced:
        out.append(balanced)
    if t:
        out.append(t)
    for m in _FENCE.finditer(text):
        inner = m.group(1).strip()
        if inner:
            out.append(inner)
    lb = t.find("[")
    rb = t.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        naive = t[lb : rb + 1]
        if naive not in out:
            out.append(naive)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _list_to_terms(data: object) -> list[dict[str, object]]:
    """Match ``prepare_data.py`` shape: ``word``, ``score`` (optional ``entity`` omitted)."""
    out: list[dict[str, object]] = []
    if not isinstance(data, list):
        return out
    for item in data:
        if isinstance(item, str):
            w = item.strip()
            if w:
                out.append({"word": w, "score": 1.0})
            continue
        if isinstance(item, dict):
            w = (
                item.get("word")
                or item.get("fr")
                or item.get("term")
                or item.get("text")
                or item.get("mention")
            )
            if isinstance(w, str) and w.strip():
                term: dict[str, object] = {"word": w.strip(), "score": float(item.get("score", 1.0))}
                for k in ("en", "level", "tier", "id"):
                    if k in item and item[k] is not None and item[k] != "":
                        term[k] = item[k]
                out.append(term)
    return out


def _dict_values_to_terms(data: dict) -> list[dict[str, object]]:
    """Recover terms when the model returns a JSON object instead of an array."""
    out: list[dict[str, object]] = []
    for _k, v in data.items():
        if isinstance(v, str) and v.strip():
            out.append({"word": v.strip(), "score": 1.0})
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    out.append({"word": x.strip(), "score": 1.0})
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for t in out:
        w = str(t["word"])
        if w not in seen:
            seen.add(w)
            deduped.append(t)
    return deduped


def _parse_bracket_list_loose(raw: str) -> list[str] | None:
    """Parse ``[a, b, c]`` when the model omits JSON string quotes."""
    t = raw.strip()
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    lb = t.find("[")
    rb = t.rfind("]")
    if lb == -1 or rb == -1 or rb <= lb:
        return None
    inner = t[lb + 1 : rb].strip()
    if not inner:
        return []
    parts = re.split(r",\s*", inner)
    out: list[str] = []
    for piece in parts:
        p = piece.strip()
        if not p:
            continue
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            out.append(p[1:-1].strip())
            continue
        if p.startswith("[") or p.startswith("{"):
            return None
        out.append(p)
    return out if out else None


def parse_llm_terms(raw: str) -> list[dict[str, object]] | None:
    """Return ``terms`` list, or ``None`` if no valid JSON was parsed."""
    for cand in _json_decode_candidates(raw):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return _list_to_terms(data)
        if isinstance(data, dict):
            if "terms" in data and isinstance(data["terms"], list):
                return _list_to_terms(data["terms"])
            alt = _dict_values_to_terms(data)
            if alt:
                return alt
    loose = _parse_bracket_list_loose(raw)
    if loose:
        return [{"word": w, "score": 1.0} for w in loose]
    recovered = _quoted_strings_after_bracket(raw)
    if recovered:
        return [{"word": w, "score": 1.0} for w in recovered]
    return None


def build_prompt(fr: str) -> list[dict[str, str]]:
    """Zero-shot French prompt (same for BioMistral, Mistral-Instruct, Ollama)."""
    body = (
        "Tu es un expert en terminologie médicale. Extrais du texte français ci-dessous "
        "tous les termes médicaux pertinents : noms de médicaments, effets indésirables, "
        "et procédures.\n"
        "Réponds UNIQUEMENT avec un tableau JSON : une liste Python de chaînes françaises, "
        "sans clés nommées, sans objet JSON {{ }}, sans markdown. "
        'Le premier caractère doit être `[` et le dernier `]`.\n'
        'Exemple valide : ["pembrolizumab", "hypothyroïdie"]\n\n'
        f"Texte:\n{fr}"
    )
    return [{"role": "user", "content": body}]


ALPACA_INSTRUCTION = (
    "Extract all medical terms (disorders, drugs, procedures) from this French medical text. "
    "Return only a JSON list of strings."
)


def build_alpaca_prompt(fr: str, *, include_response_prefix: bool = True) -> str:
    """Alpaca template used for Unsloth LoRA train/infer (English instruction, French input)."""
    block = (
        "### Instruction:\n"
        f"{ALPACA_INSTRUCTION}\n\n"
        "### Input:\n"
        f"{fr}\n\n"
        "### Response:\n"
    )
    if include_response_prefix:
        return block
    return block


def generate_terms_hf_chat(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    fr: str,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
) -> tuple[list[dict[str, object]] | None, str]:
    messages = build_prompt(fr)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = messages[0]["content"]

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        dev = model.device
    else:
        dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
    }
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    in_len = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=True)
    return parse_llm_terms(gen_text), gen_text


def generate_terms_alpaca(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    fr: str,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
) -> tuple[list[dict[str, object]] | None, str]:
    prompt = build_alpaca_prompt(fr, include_response_prefix=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        dev = model.device
    else:
        dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
    }
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    in_len = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=True)
    return parse_llm_terms(gen_text), gen_text


def ollama_chat(
    *,
    base_url: str,
    model: str,
    user_content: str,
    timeout: float | None = None,
    num_predict: int = 512,
) -> str:
    """POST /api/chat (non-streaming).

    ``timeout`` is the urllib socket timeout in seconds for the whole request (connect + read).
    ``None`` means no limit (recommended for slow CPU inference). ``num_predict`` maps to Ollama
    ``options.num_predict`` so generation does not run unbounded.
    """
    url = base_url.rstrip("/") + "/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
            "options": {
                "num_predict": max(64, int(num_predict)),
                "temperature": 0,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit(
                f"ERROR: Ollama returned 404. Is 'ollama serve' running? Is model '{model}' pulled? "
                f"Run: ollama pull {model}"
            ) from e
        raise SystemExit(f"Ollama HTTP {e.code} ({url}): {e}") from e
    except TimeoutError:
        raise
    except urllib.error.URLError as e:
        raise SystemExit(f"Ollama request failed ({url}): {e}") from e
    msg = payload.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise SystemExit(f"Unexpected Ollama response: {payload!r}")
    return content


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=(
            "Zero-shot / merged LoRA NER term extraction to JSONL. "
            "HF: pass --huggingface-model (e.g. BioMistral/BioMistral-7B or "
            "mistralai/Mistral-7B-Instruct-v0.2) for identical French prompt + JSON parsing. "
            "Ollama: --backend ollama --ollama-model llama3 (local server). "
            "Unsloth: default loads merged weights from models/biomistral-ner-merged; "
            "or pass --unsloth-lora-path for base + adapter (e.g. ontology Qwen: "
            "--unsloth-base-model Qwen/Qwen2.5-3B-Instruct --unsloth-lora-path models/...-lora). "
            "~8 GB GPUs: free VRAM first (nvidia-smi); optional BIOMISTRAL_DEVICE_MAP=auto "
            "(may fail bitsandbytes if layers offload to CPU)."
        ),
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner.jsonl",
        help="Source JSONL (preserves id, fr, en_ref; overwrites terms).",
    )
    ap.add_argument(
        "--output",
        "--out",
        type=Path,
        default=None,
        dest="output",
        help="Output JSONL. Default: data/section48/segments_ner_<slug>.jsonl from model/backend.",
    )
    ap.add_argument(
        "--backend",
        choices=("huggingface", "ollama", "unsloth"),
        default="huggingface",
        help="Inference backend (default: huggingface 4-bit).",
    )
    ap.add_argument(
        "--huggingface-model",
        "--model",
        type=str,
        default="BioMistral/BioMistral-7B",
        dest="hf_model",
        help=(
            "HF model id or path (4-bit). Examples: BioMistral/BioMistral-7B, "
            "mistralai/Mistral-7B-Instruct-v0.2."
        ),
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--ollama-model",
        type=str,
        default="llama3",
        help="Ollama model tag when --backend ollama (e.g. llama3, llama3:8b). Requires local Ollama.",
    )
    ap.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama HTTP base URL (default 127.0.0.1:11434 or OLLAMA_HOST).",
    )
    ap.add_argument(
        "--ollama-timeout",
        type=float,
        default=0.0,
        help=(
            "Per-request socket timeout in seconds for Ollama /api/chat (connect+read). "
            "0 = no limit (recommended for slow CPU or long segments). "
            "If a positive value is set and a call times out, one retry is done with no limit."
        ),
    )
    ap.add_argument(
        "--unsloth-merged-path",
        type=Path,
        default=_DEFAULT_UNSLOTH_MERGED,
        help="Merged model dir for --backend unsloth (default: models/biomistral-ner-merged).",
    )
    ap.add_argument(
        "--unsloth-base-model",
        type=str,
        default="BioMistral/BioMistral-7B",
        help=(
            "HF base when using --unsloth-lora-path (must match fine-tune base). "
            "Ontology-tuned Qwen 3B: Qwen/Qwen2.5-3B-Instruct."
        ),
    )
    ap.add_argument(
        "--unsloth-lora-path",
        type=Path,
        default=None,
        help=(
            "Optional: folder with LoRA adapter (e.g. models/biomistral-ner-lora). "
            "Loads base in 4-bit + adapter; use when merged export has no weight files."
        ),
    )
    args = ap.parse_args()

    inp = args.input if args.input.is_absolute() else ROOT / args.input
    out_path = args.output
    if out_path is None:
        if args.backend == "unsloth" and args.unsloth_lora_path is not None:
            up = args.unsloth_lora_path if args.unsloth_lora_path.is_absolute() else ROOT / args.unsloth_lora_path
        else:
            up = args.unsloth_merged_path if args.unsloth_merged_path.is_absolute() else ROOT / args.unsloth_merged_path
        out_path = default_output_path(
            backend=args.backend,
            hf_model=args.hf_model,
            ollama_model=args.ollama_model,
            unsloth_path=up,
        )
    elif not out_path.is_absolute():
        out_path = ROOT / out_path

    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer: AutoTokenizer | None = None
    model: AutoModelForCausalLM | None = None

    try:
        if args.backend == "ollama":
            log.info("Ollama model=%s base=%s (user prompt matches HF zero-shot).", args.ollama_model, args.ollama_base_url)
        elif args.backend == "unsloth":
            base_id = args.unsloth_base_model
            if args.unsloth_lora_path is not None:
                lp = args.unsloth_lora_path if args.unsloth_lora_path.is_absolute() else ROOT / args.unsloth_lora_path
                if not lp.is_dir():
                    raise SystemExit(f"LoRA adapter dir not found: {lp}")
                log.info("Loading base %s + LoRA (4-bit): %s", base_id, lp)
                tokenizer, model = _load_causal_4bit_with_lora(base_id, lp)
            else:
                merged = args.unsloth_merged_path
                merged = merged if merged.is_absolute() else ROOT / merged
                if not merged.is_dir():
                    raise SystemExit(f"Unsloth merged model not found: {merged}")
                if not _dir_has_hf_model_weights(merged):
                    raise SystemExit(
                        f"Unsloth path has tokenizers but no model weights: {merged}\n"
                        "Re-run finetune merge (PEFT fallback) or use:\n"
                        f"  --unsloth-lora-path {_DEFAULT_UNSLOTH_LORA}\n"
                        f"  --unsloth-base-model {base_id}"
                    )
                log.info("Loading Unsloth merged model (4-bit): %s", merged)
                tokenizer, model = _load_causal_4bit(merged)
        else:
            log.info("Loading %s (4-bit, HF)...", args.hf_model)
            tokenizer, model = _load_causal_4bit(args.hf_model)

        if model is not None:
            model.eval()

        lines = [ln.strip() for ln in inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_ok = 0
        n_parse_retries = 0

        def _retry_max_tokens(first: int) -> int:
            """If the first pass used a small budget, bump to at least 512; else double (capped) for truncation."""
            lo = max(512, first)
            if lo > first:
                return lo
            return min(2048, max(1024, first * 2))

        with out_path.open("w", encoding="utf-8") as f:
            log.info("Writing %d segments → %s", len(lines), out_path)
            for i, line in enumerate(lines):
                row = json.loads(line)
                seg_id = row.get("id", i)
                log.info("NER segment %s (%d/%d)", seg_id, i + 1, len(lines))
                fr = row.get("fr") or ""
                if not fr.strip():
                    row["terms"] = []
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    continue

                if args.backend == "ollama":
                    user_body = build_prompt(fr)[0]["content"]
                    timeout_eff: float | None = (
                        None if args.ollama_timeout <= 0 else float(args.ollama_timeout)
                    )
                    raw_gen = ""
                    try:
                        raw_gen = ollama_chat(
                            base_url=args.ollama_base_url,
                            model=args.ollama_model,
                            user_content=user_body,
                            timeout=timeout_eff,
                            num_predict=args.max_new_tokens,
                        )
                    except TimeoutError:
                        if timeout_eff is not None:
                            log.warning(
                                "Ollama HTTP timeout (%ss); retrying once with no socket timeout (Ctrl+C to stop).",
                                timeout_eff,
                            )
                            raw_gen = ollama_chat(
                                base_url=args.ollama_base_url,
                                model=args.ollama_model,
                                user_content=user_body,
                                timeout=None,
                                num_predict=args.max_new_tokens,
                            )
                        else:
                            raise SystemExit(
                                "ERROR: Ollama chat timed out even with no HTTP timeout. "
                                "Try a smaller model (e.g. llama3.2:3b), fewer/long prompts, "
                                "or free VRAM/RAM (watch `ollama ps`)."
                            ) from None
                    terms = parse_llm_terms(raw_gen)
                elif args.backend == "unsloth":
                    assert tokenizer is not None and model is not None
                    terms, raw_gen = generate_terms_alpaca(
                        tokenizer,
                        model,
                        fr,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if terms is None:
                        r2 = _retry_max_tokens(args.max_new_tokens)
                        terms, raw_gen = generate_terms_alpaca(
                            tokenizer,
                            model,
                            fr,
                            max_new_tokens=r2,
                            do_sample=False,
                        )
                        if terms is not None:
                            n_parse_retries += 1
                else:
                    assert tokenizer is not None and model is not None
                    terms, raw_gen = generate_terms_hf_chat(
                        tokenizer,
                        model,
                        fr,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if terms is None:
                        r2 = _retry_max_tokens(args.max_new_tokens)
                        terms, raw_gen = generate_terms_hf_chat(
                            tokenizer,
                            model,
                            fr,
                            max_new_tokens=r2,
                            do_sample=False,
                        )
                        if terms is not None:
                            n_parse_retries += 1

                if terms is None:
                    log.warning(
                        "Segment %s: unparseable JSON; empty terms. Raw (truncated): %s",
                        row.get("id", i),
                        (raw_gen[:400] + "…") if len(raw_gen) > 400 else raw_gen,
                    )
                    row["terms"] = []
                else:
                    row["terms"] = terms
                    if terms:
                        n_ok += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

        log.info(
            "Wrote %s (%d rows, %d with non-empty parsed terms, %d JSON parse retries).",
            out_path,
            len(lines),
            n_ok,
            n_parse_retries,
        )
    finally:
        if args.backend in ("huggingface", "unsloth"):
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
