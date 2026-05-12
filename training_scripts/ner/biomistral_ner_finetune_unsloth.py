#!/usr/bin/env python3
"""LoRA SFT for BioMistral NER (Alpaca JSON targets) with Unsloth + TRL."""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

import argparse
import gc
import importlib.util
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path

# Repo root (this script lives under training_scripts/ner/).
ROOT = Path(__file__).resolve().parents[2]
_TOOLS = ROOT / "tools"
_NER_AB = Path(__file__).resolve().parent
for _p in (ROOT, _TOOLS, _NER_AB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Default hierarchical Alpaca ontology JSONL (export with tools/data/export_full_ontology_ner_sft_jsonl.py).
# With --ontology-only and no explicit paths, this file is loaded and split 90/10 train/val in memory.
_DEFAULT_ONTOLOGY_ALPACA_JSONL = ROOT / "data" / "ontology_ner_full_hierarchical_alpaca.jsonl"
# CLI default for --max-seq-length (used with --fit-8gb + --ontology-only to apply fit-8gb-max-seq).
_DEFAULT_MAX_SEQ_LENGTH_CLI = 256

from systems import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

try:
    from unsloth import FastLanguageModel
except ImportError as _e:
    msg = str(_e).lower()
    if "nccl" in msg or "undefined symbol" in msg or "libtorch_cuda" in msg:
        raise SystemExit(
            "ERROR: PyTorch CUDA failed while loading Unsloth (see NCCL note above if shown). "
            "Reinstall torch from https://pytorch.org/get-started/locally/ for the installed CUDA version."
        ) from _e
    raise ImportError(
        "Unsloth / dependencies missing. Install:\n"
        "  pip install unsloth_zoo trl 'datasets>=2.18'\n"
        "  pip install 'unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git'\n"
        "(Use cu121/cu124/cu128 etc. to match the PyTorch CUDA build.)"
    ) from _e

# Text-only LoRA: avoid importing torchvision. HF sets ``_torchvision_available`` from metadata even when
# ``import torchvision`` crashes with torch/torchvision ABI mismatch (e.g. RuntimeError: torchvision::nms).
try:
    import transformers.utils.import_utils as _hf_import_utils

    _hf_import_utils._torchvision_available = False
except Exception:
    pass


def _import_torch_cuda_first() -> None:
    """Fail fast when CUDA/NCCL libs mismatch (e.g. ``undefined symbol: ncclDevCommDestroy``)."""
    try:
        import torch  # noqa: F401
    except Exception as e:
        chain = f"{e!s} {getattr(e, '__cause__', '')!s}"
        low = chain.lower()
        if "nccl" in low or "undefined symbol" in low or "libtorch_cuda" in low:
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise SystemExit(
                "ERROR: PyTorch failed to load CUDA (NCCL / symbol mismatch). Common fixes:\n\n"
                "1) Wrong NCCL on PYTHONPATH/LD path — pip's ``nvidia-nccl-cu12`` often conflicts with "
                "CUDA 13 builds (``torch …+cu130``). Try removing it so PyTorch uses its bundled NCCL:\n"
                "     pip uninstall -y nvidia-nccl-cu12\n"
                "   Then: python -c \"import torch; print(torch.__version__)\"\n\n"
                "2) Python " + py + ": PyTorch may have **no** wheels on ``cu124`` — use the CUDA index "
                "that matches the CUDA/driver stack (driver shows CUDA 13 → often ``cu130``):\n"
                "     pip install --force-reinstall torch torchvision torchaudio \\\n"
                "       --index-url https://download.pytorch.org/whl/cu130\n\n"
                "3) Prefer versions from https://pytorch.org/get-started/locally/ (same cu* as the GPU stack).\n\n"
                "4) ``unsloth-zoo`` may pin ``torch<2.11``; if pip warns, use torch 2.10.x with the same cu* URL "
                "or upgrade Unsloth when supported.\n\n"
                f"Underlying error: {e!r}"
            ) from e
        raise


_import_torch_cuda_first()


def _patch_datasets_fingerprint_if_py314() -> None:
    """HF ``datasets`` fingerprint pickle breaks on Python 3.14 + dill (same workaround as token HF trainers)."""
    if sys.version_info < (3, 14):
        return
    import hashlib

    import datasets.arrow_dataset as ads
    import datasets.fingerprint as fp
    import pyarrow as pa

    _orig = fp.generate_fingerprint

    def _safe_generate_fingerprint(dataset):
        try:
            return _orig(dataset)
        except TypeError as e:
            if "_batch_setitems" not in str(e):
                raise
            table = dataset.data.table
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()

    fp.generate_fingerprint = _safe_generate_fingerprint  # type: ignore[assignment]
    ads.generate_fingerprint = _safe_generate_fingerprint  # type: ignore[assignment]


def _dataset_from_text_rows(rows: list[dict[str, str]]):
    """Avoid ``Dataset.from_list`` on Py 3.14 (same fingerprint issue as CamemBERT fine-tune)."""
    from datasets import Dataset, Features, Value

    return Dataset.from_dict(
        {"text": [r["text"] for r in rows]},
        features=Features({"text": Value("string")}),
    )


def _log_ontology_token_lengths_vs_cap(
    tokenizer,
    ontology_rows: list[dict[str, str]],
    max_seq_length: int,
    *,
    n_sample: int = 24,
) -> None:
    """Warn when ontology Alpaca rows exceed ``max_seq_length`` (common cause of ~0 CE + tiny grad_norm)."""
    if not ontology_rows:
        return
    n = min(len(ontology_rows), max(1, int(n_sample)))
    lengths: list[int] = []
    over = 0
    for i in range(n):
        ln = len(tokenizer.encode(ontology_rows[i]["text"], add_special_tokens=True))
        lengths.append(ln)
        if ln > max_seq_length:
            over += 1
    mean_ln = sum(lengths) / len(lengths)
    mx = max(lengths)
    print(
        f"[train] ontology token lengths (sample n={n}): min/mean/max = "
        f"{min(lengths)}/{mean_ln:.0f}/{mx}; max_seq_length={max_seq_length}",
        file=sys.stderr,
    )
    if over:
        print(
            f"[train] WARNING: {over}/{n} sampled ontology rows exceed max_seq_length. "
            "Truncation often removes the JSON completion; train/eval loss can stay ~0 with tiny grad_norm. "
            "Raise --max-seq-length and (with --fit-8gb) increase --fit-8gb-max-seq if VRAM allows, "
            "or train without --fit-8gb on a larger GPU.",
            file=sys.stderr,
        )


def _training_optim_name(requested: str) -> str:
    """Pick a Transformers ``optim`` id; fused AdamW when CUDA + torch>=2 (faster on many GPUs)."""
    req = (requested or "auto").strip().lower()
    if req not in ("auto", ""):
        if req == "adamw_torch_fused":
            import torch as _t

            if not _t.cuda.is_available():
                return "adamw_torch"
            try:
                ver = tuple(int(x) for x in _t.__version__.split(".")[:2])
            except (ValueError, AttributeError):
                ver = (0, 0)
            if ver < (2, 0):
                return "adamw_torch"
        return req

    import torch as _t

    if not _t.cuda.is_available():
        return "adamw_torch"
    try:
        ver = tuple(int(x) for x in _t.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        ver = (0, 0)
    if ver >= (2, 0):
        return "adamw_torch_fused"
    return "adamw_torch"


def _python_dev_headers_present() -> bool:
    """Triton builds a tiny helper with gcc and needs ``Python.h`` (Fedora: ``python3-devel``)."""
    import sysconfig

    return (Path(sysconfig.get_path("include")) / "Python.h").is_file()


def _apply_unsloth_torch_rms_fallback(reason: str) -> None:
    """Replace Unsloth's Triton RMSNorm with pure PyTorch (slower, no gcc/Python.h)."""
    import torch

    def _fn(layernorm, X, gemma: bool = False):
        input_dtype = X.dtype
        x = X.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        eps = (
            layernorm.variance_epsilon
            if hasattr(layernorm, "variance_epsilon")
            else getattr(layernorm, "eps", 1e-6)
        )
        x = x * torch.rsqrt(variance + eps)
        if gemma:
            x = x * (layernorm.weight.to(torch.float32) + 1.0)
        else:
            x = x * layernorm.weight.to(torch.float32)
        return x.to(input_dtype)

    import unsloth.kernels.rms_layernorm as _rms_mod
    import unsloth.models.llama as _ullama

    _rms_mod.fast_rms_layernorm = _fn
    _ullama.fast_rms_layernorm = _fn
    try:
        import unsloth.models.mistral as _umistral

        _umistral.fast_rms_layernorm = _fn
    except Exception:
        pass
    try:
        import unsloth.kernels as _ukern

        if hasattr(_ukern, "fast_rms_layernorm"):
            _ukern.fast_rms_layernorm = _fn
    except Exception:
        pass
    print(f"Unsloth: PyTorch RMSNorm fallback enabled ({reason}).", file=sys.stderr)


def _dir_has_hf_model_weights(d: Path) -> bool:
    """True if ``d`` looks like a HF model checkpoint (not tokenizer-only)."""
    if not d.is_dir():
        return False
    if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
        return True
    return any(d.glob("model-*.safetensors"))


def _merge_lora_peft_fallback(
    base_model_id: str,
    lora_dir: Path,
    merged_dir: Path,
    tokenizer,
) -> None:
    """When Unsloth ``save_pretrained_merged`` writes only tokenizers, merge via PEFT (CPU by default)."""
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    preference = os.environ.get("UNSLOTH_MERGE_DEVICE", "cpu").strip().lower()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if preference == "cuda" and torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"

    print(
        f"Merging LoRA → dense weights via PEFT (device_map={device_map!r}; "
        f"UNSLOTH_MERGE_DEVICE=cuda to try GPU). Base={base_model_id!r}",
        file=sys.stderr,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    merged_model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


_DEFAULT_BRAT = ROOT / "data" / "QUAERO_FrenchMed" / "corpus" / "train" / "EMEA"
_BASE_MODEL = "BioMistral/BioMistral-7B"
# Smaller instruct LM for ~8 GB GPUs when using ``--fit-8gb`` (default base swap only).
_BASE_MODEL_FIT_8GB = "Qwen/Qwen2.5-3B-Instruct"


def _default_lora_dirname(base_model_id: str) -> str:
    """Default ``models/<name>`` folder for LoRA checkpoints (when ``--lora-dir`` is omitted)."""
    low = (base_model_id or "").lower()
    if "biomistral" in low:
        return "biomistral-ner-lora"
    if "qwen" in low:
        return "qwen-finetuned-on-ontology-lora"
    tail = base_model_id.rsplit("/", 1)[-1].lower().replace(".", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", tail).strip("-") or "base"
    return f"{slug}-ner-lora"[:120]


def _default_merged_dirname(base_model_id: str) -> str:
    """Default ``models/<name>`` folder for merged weights (when ``--merged-dir`` is omitted)."""
    low = (base_model_id or "").lower()
    if "biomistral" in low:
        return "biomistral-ner-merged"
    if "qwen" in low:
        return "qwen-finetuned-on-ontology-merged"
    tail = base_model_id.rsplit("/", 1)[-1].lower().replace(".", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", tail).strip("-") or "base"
    return f"{slug}-ner-merged"[:120]


def _base_lm_needs_tight_gpu_pin(model_name: str) -> bool:
    """Heuristic: 7B+ / BioMistral need explicit cuda:0 pinning on tight VRAM; 3B-class can load without."""
    n = (model_name or "").lower()
    if "biomistral" in n:
        return True
    return any(x in n for x in ("7b", "8b", "13b", "70b", "72b"))


ALPACA_INSTRUCTION = (
    "Extract all medical terms (disorders, drugs, procedures) from this French medical text. "
    "Return only a JSON list of strings."
)


def _load_plot_overlay():
    p = ROOT / "tools" / "plot_training_curves.py"
    spec = importlib.util.spec_from_file_location("plot_training_curves", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.plot_training_curves_overlay


def merge_bio_to_entity_strings(tokens: list[str], tags: list[str]) -> list[str]:
    """Reconstruct entity strings from BIO tags (DISO/CHEM/PROC)."""
    out: list[str] = []
    cur: list[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        if tag.startswith("B-"):
            if cur:
                out.append(" ".join(cur))
            cur = [tok]
        elif tag.startswith("I-"):
            if cur:
                cur.append(tok)
            else:
                cur = [tok]
    if cur:
        out.append(" ".join(cur))
    return out


def to_alpaca_text(fr_sentence: str, json_list_str: str) -> str:
    return (
        "### Instruction:\n"
        f"{ALPACA_INSTRUCTION}\n\n"
        "### Input:\n"
        f"{fr_sentence}\n\n"
        "### Response:\n"
        f"{json_list_str}"
    )


def split_prompt_and_gold(full_text: str) -> tuple[str, list[str]]:
    marker = "### Response:\n"
    i = full_text.rfind(marker)
    if i == -1:
        raise ValueError("Alpaca example missing '### Response:' marker")
    prefix = full_text[: i + len(marker)]
    tail = full_text[i + len(marker) :].strip()
    gold = json.loads(tail)
    if not isinstance(gold, list):
        raise ValueError("Gold response is not a JSON list")
    strings: list[str] = []
    for x in gold:
        if isinstance(x, str):
            s = x.strip()
            if s:
                strings.append(s)
        elif isinstance(x, dict):
            w = x.get("fr") or x.get("word")
            if isinstance(w, str) and w.strip():
                strings.append(w.strip())
    return prefix, strings


def _load_ontology_sft_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise SystemExit(f"Ontology SFT line missing 'text' key: {path}")
            rows.append({"text": str(obj["text"])})
    return rows


def _train_val_split_rows(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    if n <= 1:
        return list(rows), list(rows)
    n_train = max(1, int(0.9 * n))
    if n_train >= n:
        n_train = n - 1
    train = [rows[i] for i in idx[:n_train]]
    val = [rows[i] for i in idx[n_train:]]
    return train, val


def _eval_fr_multiset_f1_batch(
    model,
    tokenizer,
    row_dicts: list[dict[str, str]],
    *,
    max_seq_length: int,
    max_new_tokens: int,
    eval_cap: int,
    multiset_f1_fn,
    split_prompt_and_gold_fn,
    parse_llm_terms_fn,
) -> tuple[float, int]:
    """Mean multiset F1 over ``fr`` spans (gold vs generated JSON). Returns (mean_f1, n_used)."""
    import torch

    if not row_dicts:
        return 0.0, 0
    proc = tokenizer
    cap = min(max(1, eval_cap), len(row_dicts))
    scores: list[float] = []
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for i in range(cap):
            full = row_dicts[i]["text"]
            prefix, gold_list = split_prompt_and_gold_fn(full)
            inputs = proc(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(proc, "pad_token_id", None) or proc.eos_token_id,
                use_cache=False,
            )
            in_len = inputs["input_ids"].shape[1]
            gen_text = proc.decode(out_ids[0, in_len:], skip_special_tokens=True)
            pt = parse_llm_terms_fn(gen_text)
            pred_terms = pt if pt else []
            scores.append(multiset_f1_fn(gold_list, pred_terms))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model.train()
    return (sum(scores) / len(scores)) if scores else 0.0, cap


def _train_val_split_recs(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = len(records)
    if n <= 1:
        return list(records), list(records)
    n_train = max(1, int(0.9 * n))
    if n_train >= n:
        n_train = n - 1
    train_recs = [records[i] for i in idx[:n_train]]
    val_recs = [records[i] for i in idx[n_train:]]
    return train_recs, val_recs


def main() -> None:
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit("Install trl and datasets: pip install trl datasets") from e

    from collections import Counter

    from quaero_brat_reader import ID2LABEL, load_quaero_brat
    from biomistral_prompt_ner import parse_llm_terms

    def multiset_f1(gold: list[str], pred_terms: list[dict]) -> float:
        pred_strings = [
            str(t["word"]).strip() for t in pred_terms if t and str(t.get("word", "")).strip()
        ]
        cg = Counter(gold)
        cp = Counter(pred_strings)
        inter = sum((cg & cp).values())
        if inter == 0:
            return 0.0
        p = inter / max(sum(cp.values()), 1)
        r = inter / max(sum(cg.values()), 1)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    p = argparse.ArgumentParser(description="Unsloth LoRA SFT for BioMistral JSON-list NER.")
    p.add_argument("--brat-dir", type=Path, default=_DEFAULT_BRAT, help="QUAERO BRAT folder (.txt + .ann).")
    p.add_argument(
        "--lora-dir",
        type=Path,
        default=None,
        help=(
            "LoRA adapter + checkpoints. Default: models/biomistral-ner-lora for BioMistral bases, "
            "models/qwen-finetuned-on-ontology-lora for Qwen bases, else derived from --base-model."
        ),
    )
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help=(
            "Merged full-precision export. Default: models/biomistral-ner-merged for BioMistral, "
            "models/qwen-finetuned-on-ontology-merged for Qwen, else derived from --base-model."
        ),
    )
    p.add_argument("--base-model", type=str, default=_BASE_MODEL, help="HF base model id.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-combine-medline",
        action="store_true",
        help="Do not also load sibling MEDLINE/ (same default as ``quaero_brat_reader.load_quaero_brat``).",
    )
    p.add_argument(
        "--limit-records",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N BRAT sentences after shuffle (debug / smoke test).",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the base model in bf16/fp16 instead of 4-bit (more VRAM). Use when bitsandbytes fails "
        "(e.g. missing libnvJitLink.so.* on the loader path).",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="Cap context at 192 (384 with --fit-8gb and default --fit-8gb-max-seq), smaller LoRA (r=8), and minimal eval generation.",
    )
    p.add_argument(
        "--fit-8gb",
        action="store_true",
        dest="fit_8gb",
        help=(
            "Preset for ~8 GB GPUs: if --base-model is still the default BioMistral-7B, switch to "
            f"{_BASE_MODEL_FIT_8GB}; enable --low-vram; cap context at --fit-8gb-max-seq (default 384); "
            "lighter eval. Large-model VRAM pinning at load applies only to 7B-class bases."
        ),
    )
    p.add_argument(
        "--fit-8gb-max-seq",
        type=int,
        default=384,
        metavar="L",
        help=(
            "With --fit-8gb, low-VRAM cap for max_seq_length (default 384 so typical ontology Alpaca rows "
            "fit; reduce if OOM). Rows can exceed 900 tokens — raise further only if VRAM allows, or drop "
            "--fit-8gb on a larger GPU."
        ),
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        metavar="L",
        help=(
            "Training / eval context cap (lower = less VRAM). Default 256. Omitted with "
            "--fit-8gb --ontology-only lets the run use --fit-8gb-max-seq (384 by default) so long "
            "Alpaca lines are not truncated; pass this flag explicitly to cap lower."
        ),
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=16,
        metavar="R",
        help="LoRA rank (default 16; use 8 with --low-vram or on GPU OOM).",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        metavar="A",
        help="LoRA alpha (default 32; often 2× rank).",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (0 saves a bit of memory with Unsloth fast path).",
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Micro-batch per GPU (default 1 for tight VRAM).",
    )
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Eval micro-batch per GPU (default 1).",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        metavar="N",
        help="Gradient accumulation steps (default 16 → effective batch 16 with batch size 1).",
    )
    p.add_argument(
        "--force-torch-rms",
        action="store_true",
        help="Use PyTorch RMSNorm instead of Unsloth Triton kernels (slower; avoids needing Python.h).",
    )
    p.add_argument(
        "--allow-triton-rms",
        action="store_true",
        help="Keep Unsloth Triton RMSNorm (requires Python dev headers for first-time Triton driver build).",
    )
    p.add_argument(
        "--export-merged-only",
        action="store_true",
        help="Skip training: merge existing adapter in --lora-dir into --merged-dir using PEFT (needs RAM for 7B).",
    )
    p.add_argument(
        "--ontology-sft-jsonl",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Alpaca JSONL from tools/data/export_full_ontology_ner_sft_jsonl.py (one {\"text\": ...} per line). "
            "Pass multiple times to concatenate several files. Mixed with QUAERO unless --ontology-only. "
            "When used alone, rows are split 90%% train / 10%% val. Use --ontology-train-jsonl / "
            "--ontology-val-jsonl for fixed train/dev/test files instead."
        ),
    )
    p.add_argument(
        "--ontology-train-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help="Ontology train split (requires --ontology-val-jsonl). Omit both train and val to use "
        "data/ontology_ner_full_hierarchical_alpaca.jsonl with an in-memory 90/10 split when that file exists.",
    )
    p.add_argument(
        "--ontology-val-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help="Ontology validation / dev split (requires --ontology-train-jsonl). "
        "Omit both train and val to use the default single-file flow (see --ontology-train-jsonl).",
    )
    p.add_argument(
        "--ontology-test-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help="Ontology test split — evaluated once after training. Optional; pass explicitly when you "
        "have a held-out JSONL (there is no default test file when using the single canonical Alpaca export).",
    )
    p.add_argument(
        "--ontology-test-eval-cap",
        type=int,
        default=512,
        metavar="N",
        help="Max test examples for post-training multiset F1 (default 512).",
    )
    p.add_argument(
        "--ontology-only",
        action="store_true",
        help="Skip QUAERO BRAT; train and validate on ontology JSONL (--ontology-sft-jsonl and/or default "
        "data/ontology_ner_full_hierarchical_alpaca.jsonl with 90/10 split, or explicit train/val paths).",
    )
    p.add_argument(
        "--full-ontology-finetune",
        action="store_true",
        help=(
            "Ontology SFT preset: requires --ontology-only. Drops any --max-steps cap so training uses the "
            "full --num-train-epochs budget (default 2) over the train split. If --fast was also passed, it is "
            "ignored. Typical: --fit-8gb --ontology-only --full-ontology-finetune --resume-from-checkpoint auto"
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Shorter runs: cap --num-train-epochs at 1, enable --no-eval-ner-f1, set --logging-steps to 50 "
            "when still at default. Add --max-steps N for a shorter smoke run. "
            "(DataLoader workers stay 0: Unsloth/TRL collators are not picklable for multiprocessing.)"
        ),
    )
    p.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        metavar="E",
        help="Training epochs (default 2). Ignored when --max-steps > 0.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N optimizer updates (Trainer max_steps; overrides the epoch budget when set).",
    )
    p.add_argument(
        "--optim",
        type=str,
        default="auto",
        metavar="NAME",
        help="Optimizer: auto (adamw_torch_fused on CUDA+torch>=2 when available, else adamw_torch), "
        "adamw_torch, adamw_torch_fused, adamw_hf.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        metavar="LR",
        help="Peak LR (default 2e-4).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        metavar="WD",
        help="AdamW weight decay (default 0.01).",
    )
    p.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the validation set each epoch (much faster; no eval_loss / eval_ner_f1). "
        "Checkpoints still save per --save-strategy (default: each epoch under --lora-dir). "
        "load_best_model_at_end is disabled (no metric to pick best).",
    )
    p.add_argument(
        "--no-eval-ner-f1",
        action="store_true",
        help="Keep per-epoch validation loss but skip the extra generate+multiset-F1 pass (saves time each epoch).",
    )
    p.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        metavar="N",
        help="Must stay 0 for this Unsloth SFT entrypoint (TRL collator + padding-free hooks are not picklable; "
        "workers>0 raises PicklingError).",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        metavar="N",
        help="Log every N steps (default 10; increase to reduce logging overhead).",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        metavar="N",
        help="LR linear warmup steps (default 10).",
    )
    p.add_argument(
        "--save-strategy",
        choices=("epoch", "steps", "no"),
        default="epoch",
        help="When to write checkpoint-* under --lora-dir (default epoch). ``no`` = only final save at end.",
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=500,
        metavar="N",
        help="With --save-strategy steps, save every N global steps (default 500).",
    )
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=4,
        metavar="N",
        help="Rotate checkpoints: keep at most N checkpoint-* dirs (default 4). Use 0 for no limit.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="PATH|auto",
        help="Resume: path to checkpoint-* dir, or ``auto`` / ``true`` for latest checkpoint under --lora-dir.",
    )
    args = p.parse_args()

    max_seq_length_explicit = args.max_seq_length is not None
    if args.max_seq_length is None:
        args.max_seq_length = _DEFAULT_MAX_SEQ_LENGTH_CLI

    if int(args.dataloader_num_workers) != 0:
        print(
            "[train] forcing dataloader_num_workers=0 (Unsloth padding-free + TRL collator cannot be "
            "pickled for DataLoader worker processes).",
            file=sys.stderr,
        )
        args.dataloader_num_workers = 0

    if args.full_ontology_finetune:
        if not args.ontology_only:
            raise SystemExit("--full-ontology-finetune requires --ontology-only")
        if args.max_steps is not None and int(args.max_steps) > 0:
            print(
                "[train] --full-ontology-finetune: clearing --max-steps (training by --num-train-epochs).",
                file=sys.stderr,
            )
            args.max_steps = None
        if args.fast:
            print(
                "[train] --full-ontology-finetune: ignoring --fast for full epoch-based training.",
                file=sys.stderr,
            )
            args.fast = False
        print(
            "[train] --full-ontology-finetune: epoch-based run (no max_steps cap); "
            f"num_train_epochs={args.num_train_epochs} — change with --num-train-epochs",
            file=sys.stderr,
        )

    if args.fast:
        args.num_train_epochs = min(float(args.num_train_epochs), 1.0)
        args.no_eval_ner_f1 = True
        if int(args.logging_steps) <= 10:
            args.logging_steps = 50
        print(
            "[train] --fast: epochs≤1, no eval multiset F1, "
            f"logging_steps={args.logging_steps}",
            file=sys.stderr,
        )

    fit_8gb = bool(getattr(args, "fit_8gb", False))
    if fit_8gb:
        if args.base_model == _BASE_MODEL:
            args.base_model = _BASE_MODEL_FIT_8GB
        args.low_vram = True

    if args.lora_dir is None:
        args.lora_dir = ROOT / "models" / _default_lora_dirname(args.base_model)
    if args.merged_dir is None:
        args.merged_dir = ROOT / "models" / _default_merged_dirname(args.base_model)

    if args.force_torch_rms:
        _apply_unsloth_torch_rms_fallback("CLI --force-torch-rms")
    elif not args.allow_triton_rms and not _python_dev_headers_present():
        _apply_unsloth_torch_rms_fallback(
            "Python.h not found — install ``python3-devel`` (Fedora) for fast Triton RMS, or pass ``--force-torch-rms``"
        )

    lora_r = max(1, int(args.lora_r))
    lora_alpha = max(1, int(args.lora_alpha))
    lora_dropout = float(args.lora_dropout)
    max_seq_user = max(64, int(args.max_seq_length))
    if args.low_vram:
        seq_cap = max(64, int(args.fit_8gb_max_seq)) if fit_8gb else 192
        max_seq_length_run = min(max_seq_user, seq_cap)
        if (
            fit_8gb
            and args.ontology_only
            and max_seq_length_run < seq_cap
            and not max_seq_length_explicit
        ):
            max_seq_length_run = seq_cap
            print(
                f"[fit-8gb] ontology-only: using max_seq_length={max_seq_length_run} "
                f"(--fit-8gb-max-seq cap; pass --max-seq-length to cap lower)",
                file=sys.stderr,
            )
        lora_r = min(lora_r, 8)
        lora_alpha = min(lora_alpha, 16)
        lora_dropout = 0.0
    else:
        max_seq_length_run = max_seq_user

    if fit_8gb:
        print(
            f"[fit-8gb] base_model={args.base_model} max_seq_length={max_seq_length_run} "
            f"fit_8gb_max_seq={max(64, int(args.fit_8gb_max_seq))} lora_r={lora_r}",
            file=sys.stderr,
        )

    _patch_datasets_fingerprint_if_py314()

    brat_dir = args.brat_dir if args.brat_dir.is_absolute() else ROOT / args.brat_dir
    if not args.ontology_only and not brat_dir.is_dir():
        raise SystemExit(
            f"QUAERO BRAT not found: {brat_dir}\n"
            "Use --ontology-only with hierarchical ontology JSONL (see docs/mistral_instruct_Ontology-Fine-tuning.md), "
            "or install the QUAERO EMEA corpus and pass --brat-dir."
        )
    lora_dir = args.lora_dir if args.lora_dir.is_absolute() else ROOT / args.lora_dir
    merged_dir = args.merged_dir if args.merged_dir.is_absolute() else ROOT / args.merged_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] output_dir (checkpoints + adapter): {lora_dir.resolve()}", file=sys.stderr)

    if args.export_merged_only:
        from transformers import AutoTokenizer

        if not lora_dir.is_dir():
            raise SystemExit(f"--export-merged-only: LoRA dir not found: {lora_dir}")
        tok = AutoTokenizer.from_pretrained(str(lora_dir))
        try:
            _merge_lora_peft_fallback(args.base_model, lora_dir, merged_dir, tok)
        except Exception as ex:
            raise SystemExit(
                "PEFT merge failed (needs enough CPU RAM for 7B, or try UNSLOTH_MERGE_DEVICE=cuda).\n"
                f"{ex}"
            ) from ex
        if not _dir_has_hf_model_weights(merged_dir):
            raise SystemExit(f"Merge produced no weight files under {merged_dir}")
        print(f"Merged HF weights: {merged_dir}", file=sys.stderr)
        return

    def _resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else ROOT / p

    ontology_paths = list(args.ontology_sft_jsonl) if args.ontology_sft_jsonl else []
    otr = _resolve(args.ontology_train_jsonl)
    oval = _resolve(args.ontology_val_jsonl)
    ote = _resolve(args.ontology_test_jsonl)

    if (
        args.ontology_only
        and not ontology_paths
        and args.ontology_train_jsonl is None
        and args.ontology_val_jsonl is None
        and _DEFAULT_ONTOLOGY_ALPACA_JSONL.is_file()
    ):
        ontology_paths = [_DEFAULT_ONTOLOGY_ALPACA_JSONL]
        print(
            f"[train] --ontology-only: using {_DEFAULT_ONTOLOGY_ALPACA_JSONL.name} "
            "(90/10 train/val split in memory).",
            file=sys.stderr,
        )

    if args.ontology_only and not ontology_paths and not (otr and oval):
        raise SystemExit(
            "--ontology-only needs ontology data: pass --ontology-sft-jsonl and/or "
            "--ontology-train-jsonl + --ontology-val-jsonl, or place\n"
            f"  {_DEFAULT_ONTOLOGY_ALPACA_JSONL}\n"
            "Create it with:\n"
            "  PYTHONPATH=. python tools/data/export_full_ontology_ner_sft_jsonl.py --prompt-style alpaca --out data/ontology_ner_full_hierarchical_alpaca.jsonl\n"
            "Optionally split with tools/data/split_ontology_sft_jsonl.py if you prefer separate train/val files "
            "and pass them via --ontology-train-jsonl / --ontology-val-jsonl."
        )

    if (otr or oval) and not (otr and oval):
        raise SystemExit("--ontology-train-jsonl and --ontology-val-jsonl must be used together.")
    if (otr and oval) and ontology_paths:
        raise SystemExit(
            "Do not combine --ontology-sft-jsonl with --ontology-train-jsonl/--ontology-val-jsonl "
            "(pick automatic split or pre-split files)."
        )

    on_train_rows: list[dict[str, str]] = []
    on_val_rows: list[dict[str, str]] = []
    on_test_rows: list[dict[str, str]] = []

    def _bad_path_hint(p: Path) -> str:
        s = str(p).replace("\\", "/")
        if "PATH/TO/" in s.upper() or "/PATH/" in s.upper():
            return (
                "\nHint: use real JSONL paths, or omit --ontology-train-jsonl and "
                "--ontology-val-jsonl to use data/ontology_ner_full_hierarchical_alpaca.jsonl when present."
            )
        return ""

    if otr and oval:
        if not otr.is_file():
            raise SystemExit(f"--ontology-train-jsonl not found: {otr}{_bad_path_hint(otr)}")
        if not oval.is_file():
            raise SystemExit(f"--ontology-val-jsonl not found: {oval}{_bad_path_hint(oval)}")
        try:
            on_train_rows = _load_ontology_sft_jsonl(otr)
            on_val_rows = _load_ontology_sft_jsonl(oval)
        except (json.JSONDecodeError, OSError, UnicodeError) as ex:
            raise SystemExit(f"Failed reading ontology train/val JSONL: {ex}") from ex
        if ote:
            if not ote.is_file():
                raise SystemExit(f"--ontology-test-jsonl not found: {ote}{_bad_path_hint(ote)}")
            try:
                on_test_rows = _load_ontology_sft_jsonl(ote)
            except (json.JSONDecodeError, OSError, UnicodeError) as ex:
                raise SystemExit(f"Failed reading ontology test JSONL: {ex}") from ex
    else:
        on_all_rows: list[dict[str, str]] = []
        for rel in ontology_paths:
            p = rel if rel.is_absolute() else ROOT / rel
            if not p.is_file():
                raise SystemExit(f"--ontology-sft-jsonl not found: {p}")
            try:
                on_all_rows.extend(_load_ontology_sft_jsonl(p))
            except (json.JSONDecodeError, OSError, UnicodeError) as ex:
                raise SystemExit(f"Failed reading ontology SFT JSONL {p}: {ex}") from ex

        on_train_rows, on_val_rows = (
            _train_val_split_rows(on_all_rows, args.seed + 1337) if on_all_rows else ([], [])
        )

    def record_to_text(rec: dict) -> str:
        tokens = rec["tokens"]
        tags = [ID2LABEL[int(i)] for i in rec["ner_tags"]]
        fr = " ".join(tokens)
        entities = merge_bio_to_entity_strings(tokens, tags)
        payload = json.dumps(entities, ensure_ascii=False)
        return to_alpaca_text(fr, payload)

    quaero_train_rows: list[dict[str, str]] = []
    quaero_val_rows: list[dict[str, str]] = []
    if not args.ontology_only:
        records = load_quaero_brat(brat_dir, combine_medline=not args.no_combine_medline)
        if args.limit_records is not None:
            lim = max(1, int(args.limit_records))
            records = records[:lim]
        train_recs, val_recs = _train_val_split_recs(records, args.seed)
        quaero_train_rows = [{"text": record_to_text(r)} for r in train_recs]
        quaero_val_rows = [{"text": record_to_text(r)} for r in val_recs]

    train_row_dicts = quaero_train_rows + on_train_rows
    val_row_dicts = quaero_val_rows + on_val_rows
    if not train_row_dicts:
        raise SystemExit(
            "No training rows: use QUAERO (default) and/or --ontology-sft-jsonl "
            "(see tools/data/export_full_ontology_ner_sft_jsonl.py)."
        )

    train_ds = _dataset_from_text_rows(train_row_dicts)
    val_ds = _dataset_from_text_rows(val_row_dicts)

    import torch

    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    max_seq_length = max_seq_length_run
    load_in_4bit = not args.no_4bit
    if load_in_4bit:
        dtype = None
    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_pretrained_kw: dict = {}
    # bitsandbytes 4-bit: no partial CPU/disk offload → pin full model on cuda:0 on small GPUs.
    # Only pin when enough VRAM is *actually free* before load; otherwise HF/Unsloth peaks OOM and
    # a same-process "retry" cannot reclaim the failed first load's allocations.
    min_free_mib_4bit = 3072  # ~3 GiB headroom for 3B 4-bit + Unsloth init spikes on ~8 GB GPUs
    min_free_mib_fp = 2048
    free_mib = 99_999
    total_mib = 99_999
    if torch.cuda.is_available():
        try:
            free_b, tot_b = torch.cuda.mem_get_info()
            free_mib = int(free_b // (1024 * 1024))
            total_mib = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            pass
        need = min_free_mib_4bit if load_in_4bit else min_free_mib_fp
        if free_mib < need:
            raise SystemExit(
                f"Insufficient free GPU memory ({free_mib} MiB free; need about >= {need} MiB before "
                f"loading the base model). Another process is using the GPU — run  nvidia-smi  and "
                f"stop other python / ML jobs (or reboot). Then in a **new** shell:\n"
                f"  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                f"  PYTHONPATH=. python training_scripts/ner/biomistral_ner_finetune_unsloth.py ...\n"
                f"(Total GPU: ~{total_mib} MiB.)"
            )

    if load_in_4bit and torch.cuda.is_available():
        if total_mib < 11_000 or _base_lm_needs_tight_gpu_pin(args.base_model):
            load_pretrained_kw["device_map"] = {"": 0}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        **load_pretrained_kw,
    )

    _log_ontology_token_lengths_vs_cap(tokenizer, on_train_rows, max_seq_length)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Unsloth keeps compute weights in bf16 when supported; mixed precision must match (fp16 + bf16 model → TypeError).
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    num_train_epochs = max(0.01, float(args.num_train_epochs))
    max_steps_arg = getattr(args, "max_steps", None)
    max_steps: int | None = None
    if max_steps_arg is not None and int(max_steps_arg) > 0:
        max_steps = int(max_steps_arg)

    skip_eval = bool(args.skip_evaluation)
    lr = float(args.learning_rate)
    wd = float(args.weight_decay)
    optim_resolved = _training_optim_name(args.optim)

    cfg_kwargs: dict = {
        "output_dir": str(lora_dir),
        "per_device_train_batch_size": max(1, int(args.per_device_train_batch_size)),
        "per_device_eval_batch_size": max(1, int(args.per_device_eval_batch_size)),
        "gradient_accumulation_steps": max(1, int(args.gradient_accumulation_steps)),
        "learning_rate": lr,
        "weight_decay": wd,
        "lr_scheduler_type": "cosine",
        "warmup_steps": max(0, int(args.warmup_steps)),
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": max(1, int(args.logging_steps)),
        "report_to": "none",
        "seed": args.seed,
    }
    if max_steps is not None:
        cfg_kwargs["max_steps"] = max_steps
    cfg_kwargs["num_train_epochs"] = num_train_epochs
    if skip_eval:
        cfg_kwargs["eval_strategy"] = "no"
        cfg_kwargs["load_best_model_at_end"] = False
    else:
        cfg_kwargs["eval_strategy"] = "epoch"
        cfg_kwargs["load_best_model_at_end"] = True
        cfg_kwargs["metric_for_best_model"] = "eval_loss"
        cfg_kwargs["greater_is_better"] = False

    save_strat = (args.save_strategy or "epoch").strip().lower()
    if save_strat == "steps":
        cfg_kwargs["save_strategy"] = "steps"
        cfg_kwargs["save_steps"] = max(1, int(args.save_steps))
    elif save_strat == "no":
        cfg_kwargs["save_strategy"] = "no"
    else:
        cfg_kwargs["save_strategy"] = "epoch"

    if not skip_eval and save_strat == "steps":
        # TrainingArguments: load_best_model_at_end requires eval and save strategies to align.
        cfg_kwargs["eval_strategy"] = "steps"
        cfg_kwargs["eval_steps"] = int(cfg_kwargs["save_steps"])

    stl = max(0, int(args.save_total_limit))
    if stl > 0:
        cfg_kwargs["save_total_limit"] = stl

    dl_workers = max(0, int(args.dataloader_num_workers))
    if dl_workers:
        cfg_kwargs["dataloader_num_workers"] = dl_workers

    try:
        import torch.distributed as dist

        _ws = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        _ws = int(os.environ.get("WORLD_SIZE", "1"))
    _ws = max(1, _ws)
    _eff = cfg_kwargs["per_device_train_batch_size"] * cfg_kwargs["gradient_accumulation_steps"] * _ws
    print(
        f"[train] per_device_train_batch_size={cfg_kwargs['per_device_train_batch_size']} "
        f"grad_accum={cfg_kwargs['gradient_accumulation_steps']} world_size={_ws} "
        f"→ effective train batch ≈ {_eff} (increase micro-batch for speed if VRAM allows; reduce grad_accum to match).",
        file=sys.stderr,
    )

    sig = inspect.signature(SFTConfig.__init__)
    if "completion_only_loss" in sig.parameters:
        cfg_kwargs["completion_only_loss"] = True
    elif "train_on_responses_only" in sig.parameters:
        cfg_kwargs["train_on_responses_only"] = True
    if "response_template" in sig.parameters:
        cfg_kwargs["response_template"] = "\n### Response:\n"
    if "max_seq_length" in sig.parameters:
        cfg_kwargs["max_seq_length"] = max_seq_length
    if "packing" in sig.parameters:
        cfg_kwargs["packing"] = False

    if "eval_strategy" not in sig.parameters and "evaluation_strategy" in sig.parameters:
        cfg_kwargs["evaluation_strategy"] = cfg_kwargs.pop("eval_strategy")

    if dl_workers and torch.cuda.is_available() and "dataloader_pin_memory" in sig.parameters:
        cfg_kwargs["dataloader_pin_memory"] = True

    if "optim" in sig.parameters:
        cfg_kwargs["optim"] = optim_resolved

    filtered = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters}
    training_args = SFTConfig(**filtered)
    print(
        f"[train] optim={optim_resolved} lr={lr} weight_decay={wd} "
        f"num_train_epochs={num_train_epochs} max_steps={max_steps if max_steps is not None else '-'}",
        file=sys.stderr,
    )

    eval_ner_cap = 2 if fit_8gb else (4 if args.low_vram else 16)
    eval_max_new = min(96, max(32, max_seq_length // 2))
    if fit_8gb:
        eval_max_new = min(eval_max_new, 64)

    do_epoch_ner_f1 = (not args.no_eval_ner_f1) and (not skip_eval)

    class NerSFTTrainer(SFTTrainer):
        """Adds multiset JSON-list F1 on the validation set each evaluation."""

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            import torch

            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if metrics is None:
                metrics = {}
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            if ds is None:
                return metrics
            if not do_epoch_ner_f1:
                self.model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return metrics

            eval_ds_list = val_row_dicts  # closure: same order as val_ds

            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if proc is None:
                return metrics
            cap = min(eval_ner_cap, len(eval_ds_list))
            mean_f1, n_used = _eval_fr_multiset_f1_batch(
                self.model,
                proc,
                eval_ds_list,
                max_seq_length=max_seq_length,
                max_new_tokens=eval_max_new,
                eval_cap=cap,
                multiset_f1_fn=multiset_f1,
                split_prompt_and_gold_fn=split_prompt_and_gold,
                parse_llm_terms_fn=parse_llm_terms,
            )
            if n_used:
                metrics["eval_ner_f1"] = mean_f1
            self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

    tr_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kw: dict = {
        "model": model,
        "train_dataset": train_ds,
        "args": training_args,
        "dataset_text_field": "text",
    }
    if not skip_eval:
        trainer_kw["eval_dataset"] = val_ds
    if "processing_class" in tr_sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    if "max_seq_length" not in sig.parameters and "max_seq_length" in tr_sig.parameters:
        trainer_kw["max_seq_length"] = max_seq_length

    trainer = NerSFTTrainer(**trainer_kw)

    resume_raw = getattr(args, "resume_from_checkpoint", None)
    resume_arg: bool | str | None = None
    if resume_raw:
        rs_strip = str(resume_raw).strip()
        rs = rs_strip.lower()
        if rs in ("auto", "true", "1", "yes"):
            from transformers.trainer_utils import get_last_checkpoint

            found = get_last_checkpoint(str(lora_dir))
            if found:
                resume_arg = found
                print(f"[train] resuming from checkpoint: {found}", file=sys.stderr)
            else:
                print(
                    f"[train] --resume-from-checkpoint {rs_strip!r}: no Trainer checkpoint under "
                    f"{lora_dir}; starting from scratch.",
                    file=sys.stderr,
                )
                resume_arg = None
        else:
            rp = Path(rs_strip)
            if not rp.is_absolute():
                rp = ROOT / rp
            if not rp.is_dir():
                raise SystemExit(f"--resume-from-checkpoint: not a directory: {rp.resolve()}")
            resume_arg = str(rp.resolve())

    trainer.train(resume_from_checkpoint=resume_arg)

    if on_test_rows:
        cap_test = max(1, int(args.ontology_test_eval_cap))
        if fit_8gb:
            cap_test = min(cap_test, 256)
        test_f1, n_used = _eval_fr_multiset_f1_batch(
            trainer.model,
            tokenizer,
            on_test_rows,
            max_seq_length=max_seq_length,
            max_new_tokens=eval_max_new,
            eval_cap=cap_test,
            multiset_f1_fn=multiset_f1,
            split_prompt_and_gold_fn=split_prompt_and_gold,
            parse_llm_terms_fn=parse_llm_terms,
        )
        rep = {
            "ontology_test_fr_f1": test_f1,
            "n_evaluated": n_used,
            "cap_requested": cap_test,
        }
        print(f"Ontology test multiset F1 (fr spans): {test_f1:.4f} (n={n_used})", file=sys.stderr)
        (lora_dir / "ontology_test_metrics.json").write_text(
            json.dumps(rep, indent=2) + "\n",
            encoding="utf-8",
        )

    # Save adapter + tokenizer (root of lora_dir)
    trainer.save_model(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # Merge LoRA into dense weights for downstream HF / 4-bit load in biomistral_prompt_ner.py
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as ex:
        print(f"warning: save_pretrained_merged failed ({ex}); will try PEFT fallback if needed.", file=sys.stderr)
    tokenizer.save_pretrained(str(merged_dir))

    if not _dir_has_hf_model_weights(merged_dir):
        print(
            "Merged export has no weight files (tokenizer-only). Running PEFT merge_and_unload fallback…",
            file=sys.stderr,
        )
        try:
            _merge_lora_peft_fallback(args.base_model, lora_dir, merged_dir, tokenizer)
        except Exception as ex:
            raise SystemExit(
                "Could not create merged weight files under:\n"
                f"  {merged_dir}\n"
                "Your LoRA adapter is saved here (use with biomistral_prompt_ner --backend unsloth --unsloth-lora-path):\n"
                f"  {lora_dir}\n"
                "PEFT merge needs enough RAM (often ~16 GB+ system RAM for 7B on CPU). "
                "Optional: UNSLOTH_MERGE_DEVICE=cuda if GPU fits bf16 merge.\n"
                f"Error: {ex}"
            ) from ex

    if not _dir_has_hf_model_weights(merged_dir):
        raise SystemExit(
            f"Merged dir still has no model weights after fallback: {merged_dir}\n"
            f"Use LoRA adapter at {lora_dir} with: biomistral_prompt_ner --backend unsloth --unsloth-lora-path ..."
        )

    # Training curves (this run only; use plot_training_curves.py CLI to overlay CamemBERT, etc.)
    try:
        plot_overlay = _load_plot_overlay()
        plot_overlay([lora_dir], lora_dir / "training_curves.png")
    except Exception as ex:
        print(f"warning: could not plot training curves: {ex}", file=sys.stderr)

    print(f"LoRA adapter: {lora_dir}")
    print(f"Merged model: {merged_dir}")


if __name__ == "__main__":
    main()
