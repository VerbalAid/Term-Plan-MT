"""TermPlanMT translation systems S1–S6.

All systems share the same inputs:
- A segment dict with ``fr`` (French text), ``terms`` (NER spans), ``en_ref``.
- A ``TermGraph`` instance for grounding.
- A planning ``locks`` dict mapping French terms to English renderings.

Usage::

    from pipeline import TermGraph, load_or_compute_locks
    from systems import run_system, load_all_segments

    with TermGraph() as graph:
        locks = load_or_compute_locks(seg_path, graph)
        run_system("s3", seg_path, out_path, graph=graph, locks=locks)
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import math
import os
import site
import statistics
import sys
import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

log = logging.getLogger(__name__)


# ── CUDA library path setup ────────────────────────────────────────────────
# PyTorch+cu130 wheels ship libnvJitLink.so under site-packages/nvidia/.
# Without prepending those directories to LD_LIBRARY_PATH, bitsandbytes
# (4-bit quantisation) fails at import time.

_CUDA_LIBS_DONE = False


def ensure_cuda_pip_libs_visible() -> None:
    """Prepend NVIDIA pip-wheel library directories to LD_LIBRARY_PATH.

    Call this once before importing torch or bitsandbytes.
    """
    global _CUDA_LIBS_DONE
    if _CUDA_LIBS_DONE:
        return

    # Collect all site-packages directories.
    roots: list[Path] = []
    try:
        for r in (*site.getsitepackages(), site.getusersitepackages()):
            if r:
                roots.append(Path(r).resolve())
    except AttributeError:
        pass
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for base in (Path(sys.prefix) / "lib", Path(sys.prefix) / "lib64"):
        sp = base / f"python{ver}" / "site-packages"
        if sp.is_dir():
            roots.append(sp.resolve())

    # Find directories that contain libnvJitLink.so*.
    extra: list[str] = []
    seen: set[str] = set()
    for base in roots:
        for rel in ("nvidia/cu13/lib", "nvidia/nvjitlink/lib", "nvidia/cuda_nvjitlink/lib"):
            d = (base / rel).resolve()
            if d.is_dir() and any(d.glob("libnvJitLink.so*")):
                s = str(d)
                if s not in seen:
                    seen.add(s); extra.append(s)
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for child in nvidia.iterdir():
                lib = (child / "lib").resolve()
                if lib.is_dir() and any(lib.glob("libnvJitLink.so*")):
                    s = str(lib)
                    if s not in seen:
                        seen.add(s); extra.append(s)

    if extra:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        parts = extra + ([prev] if prev else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        try:
            RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 256)
            mode = ctypes.DEFAULT_MODE | RTLD_GLOBAL
            for d in extra:
                for name in ("libnvJitLink.so.13", "libnvJitLink.so.12"):
                    so = Path(d) / name
                    if so.is_file():
                        try:
                            ctypes.CDLL(str(so), mode=mode)
                        except OSError:
                            pass
                        break
        except Exception:
            pass

    _CUDA_LIBS_DONE = True


# ── Segment loading helpers ────────────────────────────────────────────────


def parse_exclude_segment_ids(spec: str | None) -> frozenset[str]:
    """Parse a comma-separated list of segment ids to exclude (e.g. ``"48_028"``)."""
    if not spec or not str(spec).strip():
        return frozenset()
    return frozenset(x.strip() for x in str(spec).split(",") if x.strip())


def iter_segments(path: Path) -> Iterator[dict[str, Any]]:
    """Yield segment dicts from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_segments_filtered(
    path: Path,
    exclude_segment_ids: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield segments, skipping any ids in ``exclude_segment_ids``."""
    for seg in iter_segments(path):
        if exclude_segment_ids and seg.get("id") in exclude_segment_ids:
            continue
        yield seg


def iter_limited(
    path: Path,
    limit: int | None,
    exclude_segment_ids: frozenset[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield at most ``limit`` segments (after exclusion filtering)."""
    for i, seg in enumerate(iter_segments_filtered(path, exclude_segment_ids)):
        if limit is not None and i >= limit:
            break
        yield seg


def load_all_segments(
    path: Path,
    exclude_segment_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Load all segments from a JSONL file into a list."""
    return list(iter_segments_filtered(path, exclude_segment_ids))


def write_result_row(
    out_f: TextIO,
    *,
    system: str,
    seg: dict[str, Any],
    hyp: str,
    inference_s: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one result row to an open file handle and flush immediately."""
    row: dict[str, Any] = {
        "id": seg["id"], "system": system, "fr": seg["fr"],
        "hyp": hyp, "en_ref": seg["en_ref"],
        "inference_s": inference_s,
    }
    if extra:
        row.update(extra)
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_f.flush()


# ── Model singletons ───────────────────────────────────────────────────────
# Models are loaded once per process and cached in module-level globals.

_NLLB_TOK = _NLLB_MODEL = None
_MIS_TOK   = _MIS_MODEL  = None


def _nllb():
    """Load (or return cached) NLLB-200 tokeniser and model."""
    global _NLLB_TOK, _NLLB_MODEL
    if _NLLB_MODEL is None:
        _NLLB_TOK   = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=True)
        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        if torch.cuda.is_available():
            _NLLB_MODEL = _NLLB_MODEL.cuda()
    return _NLLB_TOK, _NLLB_MODEL


def _mistral(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load (or return cached) Mistral-7B-Instruct in 4-bit precision."""
    global _MIS_TOK, _MIS_MODEL
    if _MIS_MODEL is None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        _MIS_TOK   = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        dm  = {"": 0} if torch.cuda.is_available() else "auto"
        _MIS_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map=dm
        )
    return _MIS_TOK, _MIS_MODEL


# ── Model management (public wrappers) ─────────────────────────────────────


def load_nllb() -> tuple:
    """Return (tokeniser, model) for NLLB-200, loading if necessary."""
    return _nllb()


def load_mistral_4bit(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> tuple:
    """Return (tokeniser, model) for Mistral 4-bit, loading if necessary."""
    return _mistral(model_id)


def unload_nllb() -> None:
    """Release the NLLB model from GPU memory."""
    global _NLLB_TOK, _NLLB_MODEL
    if _NLLB_MODEL is not None:
        del _NLLB_MODEL; _NLLB_MODEL = None
        del _NLLB_TOK;   _NLLB_TOK   = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_mistral() -> None:
    """Release the Mistral model from GPU memory."""
    global _MIS_TOK, _MIS_MODEL
    if _MIS_MODEL is not None:
        del _MIS_MODEL; _MIS_MODEL = None
        del _MIS_TOK;   _MIS_TOK   = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Shared decoding helpers ────────────────────────────────────────────────


def _nllb_bos(tok) -> int:
    """Return the NLLB English BOS token id."""
    if hasattr(tok, "lang_code_to_id") and "eng_Latn" in tok.lang_code_to_id:
        return tok.lang_code_to_id["eng_Latn"]
    return tok.convert_tokens_to_ids("eng_Latn")


def _strip_inst(text: str) -> str:
    """Remove the Mistral ``[INST]…[/INST]`` echo from generated text."""
    if "[/INST]" in text:
        return text.rsplit("[/INST]", 1)[-1].strip()
    return text.strip()


def _mistral_prompt(fr: str, meddra_lines: list[str]) -> str:
    sys_msg = (
        "You are a medical translator for EMA regulatory documents. "
        "Use the MedDRA renderings exactly where applicable. Return only the translation."
    )
    ctx  = "\n".join(meddra_lines) or "(no grounded terms)"
    user = f"MedDRA context:\n{ctx}\n\nTranslate:\n{fr}"
    return f"<s>[INST] {sys_msg}\n\n{user} [/INST]"


def _meddra_lines(seg: dict, graph: Any, locks: dict) -> list[str]:
    """Build the MedDRA context block for Mistral prompts."""
    lines = []
    fr_ctx = (seg.get("fr") or "").strip() or None
    for t in seg.get("terms") or []:
        w = (t.get("word") or "").strip()
        if not w:
            continue
        concept = graph.ground(w, context=fr_ctx)
        if not concept:
            continue
        en = locks.get(w) or concept["name"]
        lines.append(f"  '{w}' → '{en}' (MedDRA {concept['tier']} L{concept['level']})")
    return lines


class PhraseLogitBoost(LogitsProcessor):
    """Boost logits for tokens that appear in target English phrases."""

    def __init__(self, tok: Any, phrases: list[str], boost: float = 1.75):
        self.boost = boost
        self.ids: set[int] = set()
        for p in phrases:
            for tid in tok(p, add_special_tokens=False)["input_ids"]:
                self.ids.add(int(tid))

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for tid in self.ids:
            if 0 <= tid < scores.shape[-1]:
                scores[:, tid] += self.boost
        return scores


def _phrase_list(seg: dict, graph: Any, locks: dict) -> list[str]:
    """Collect locked English phrases for a segment."""
    phrases, seen = [], set()
    fr_ctx = (seg.get("fr") or "").strip() or None
    for t in seg.get("terms") or []:
        w = (t.get("word") or "").strip()
        if not w:
            continue
        concept = graph.ground(w, context=fr_ctx)
        if not concept:
            continue
        en = locks.get(w) or concept["name"]
        if en and en not in seen:
            seen.add(en)
            phrases.append(en)
    return phrases


def _norm_fr(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip()).casefold()
    return " ".join(s.split())


def _glossary_phrases(seg: dict, glossary: dict[str, str]) -> list[str]:
    """Look up English phrases from a hand-built glossary for a segment."""
    phrases, seen = [], set()
    for t in seg.get("terms") or []:
        w = (t.get("word") or "").strip()
        en = glossary.get(_norm_fr(w))
        if en and en not in seen:
            seen.add(en)
            phrases.append(en)
    return phrases


# ── The six translation systems ────────────────────────────────────────────

def _s1(seg: dict, **_) -> str:
    """S1: NLLB-200 baseline, no terminology constraint."""
    tok, model = _nllb()
    tok.src_lang = "fra_Latn"
    dev = next(model.parameters()).device
    inp = tok(seg["fr"], return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = model.generate(**inp, forced_bos_token_id=_nllb_bos(tok),
                             num_beams=5, max_new_tokens=256)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _s2(seg: dict, full_doc_fr: str = "", **_) -> str:
    """S2: Mistral with full-document context (no terminology constraint)."""
    tok, model = _mistral()
    sys_msg = (
        "You are a medical translator for EMA SmPC documents. "
        "Translate ONLY the TARGET sentence. Return only the translation."
    )
    user = (
        f"FULL SECTION (context):\n{full_doc_fr[:6000]}\n\n"
        f"TARGET:\n{seg['fr']}"
    )
    prompt = f"<s>[INST] {sys_msg}\n\n{user} [/INST]"
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=512, pad_token_id=tok.eos_token_id)
    return _strip_inst(tok.decode(out[0], skip_special_tokens=True))


def _s3(seg: dict, graph: Any = None, locks: dict | None = None, **_) -> str:
    """S3: Mistral with MedDRA concept context (GraphRAG-style prompt)."""
    lines = _meddra_lines(seg, graph, locks or {}) if graph else []
    tok, model = _mistral()
    inp = tok(_mistral_prompt(seg["fr"], lines), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=512, pad_token_id=tok.eos_token_id)
    return _strip_inst(tok.decode(out[0], skip_special_tokens=True))


def _s4(seg: dict, graph: Any = None, locks: dict | None = None,
         n_samples: int = 3, **_) -> str:
    """S4: GraphRAG + reranking — sample ``n_samples`` candidates, pick the best."""
    lines = _meddra_lines(seg, graph, locks or {}) if graph else []
    tok, model = _mistral()
    prompt = _mistral_prompt(seg["fr"], lines)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    cands = []
    with torch.inference_mode():
        for _ in range(n_samples):
            out = model.generate(**inp, max_new_tokens=512, pad_token_id=tok.eos_token_id,
                                 do_sample=True, temperature=0.2)
            cands.append(_strip_inst(tok.decode(out[0], skip_special_tokens=True)))
    if not graph:
        return cands[0]

    # Score each candidate by the number of grounded MedDRA terms it contains.
    fr_ctx = (seg.get("fr") or "").strip() or None

    def score(c: str) -> float:
        total = 0.0
        for t in seg.get("terms") or []:
            concept = graph.ground((t.get("word") or "").strip(), context=fr_ctx)
            if not concept:
                continue
            if concept["name"].lower() in c.lower():
                total += 1.0
            elif graph.same_branch(c.split()[0], concept["name"]):
                total += 0.5
        return total

    return max(cands, key=score)


def _s5_nllb(seg: dict, graph: Any = None, locks: dict | None = None, **_) -> str:
    """S5 (NLLB): NLLB-200 with logit boost and force-words for locked terms."""
    phrases = _phrase_list(seg, graph, locks or {}) if graph else []
    tok, model = _nllb()
    tok.src_lang = "fra_Latn"
    dev = next(model.parameters()).device
    inp = tok(seg["fr"], return_tensors="pt").to(dev)
    boost = PhraseLogitBoost(tok, phrases, boost=1.75)
    beams = min(10, 5 + len(phrases))
    try:
        phrase_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in phrases if p]
        with torch.inference_mode():
            out = model.generate(
                **inp, forced_bos_token_id=_nllb_bos(tok),
                logits_processor=LogitsProcessorList([boost]),
                force_words_ids=phrase_ids, num_beams=beams, max_new_tokens=256,
            )
    except Exception:
        # Fall back without force_words_ids if phrase decoding fails.
        with torch.inference_mode():
            out = model.generate(
                **inp, forced_bos_token_id=_nllb_bos(tok),
                logits_processor=LogitsProcessorList([boost]),
                num_beams=beams, max_new_tokens=256,
            )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _s5_mistral(seg: dict, graph: Any = None, locks: dict | None = None, **_) -> str:
    """S5 (Mistral): Mistral with both MedDRA context and logit boost."""
    phrases = _phrase_list(seg, graph, locks or {}) if graph else []
    lines   = _meddra_lines(seg, graph, locks or {}) if graph else []
    tok, model = _mistral()
    inp = tok(_mistral_prompt(seg["fr"], lines), return_tensors="pt").to(model.device)
    boost = PhraseLogitBoost(tok, phrases, boost=1.25)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=512, pad_token_id=tok.eos_token_id,
                             logits_processor=LogitsProcessorList([boost]))
    return _strip_inst(tok.decode(out[0], skip_special_tokens=True))


def _s6_nllb(seg: dict, glossary: dict | None = None, **_) -> str:
    """S6 (NLLB): NLLB-200 with logit boost from a hand-built glossary (oracle ablation)."""
    phrases = _glossary_phrases(seg, glossary or {})
    tok, model = _nllb()
    tok.src_lang = "fra_Latn"
    dev = next(model.parameters()).device
    inp = tok(seg["fr"], return_tensors="pt").to(dev)
    boost = PhraseLogitBoost(tok, phrases, boost=1.75)
    beams = min(10, 5 + len(phrases))
    try:
        phrase_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in phrases if p]
        with torch.inference_mode():
            out = model.generate(
                **inp, forced_bos_token_id=_nllb_bos(tok),
                logits_processor=LogitsProcessorList([boost]),
                force_words_ids=phrase_ids, num_beams=beams, max_new_tokens=256,
            )
    except Exception:
        with torch.inference_mode():
            out = model.generate(
                **inp, forced_bos_token_id=_nllb_bos(tok),
                logits_processor=LogitsProcessorList([boost]),
                num_beams=beams, max_new_tokens=256,
            )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _s6_mistral(seg: dict, glossary: dict | None = None, **_) -> str:
    """S6 (Mistral): Mistral with glossary context and logit boost."""
    phrases = _glossary_phrases(seg, glossary or {})
    lines = [
        f"  '{_norm_fr(k)}' → '{v}'"
        for k, v in (glossary or {}).items()
        if any(_norm_fr((t.get("word") or "")) == k for t in seg.get("terms") or [])
    ]
    tok, model = _mistral()
    inp = tok(_mistral_prompt(seg["fr"], lines), return_tensors="pt").to(model.device)
    boost = PhraseLogitBoost(tok, phrases, boost=1.25)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=512, pad_token_id=tok.eos_token_id,
                             logits_processor=LogitsProcessorList([boost]))
    return _strip_inst(tok.decode(out[0], skip_special_tokens=True))


# ── System registry and runner ─────────────────────────────────────────────

_SYSTEMS: dict[str, tuple] = {
    "s1":        (_s1,        "s1.jsonl"),
    "s2":        (_s2,        "s2.jsonl"),
    "s3":        (_s3,        "s3.jsonl"),
    "s4":        (_s4,        "s4.jsonl"),
    "s5":        (_s5_nllb,   "s5.jsonl"),
    "s5_mistral":(_s5_mistral,"s5_mistral.jsonl"),
    "s6":        (_s6_nllb,   "s6.jsonl"),
    "s6_mistral":(_s6_mistral,"s6_mistral.jsonl"),
}


def run_system(
    system: str,
    segments_path: Path,
    out_path: Path,
    *,
    graph: Any = None,
    locks: dict | None = None,
    glossary: dict | None = None,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
    exclude_ids: frozenset[str] | None = None,
    full_doc_fr: str = "",
) -> None:
    """Run translation system ``system`` over a segments JSONL file.

    Writes one result row per segment to ``out_path`` (JSONL), flushing after each
    row so that partial runs are resumable.
    """
    fn, _ = _SYSTEMS[system]
    segments = load_all_segments(segments_path, exclude_ids)
    if limit:
        segments = segments[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for seg in tqdm(segments, desc=system):
            if skip_ids and seg["id"] in skip_ids:
                continue
            t0  = time.perf_counter()
            hyp = fn(seg, graph=graph, locks=locks or {}, glossary=glossary or {},
                     full_doc_fr=full_doc_fr)
            row = {
                "id": seg["id"], "system": system, "fr": seg["fr"],
                "hyp": hyp, "en_ref": seg["en_ref"],
                "inference_s": round(time.perf_counter() - t0, 4),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
