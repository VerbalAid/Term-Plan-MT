"""Load translation models once (singletons) and small decode helpers."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# bitsandbytes 4-bit loads nvJitLink before torch runpaths apply — bootstrap pip CUDA libs first.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline.cuda_ld_path import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

_MISTRAL_TOK = None
_MISTRAL_MODEL = None
_NLLB_TOK = None
_NLLB_MODEL = None


def load_mistral_4bit(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    global _MISTRAL_TOK, _MISTRAL_MODEL
    if _MISTRAL_MODEL is not None:
        return _MISTRAL_TOK, _MISTRAL_MODEL
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    _MISTRAL_TOK = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # "auto" sometimes splits weights across GPU/CPU when VRAM is fragmented; bitsandbytes 4-bit
    # then raises (no mixed dispatch). Prefer a single GPU map on one CUDA device unless overridden.
    dm_raw = os.environ.get("MISTRAL_DEVICE_MAP", "").strip().lower()
    if dm_raw == "auto":
        device_map = "auto"
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    _MISTRAL_MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map=device_map,
    )
    return _MISTRAL_TOK, _MISTRAL_MODEL


def unload_mistral() -> None:
    """Drop Mistral singleton and free CUDA cache (e.g. before loading NLLB on small GPUs)."""
    global _MISTRAL_TOK, _MISTRAL_MODEL
    _MISTRAL_TOK = None
    _MISTRAL_MODEL = None
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_nllb() -> None:
    """Drop NLLB singleton and free CUDA cache (e.g. after S1 before Mistral systems)."""
    global _NLLB_TOK, _NLLB_MODEL
    _NLLB_TOK = None
    _NLLB_MODEL = None
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_nllb(model_id: str = "facebook/nllb-200-distilled-600M"):
    global _NLLB_TOK, _NLLB_MODEL
    if _NLLB_MODEL is not None:
        return _NLLB_TOK, _NLLB_MODEL
    _NLLB_TOK = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        _NLLB_MODEL = _NLLB_MODEL.cuda()
    return _NLLB_TOK, _NLLB_MODEL


def nllb_forced_bos_eng(tok) -> int:
    if hasattr(tok, "lang_code_to_id") and "eng_Latn" in getattr(tok, "lang_code_to_id", {}):
        return tok.lang_code_to_id["eng_Latn"]
    return tok.convert_tokens_to_ids("eng_Latn")


def strip_inst_echo(decoded_full: str) -> str:
    if "[/INST]" in decoded_full:
        return decoded_full.rsplit("[/INST]", 1)[-1].strip()
    return decoded_full.strip()
