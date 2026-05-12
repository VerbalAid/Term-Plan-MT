"""Tests for Mistral hierarchical JSONL → Alpaca reframing."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util


def _load_converter():
    p = Path(__file__).resolve().parents[1] / "tools" / "data" / "mistral_hierarchical_jsonl_to_alpaca.py"
    spec = importlib.util.spec_from_file_location("mistral_hierarchical_jsonl_to_alpaca", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_split_mistral_hierarchical_roundtrip_keys() -> None:
    mod = _load_converter()
    from ontology_sft import to_mistral_instruct_hierarchical

    fr_body = "French: Une pneumopathie.\n\nExtracted terms:\n- x → MedDRA ID: 1"
    assistant = '[{"fr": "x", "en": "y", "id": "1"}]'
    mistral = to_mistral_instruct_hierarchical(fr_body, assistant)
    fb, asst = mod.split_mistral_hierarchical_text(mistral)
    assert fb == fr_body
    assert asst == assistant


def test_reframe_to_alpaca_contains_response_json() -> None:
    mod = _load_converter()
    from ontology_sft import to_alpaca_hierarchical, to_mistral_instruct_hierarchical

    fr_body = "French: test.\n\nTerms:\n- a → ID: 9"
    assistant = '[{"fr": "a", "en": "b", "tier": "LLT", "id": "9"}]'
    mistral = to_mistral_instruct_hierarchical(fr_body, assistant)
    fb, asst = mod.split_mistral_hierarchical_text(mistral)
    alpaca = to_alpaca_hierarchical(fb, asst)
    assert "### Response:\n" + assistant in alpaca
