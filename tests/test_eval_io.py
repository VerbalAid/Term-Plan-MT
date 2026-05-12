"""Tests for JSONL alignment helpers used by evaluate / plots."""

from __future__ import annotations

from metrics import (
    CONTAMINATION_PLACEHOLDER_HYP,
    align_hyp_ref_by_doc,
    fluency_hypothesis_text,
)


def test_fluency_hypothesis_text_plain():
    assert fluency_hypothesis_text("hello") == "hello"
    assert fluency_hypothesis_text("") == ""


def test_fluency_hypothesis_text_none():
    assert fluency_hypothesis_text(None) == ""


def test_fluency_hypothesis_text_contamination_placeholder():
    assert fluency_hypothesis_text(CONTAMINATION_PLACEHOLDER_HYP) == ""
    assert fluency_hypothesis_text("__CONTAMINATED__") == ""


def test_contamination_placeholder_constant():
    assert CONTAMINATION_PLACEHOLDER_HYP == "__CONTAMINATED__"


def test_align_hyp_ref_strips_contamination_placeholder():
    results = [
        {"id": "48_001", "hyp": CONTAMINATION_PLACEHOLDER_HYP},
        {"id": "48_002", "hyp": "ok translation"},
    ]
    id_to_ref = {"48_001": "ref one", "48_002": "ref two"}
    id_to_doc = {"48_001": "48", "48_002": "48"}
    hyps, refs, _ = align_hyp_ref_by_doc(results, id_to_ref, id_to_doc)
    assert hyps[0] == ""
    assert hyps[1] == "ok translation"
    assert refs == ["ref one", "ref two"]
