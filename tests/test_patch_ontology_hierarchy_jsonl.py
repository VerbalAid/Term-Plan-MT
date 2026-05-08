"""Tests for mdhier-based JSONL hierarchy patcher."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_patch_module():
    p = ROOT / "tools" / "data" / "patch_ontology_sft_hierarchy_jsonl.py"
    spec = importlib.util.spec_from_file_location("patch_ontology_sft_hierarchy_jsonl", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_split_alpaca_roundtrip() -> None:
    mod = _load_patch_module()
    text = (
        "### Instruction:\nGo\n\n### Input:\nX\n\n### Response:\n"
        + json.dumps([{"a": 1}], ensure_ascii=False)
    )
    sp = mod.split_text_response(text)
    assert sp is not None
    pre, body, suf = sp
    assert pre + body + suf == text
    assert json.loads(body) == [{"a": 1}]


def test_split_mistral_roundtrip() -> None:
    mod = _load_patch_module()
    inner = json.dumps([{"x": "y"}], ensure_ascii=False)
    text = f"<s>[INST] u [/INST]  {inner}  </s>"
    sp = mod.split_text_response(text)
    assert sp is not None
    pre, body, suf = sp
    assert pre + body + suf == text


def test_patch_llt_from_mdhier(tmp_path: Path) -> None:
    mod = _load_patch_module()
    md = tmp_path / "mdhier.asc"
    # llt|llt|pt|pt|hlt|hlt|hlgt|hlgt|soc|soc
    md.write_text(
        "100001|Child asthma|200001|Asthma|300001|HLT X|400001|HLGT Y|500001|SOC Z\n",
        encoding="utf-8",
    )
    idx = mod.MdhierNameIndex(md)
    obj = {
        "fr": "asthme",
        "en": "Asthma",
        "en_resolved": "Child asthma",
        "tier": "LLT",
        "id": "100001",
        "level": 5,
        "soc": None,
        "hlgt": None,
        "hlt": None,
        "pt": None,
        "llt": None,
    }
    assert mod.patch_hierarchical_obj(obj, idx) is True
    assert obj["soc"] == "SOC Z"
    assert obj["pt"] == "Asthma"
    assert obj["llt"] == "Child asthma"


def test_patch_leading_zero_id(tmp_path: Path) -> None:
    mod = _load_patch_module()
    md = tmp_path / "mdhier.asc"
    md.write_text(
        "01000001|L|200001|P|300001|H|400001|G|500001|S\n",
        encoding="utf-8",
    )
    idx = mod.MdhierNameIndex(md)
    obj = {"en_resolved": "x", "id": "1000001", "tier": "LLT", "level": 5}
    assert mod.patch_hierarchical_obj(obj, idx) is True
    assert obj["llt"] == "L"
    assert obj["soc"] == "S"


def test_patch_float_id(tmp_path: Path) -> None:
    mod = _load_patch_module()
    md = tmp_path / "mdhier.asc"
    md.write_text(
        "100001|L|200001|P|300001|H|400001|G|500001|S\n",
        encoding="utf-8",
    )
    idx = mod.MdhierNameIndex(md)
    # JSON tier is wrong (PT) but id is an LLT code — patcher uses mdhier bucket (LLT).
    obj = {"en_resolved": "x", "id": 100001.0, "tier": "PT", "level": 4}
    assert mod.patch_hierarchical_obj(obj, idx) is True
    assert obj["llt"] == "L"
    assert obj["pt"] == "P"
    assert obj["soc"] == "S"


def test_patch_swapped_pt_columns(tmp_path: Path) -> None:
    mod = _load_patch_module()
    md = tmp_path / "mdhier.asc"
    # pt_cd / pt_name columns reversed (name slot holds numeric code)
    md.write_text(
        "100001|L|TrueName|200001|300001|H|400001|G|500001|S\n",
        encoding="utf-8",
    )
    idx = mod.MdhierNameIndex(md)
    obj = {"en_resolved": "x", "id": "100001", "tier": "LLT", "level": 5}
    mod.patch_hierarchical_obj(obj, idx)
    assert obj["pt"] == "TrueName"
    assert not str(obj.get("pt") or "").isdigit()
