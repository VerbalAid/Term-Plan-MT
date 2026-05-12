"""Tests for MedDRA flat-file decoding and tier normalization."""

from __future__ import annotations

from pathlib import Path

from pipeline import (
    canonical_meddra_tier,
    enrich_mdhier_row_pt,
    load_llt_to_parent_pt,
    load_pt_names,
    parse_mdhier_row,
    read_meddra_asc,
    split_meddra_asc_line,
)


def test_read_meddra_asc_cp1252_eacute(tmp_path: Path) -> None:
    # "Apnée" as Windows-1252 bytes (é = 0xE9)
    p = tmp_path / "llt.asc"
    p.write_bytes(b"10001234|Apn\xe9e infantile|\n")
    text = read_meddra_asc(p)
    assert "Apnée" in text
    assert "\ufffd" not in text


def test_read_meddra_asc_utf8(tmp_path: Path) -> None:
    p = tmp_path / "x.asc"
    p.write_bytes("10001234|café|\n".encode("utf-8"))
    assert "café" in read_meddra_asc(p)


def test_canonical_meddra_tier_from_level() -> None:
    assert canonical_meddra_tier({"tier": None, "level": 5}) == "LLT"
    assert canonical_meddra_tier({"tier": "", "level": 4}) == "PT"
    assert canonical_meddra_tier({"tier": "", "level": 1}) == "SOC"


def test_canonical_meddra_tier_prefers_valid_string() -> None:
    assert canonical_meddra_tier({"tier": "HLT", "level": 99}) == "HLT"



def test_parse_mdhier_dollar_primary_y_and_enrich_pt(tmp_path: Path) -> None:
    # Synthetic dollar row: LLT, HLT, HLGT, SOC + four names + abbrev + empty + dup + Y + trailing empty
    line = (
        "10077321$10028947$10028971$10038738$"
        "Infantile apnoea$Neonatal hypoxic conditions$Neonatal respiratory disorders$"
        "Respiratory, thoracic and mediastinal disorders$Resp$$10038738$Y$$"
    )
    parts = split_meddra_asc_line(line)
    row = parse_mdhier_row(parts)
    assert row is not None
    assert row["soc_code"] == "10038738"
    assert row["primary_soc_fg"] == "Y"
    llt = tmp_path / "llt.asc"
    llt.write_text("10077321$Infantile apnoea$10077321$$$$$$$Y$$\n", encoding="utf-8")
    pt = tmp_path / "pt.asc"
    pt.write_text("10077321$Infantile apnoea$$10038738$$$$$$$$\n", encoding="utf-8")
    full = enrich_mdhier_row_pt(row, load_llt_to_parent_pt(llt), load_pt_names(pt))
    assert full["pt_code"] == "10077321"
    assert full["pt_name"] == "Infantile apnoea"
