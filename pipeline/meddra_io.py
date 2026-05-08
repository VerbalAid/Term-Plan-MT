"""MedDRA flat-file I/O helpers: encoding detection and tier normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# MedDRA hierarchy level → tier (ICH standard)
_LEVEL_TO_TIER: dict[int, str] = {1: "SOC", 2: "HLGT", 3: "HLT", 4: "PT", 5: "LLT"}
MEDDRA_TIERS = frozenset(_LEVEL_TO_TIER.values())


def read_meddra_asc(path: Path) -> str:
    """Decode a MedDRA ``*.asc`` / ``mdhier.asc`` file without mangling French bytes.

    Releases ship as UTF-8, Windows-1252, or ISO-8859-1 depending on locale/version.
    Using UTF-8 with ``errors='replace'`` silently turns ``é`` into U+FFFD — avoid that.
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_meddra_asc_line(line: str) -> list[str]:
    """Split one MedDRA ``*.asc`` record line on ``|`` or legacy ``$``."""
    line = line.rstrip("\n\r")
    if "|" in line:
        return [p.strip() for p in line.split("|")]
    return [p.strip() for p in line.split("$")]


def _mdhier_primary_flag(parts: list[str]) -> str:
    """Return ``Y`` / ``N`` primary-path flag; trailing ``$`` fields may leave empty tail tokens."""
    for j in range(len(parts) - 1, -1, -1):
        s = parts[j].strip().upper()
        if s in ("Y", "N"):
            return s
    return "Y"


def parse_mdhier_row(parts: list[str]) -> dict[str, str] | None:
    """Parse one ``mdhier.asc`` record.

    **Pipe / legacy (10 columns):** ``llt_cd|llt_name|pt_cd|pt_name|…|soc_name`` (ICH order).

    **Dollar (ASCII, 12+ fields):** ``llt$hlt$hlgt$soc$llt_name$hlt_name$hlgt_name$soc_name$…$primary`` —
    the fourth code is **SOC**, not PT. ``pt_code`` / ``pt_name`` are filled later from ``llt.asc`` + ``pt.asc``.
    """
    if len(parts) < 10:
        return None
    p0, p1, p2, p3 = (parts[i].strip() for i in range(4))
    primary = _mdhier_primary_flag(parts)
    if (
        len(parts) >= 11
        and p0.isdigit()
        and p1.isdigit()
        and p2.isdigit()
        and p3.isdigit()
        and not parts[4].strip().isdigit()
        and primary in ("Y", "N")
    ):
        return {
            "llt_code": p0,
            "hlt_code": p1,
            "hlgt_code": p2,
            "soc_code": p3,
            "llt_name": parts[4].strip(),
            "hlt_name": parts[5].strip(),
            "hlgt_name": parts[6].strip(),
            "soc_name": parts[7].strip(),
            "soc_abbrev": parts[8].strip() if len(parts) > 8 else "",
            "pt_soc_dup": parts[10].strip() if len(parts) > 10 else "",
            "primary_soc_fg": primary,
            "pt_code": "",
            "pt_name": "",
        }
    return {
        "llt_code": parts[0].strip(),
        "llt_name": parts[1].strip(),
        "pt_code": parts[2].strip(),
        "pt_name": parts[3].strip(),
        "hlt_code": parts[4].strip(),
        "hlt_name": parts[5].strip(),
        "hlgt_code": parts[6].strip(),
        "hlgt_name": parts[7].strip(),
        "soc_code": parts[8].strip(),
        "soc_name": parts[9].strip(),
        "primary_soc_fg": "Y",
    }


def load_llt_to_parent_pt(llt_asc: Path) -> dict[str, str]:
    """Map ``llt_cd`` → parent ``pt_cd`` from English ``llt.asc`` (third field)."""
    out: dict[str, str] = {}
    for line in read_meddra_asc(llt_asc).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) < 3:
            continue
        llt, pt = parts[0].strip(), parts[2].strip()
        if llt.isdigit() and pt.isdigit():
            out[llt] = pt
    return out


def load_pt_names(pt_asc: Path) -> dict[str, str]:
    """Map ``pt_cd`` → English PT name from ``pt.asc``."""
    out: dict[str, str] = {}
    for line in read_meddra_asc(pt_asc).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) < 2:
            continue
        cid, name = parts[0].strip(), parts[1].strip()
        if cid.isdigit():
            out[cid] = name
    return out


def enrich_mdhier_row_pt(row: dict[str, str], llt_pt: dict[str, str], pt_names: dict[str, str]) -> dict[str, str]:
    """Fill ``pt_code`` / ``pt_name`` when missing (dollar-format ``mdhier`` rows)."""
    if (row.get("pt_code") or "").strip():
        return row
    r = dict(row)
    llt = (r.get("llt_code") or "").strip()
    pt = (llt_pt.get(llt) or "").strip()
    r["pt_code"] = pt
    r["pt_name"] = (pt_names.get(pt) or r.get("llt_name") or "").strip()
    return r


def canonical_meddra_tier(node_like: dict[str, Any]) -> str:
    """Return SOC/HLGT/HLT/PT/LLT using ``tier`` when valid, else ``level`` (1–5)."""
    t = str(node_like.get("tier") or "").strip().upper()
    if t in MEDDRA_TIERS:
        return t
    try:
        lvl = int(node_like.get("level"))
    except (TypeError, ValueError):
        return t
    return _LEVEL_TO_TIER.get(lvl, t)
