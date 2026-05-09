#!/usr/bin/env python3
"""Patch hierarchical ontology SFT JSONL using English ``mdhier.asc`` (+ ``llt.asc`` / ``pt.asc``).

Each line is ``{"text": "..."}`` (Alpaca ``### Response:`` or Mistral ``[/INST]…``).
Only the JSON list in the response segment is parsed; ``soc`` / ``hlgt`` / ``hlt`` /
``pt`` / ``llt`` are overwritten from MedDRA hierarchy for the row's ``id`` (MedDRA code).

``mdhier.asc`` may be **dollar-delimited** (MedDRA MedAscii): codes are
``LLT, HLT, HLGT, SOC`` (not ``…, PT, …``); PT is resolved from ``llt.asc`` parent pointers
and ``pt.asc`` names. Only **primary** rows (``Y``) are indexed when duplicates exist.

Lookup uses **mdhier match tier** (LLT vs PT vs …) from the code column, not the JSON
``tier``/``level`` fields, so wrong labels in the export cannot null out SOC/PT. JSON ``id``
is normalized (int/float, leading zeros) so keys align with ``mdhier.asc``.

Legacy 5-key rows (no path fields) are left unchanged.

Examples::

    PYTHONPATH=. python tools/data/patch_ontology_sft_hierarchy_jsonl.py \\
      --input data/ontology_ner_full_hierarchical_alpaca.jsonl \\
      --output data/ontology_ner_full_hierarchical_alpaca.patched.jsonl

    PYTHONPATH=. python tools/data/patch_ontology_sft_hierarchy_jsonl.py \\
      --input data/ontology_ner_full_hierarchical_mistral_train.jsonl \\
      --in-place --backup .bak
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.meddra_io import (
    MEDDRA_TIERS,
    enrich_mdhier_row_pt,
    load_llt_to_parent_pt,
    load_pt_names,
    parse_mdhier_row,
    read_meddra_asc,
    split_meddra_asc_line,
)
from pipeline.ontology_sft_alpaca import hierarchy_flat_fields


def _json_id_to_str(v: Any) -> str:
    """Normalize ``id`` from JSON (int / float / str) for MedDRA key lookup."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return ""
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v):
            return str(int(v))
        return str(v).strip()
    if isinstance(v, int):
        return str(v)
    s = str(v).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def _is_meddra_numeric_code(s: str) -> bool:
    t = s.strip()
    return bool(t) and t.isdigit() and 5 <= len(t) <= 10


def _meddra_code_variants(code: str) -> list[str]:
    """Try raw, no leading zeros, and 8-digit zero-padded forms (MedDRA release quirks)."""
    s = code.strip()
    if not s:
        return []
    if not s.isdigit():
        return [s]
    stripped = s.lstrip("0") or "0"
    cand = [s, stripped]
    if len(stripped) < 8:
        cand.append(stripped.zfill(8))
    out: list[str] = []
    seen: set[str] = set()
    for c in cand:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _sanitize_mdhier_row(row: dict[str, str]) -> dict[str, str]:
    """Fix swapped code/name columns when the *name* field is actually a numeric MedDRA code."""
    r = dict(row)
    pairs = [
        ("pt_code", "pt_name"),
        ("hlt_code", "hlt_name"),
        ("hlgt_code", "hlgt_name"),
        ("soc_code", "soc_name"),
        ("llt_code", "llt_name"),
    ]
    for ck, nk in pairs:
        c = (r.get(ck) or "").strip()
        n = (r.get(nk) or "").strip()
        if _is_meddra_numeric_code(n) and c and not _is_meddra_numeric_code(c):
            r[ck], r[nk] = n, c
    return r


def _register_branch_keys(mapping: dict[str, dict[str, str]], code: str, branch: dict[str, str]) -> None:
    for k in _meddra_code_variants(code):
        mapping[k] = branch


class MdhierNameIndex:
    """English names on the primary branch, keyed by MedDRA concept id at any tier."""

    def __init__(self, mdhier_path: Path) -> None:
        self._by_llt: dict[str, dict[str, str]] = {}
        self._by_pt: dict[str, dict[str, str]] = {}
        self._by_hlt: dict[str, dict[str, str]] = {}
        self._by_hlgt: dict[str, dict[str, str]] = {}
        self._by_soc: dict[str, dict[str, str]] = {}
        en_dir = mdhier_path.parent
        llt_pt = load_llt_to_parent_pt(en_dir / "llt.asc") if (en_dir / "llt.asc").is_file() else {}
        pt_names = load_pt_names(en_dir / "pt.asc") if (en_dir / "pt.asc").is_file() else {}
        for line in read_meddra_asc(mdhier_path).splitlines():
            if not line.strip():
                continue
            raw = parse_mdhier_row(split_meddra_asc_line(line))
            if not raw:
                continue
            if (raw.get("primary_soc_fg") or "Y").upper() != "Y":
                continue
            row = enrich_mdhier_row_pt(_sanitize_mdhier_row(raw), llt_pt, pt_names)
            if not (row.get("pt_code") or "").strip():
                row = dict(row)
                row["pt_code"] = row["llt_code"]
                row["pt_name"] = (row.get("pt_name") or row.get("llt_name") or "").strip()
            llt_branch = {
                "SOC": row["soc_name"].strip(),
                "HLGT": row["hlgt_name"].strip(),
                "HLT": row["hlt_name"].strip(),
                "PT": row["pt_name"].strip(),
                "LLT": row["llt_name"].strip(),
            }
            _register_branch_keys(self._by_llt, row["llt_code"], llt_branch)
            pc = row["pt_code"].strip()
            if not any(k in self._by_pt for k in _meddra_code_variants(pc)):
                pt_branch = {
                    "SOC": row["soc_name"].strip(),
                    "HLGT": row["hlgt_name"].strip(),
                    "HLT": row["hlt_name"].strip(),
                    "PT": row["pt_name"].strip(),
                    "LLT": "",
                }
                _register_branch_keys(self._by_pt, pc, pt_branch)
            hc = row["hlt_code"].strip()
            if not any(k in self._by_hlt for k in _meddra_code_variants(hc)):
                hlt_branch = {
                    "SOC": row["soc_name"].strip(),
                    "HLGT": row["hlgt_name"].strip(),
                    "HLT": row["hlt_name"].strip(),
                    "PT": "",
                    "LLT": "",
                }
                _register_branch_keys(self._by_hlt, hc, hlt_branch)
            gc = row["hlgt_code"].strip()
            if not any(k in self._by_hlgt for k in _meddra_code_variants(gc)):
                hlgt_branch = {
                    "SOC": row["soc_name"].strip(),
                    "HLGT": row["hlgt_name"].strip(),
                    "HLT": "",
                    "PT": "",
                    "LLT": "",
                }
                _register_branch_keys(self._by_hlgt, gc, hlgt_branch)
            sc = row["soc_code"].strip()
            if not any(k in self._by_soc for k in _meddra_code_variants(sc)):
                soc_branch = {
                    "SOC": row["soc_name"].strip(),
                    "HLGT": "",
                    "HLT": "",
                    "PT": "",
                    "LLT": "",
                }
                _register_branch_keys(self._by_soc, sc, soc_branch)

    def branch_and_tier(self, cid: str) -> tuple[dict[str, str], str] | None:
        """Return (branch English names SOC→LLT, grounded MedDRA tier) using mdhier buckets only."""
        if not cid:
            return None
        for tier, mapping in (
            ("LLT", self._by_llt),
            ("PT", self._by_pt),
            ("HLT", self._by_hlt),
            ("HLGT", self._by_hlgt),
            ("SOC", self._by_soc),
        ):
            for k in _meddra_code_variants(cid):
                if k in mapping:
                    return dict(mapping[k]), tier
        return None


def _by_tier_from_branch(names: dict[str, str]) -> dict[str, dict[str, str]]:
    return {tier: {"name": names[tier]} for tier in MEDDRA_TIERS if names.get(tier)}


def patch_hierarchical_obj(obj: dict[str, Any], index: MdhierNameIndex) -> bool:
    """Mutate ``soc``…``llt`` from mdhier; return whether any field changed."""
    cid = _json_id_to_str(obj.get("id"))
    if not cid:
        return False
    if "soc" not in obj and "hlgt" not in obj and "en_resolved" not in obj:
        return False
    hit = index.branch_and_tier(cid)
    if not hit:
        return False
    branch, matched_tier = hit
    by_tier = _by_tier_from_branch(branch)
    flat = hierarchy_flat_fields(matched_tier, by_tier)
    changed = False
    for k in ("soc", "hlgt", "hlt", "pt", "llt"):
        new_v = flat.get(k)
        if obj.get(k) != new_v:
            obj[k] = new_v
            changed = True
    return changed


def split_text_response(text: str) -> tuple[str, str, str] | None:
    """Return ``(prefix, json_response_body, suffix)`` so ``prefix + body + suffix == text``."""
    marker = "### Response:\n"
    if marker in text:
        a, b = text.rsplit(marker, 1)
        return (a + marker, b, "")

    inst = "[/INST]"
    idx = text.rfind(inst)
    if idx < 0:
        return None
    prefix = text[: idx + len(inst)]
    rest = text[idx + len(inst) :]
    i = 0
    while i < len(rest) and rest[i] in " \t\n\r":
        i += 1
    prefix2 = prefix + rest[:i]
    tail = rest[i:]
    if not tail:
        return None
    dec = json.JSONDecoder()
    try:
        _, off = dec.raw_decode(tail)
    except json.JSONDecodeError:
        return None
    body = tail[:off]
    suffix = tail[off:]
    return (prefix2, body, suffix)


def process_text(text: str, index: MdhierNameIndex) -> tuple[str, dict[str, int]]:
    counts = {
        "rows_changed": 0,
        "objs_patchable": 0,
        "objs_changed": 0,
        "skip_fmt": 0,
        "skip_json": 0,
    }
    sp = split_text_response(text)
    if not sp:
        counts["skip_fmt"] = 1
        return text, counts
    pre, body, suf = sp
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        counts["skip_json"] = 1
        return text, counts
    if not isinstance(parsed, list):
        counts["skip_json"] = 1
        return text, counts
    line_changed = False
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        counts["objs_patchable"] += 1
        if patch_hierarchical_obj(obj, index):
            counts["objs_changed"] += 1
            line_changed = True
    if line_changed:
        counts["rows_changed"] = 1
        new_body = json.dumps(parsed, ensure_ascii=False)
        return pre + new_body + suf, counts
    return text, counts


def _candidate_repo_roots() -> list[Path]:
    """Roots for resolving relative CLI paths: cwd / ROOT, each with trailing-space stripped.

    The repo folder may literally be named ``…/MT_Project_Terminology `` (trailing space).
    Shell ``cd`` to the path *without* that space fails, but Python may still run from the
    spaced directory; we try both spellings so ``data/…`` resolves.
    """
    raw: list[Path] = [Path.cwd(), ROOT]
    out: list[Path] = []
    seen: set[str] = set()

    def _try_add(q: Path) -> None:
        try:
            r = q.resolve()
        except (OSError, RuntimeError):
            r = q
        k = str(r)
        if k not in seen:
            seen.add(k)
            out.append(r)

    for b in raw:
        _try_add(b)
        t = str(b).rstrip()
        if t and t != str(b):
            _try_add(Path(t))
    return out


def _resolve_cli_path(p: Path, *, is_input_file: bool) -> Path:
    """Resolve a relative path under the first repo root where it makes sense."""
    if p.is_absolute():
        return p.resolve()
    for base in _candidate_repo_roots():
        cand = base / p
        if is_input_file:
            if cand.is_file():
                return cand.resolve()
        else:
            try:
                if cand.parent.is_dir():
                    return cand.resolve()
            except OSError:
                continue
    return (Path.cwd() / p).resolve()


def _resolve_input_path(p: Path) -> Path:
    """Like ``_resolve_cli_path`` for inputs, but if ``*.jsonl.bak`` is missing use ``*.jsonl``."""
    r = _resolve_cli_path(p, is_input_file=True)
    if r.is_file():
        return r
    if p.name.endswith(".bak"):
        alt = p.parent / p.name[:-4]
        r2 = _resolve_cli_path(alt, is_input_file=True)
        if r2.is_file():
            print(f"[patch] {p} not found; reading from {r2}", file=sys.stderr)
            return r2
    tried = _resolve_cli_path(p, is_input_file=False)
    hints: list[Path] = []
    for base in _candidate_repo_roots():
        dd = base / "data"
        if dd.is_dir():
            hints.extend(sorted(dd.glob("ontology*hierarchical*train*.jsonl*"))[:25])
    msg = f"Input not found: {tried}\n"
    if hints:
        msg += "Under data/, found:\n  " + "\n  ".join(str(h) for h in hints[:20])
    else:
        msg += "No matching ontology*hierarchical*train*.jsonl* under data/ in cwd or repo root."
    raise SystemExit(msg)


def _repo_search_bases() -> list[Path]:
    """Ordered roots to find ``data/meddra`` (same logic as CLI path roots)."""
    return _candidate_repo_roots()


def _default_mdhier() -> Path:
    for base in _repo_search_bases():
        for rel in (
            ("data", "meddra", "MedAscii", "mdhier.asc"),
            ("data", "meddra", "mdhier.asc"),
        ):
            p = base.joinpath(*rel)
            if p.is_file():
                return p
    raise SystemExit("Could not find mdhier.asc under data/meddra/; pass --mdhier.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "If ``cd ~/…/MT_Project_Terminology`` fails with “No such file”, the directory name "
            "may end with a trailing space — use tab completion or quote it, e.g. "
            "``cd \"…/MT_Project_Terminology \"``. "
            "Relative ``--input`` / ``--output`` are resolved under cwd and under the script’s "
            "repo root (with and without that trailing space). "
            "If ``--input`` ends with ``.bak`` and that file is missing, the path without ``.bak`` "
            "is used when that file exists."
        ),
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true", help="Overwrite --input (use with --backup).")
    ap.add_argument("--backup", type=str, default=None, metavar="SUFFIX", help="Copy input to input+SUFFIX before overwrite.")
    ap.add_argument(
        "--mdhier",
        type=Path,
        default=None,
        help="Path to mdhier.asc (default: under data/meddra/).",
    )
    args = ap.parse_args()

    inp = _resolve_input_path(args.input)

    mdhier = args.mdhier
    if mdhier is None:
        mdhier = _default_mdhier()
    else:
        mdhier = _resolve_cli_path(mdhier, is_input_file=True)
    if not mdhier.is_file():
        raise SystemExit(f"mdhier not found: {mdhier}")

    index = MdhierNameIndex(mdhier)

    if args.in_place:
        out_path = inp
        if args.backup is not None:
            bak = inp.with_name(inp.name + args.backup)
            bak.write_bytes(inp.read_bytes())
    else:
        if args.output is None:
            raise SystemExit("Provide --output or use --in-place.")
        out_path = _resolve_cli_path(args.output, is_input_file=False)

    tot = {
        "lines": 0,
        "rows_changed": 0,
        "objs_patchable": 0,
        "objs_changed": 0,
        "skip_fmt": 0,
        "skip_json": 0,
    }
    out_lines: list[str] = []
    with inp.open(encoding="utf-8") as fin:
        for line in fin:
            raw = line.rstrip("\n\r")
            if not raw.strip():
                out_lines.append(raw)
                tot["lines"] += 1
                continue
            row = json.loads(raw)
            t = row.get("text")
            if not isinstance(t, str):
                tot["skip_fmt"] += 1
                out_lines.append(json.dumps(row, ensure_ascii=False))
                tot["lines"] += 1
                continue
            new_t, c = process_text(t, index)
            for k in tot:
                if k == "lines":
                    continue
                tot[k] += c[k]
            row["text"] = new_t
            out_lines.append(json.dumps(row, ensure_ascii=False))
            tot["lines"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(json.dumps(tot, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
