#!/usr/bin/env python3
"""Build instruction-style SFT JSONL from MedDRA ``:Concept`` rows in Neo4j (no sentence pairs).

Each example teaches: French term + ontology context → English MedDRA label + level + id.
Uses the same Neo4j auth as :class:`pipeline.graph.TermGraph`.

Stratified sampling: up to ``limit // 5`` concepts per integer level 1–5 (default ``--limit 5000`` → 1000 each),
then shuffle and split 90/10 into ``ontology_train.jsonl`` and ``ontology_val.jsonl``.

Example::

    PYTHONPATH=. python tools/data/build_ontology_sft.py --output-dir data/sft/ --limit 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.graph import TermGraph

LEVEL_NAMES = ("", "SOC", "HLGT", "HLT", "PT", "LLT")  # index by level 1–5

_INSTRUCTION = (
    "You are a MedDRA terminology specialist. "
    "Given a French medical term and its ontology context, "
    "output the correct English rendering and confirm the hierarchy level."
)


def _tier_name(tier: Any) -> str:
    t = str(tier or "").strip().upper()
    return t if t else "UNKNOWN"


def _level_int(level: Any) -> int | None:
    if level is None:
        return None
    try:
        v = int(level)
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 5:
        return v
    return None


def _fetch_rows(graph: TermGraph) -> list[dict[str, Any]]:
    """One row per :Concept with FR+EN; optional immediate broader parent."""
    q = """
    MATCH (c:Concept)
    WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
      AND c.name IS NOT NULL AND trim(c.name) <> ''
      AND c.level IS NOT NULL
    OPTIONAL MATCH (p:Concept)-[:BROADER_THAN]->(c)
    RETURN coalesce(toString(c.id), toString(c.name)) AS id,
           c.fr_label AS fr,
           c.name AS en,
           c.level AS level,
           c.tier AS tier,
           p.fr_label AS parent_fr,
           p.name AS parent_en,
           p.level AS parent_level,
           p.tier AS parent_tier
    """
    rows: list[dict[str, Any]] = []
    with graph._driver.session() as session:
        for rec in session.run(q):
            rows.append(
                {
                    "id": rec["id"],
                    "fr": (rec["fr"] or "").strip(),
                    "en": (rec["en"] or "").strip(),
                    "level": rec["level"],
                    "tier": rec["tier"],
                    "parent_fr": (rec["parent_fr"] or "").strip() if rec["parent_fr"] else None,
                    "parent_en": (rec["parent_en"] or "").strip() if rec["parent_en"] else None,
                    "parent_level": rec["parent_level"],
                    "parent_tier": rec["parent_tier"],
                }
            )
    return rows


def _dedupe_by_concept_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MedDRA should have one immediate parent; if multiple ``p`` match, keep one row per ``c.id``."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["id"])
        if key not in best:
            best[key] = row
            continue
        cur = best[key]
        cur_has = bool(cur.get("parent_en") or cur.get("parent_fr"))
        new_has = bool(row.get("parent_en") or row.get("parent_fr"))
        if not cur_has and new_has:
            best[key] = row
    return list(best.values())


def _row_to_example(row: dict[str, Any]) -> dict[str, str] | None:
    lv = _level_int(row["level"])
    if lv is None:
        return None
    tier = _tier_name(row["tier"])
    # Prefer DB tier; fall back to L1–L5 name table when tier missing
    level_name = tier if tier != "UNKNOWN" else (LEVEL_NAMES[lv] if lv < len(LEVEL_NAMES) else str(lv))

    parent_en = row["parent_en"] or "none"
    parent_fr = row["parent_fr"] or "none"
    pl = _level_int(row["parent_level"])
    ptier = _tier_name(row["parent_tier"])
    if parent_en == "none" and parent_fr == "none":
        parent_level_name = "none"
    elif pl is not None and ptier != "UNKNOWN":
        parent_level_name = ptier
    elif pl is not None:
        parent_level_name = LEVEL_NAMES[pl] if pl < len(LEVEL_NAMES) else str(pl)
    else:
        parent_level_name = str(row["parent_tier"] or "none")

    input_text = (
        f"French term: {row['fr']}\n"
        f"Level: {level_name} (L{lv})\n"
        f"Parent {parent_level_name}: {parent_en}\n"
        f"Parent French: {parent_fr}"
    )
    output_text = (
        f"English: {row['en']}\n"
        f"MedDRA level: L{lv} {level_name}\n"
        f"ID: {row['id']}\n"
        f"Hierarchy preserved: YES"
    )
    return {
        "instruction": _INSTRUCTION,
        "input": input_text,
        "output": output_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sft"),
        help="Directory for ontology_train.jsonl and ontology_val.jsonl (created if missing).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum total training+val examples after stratified sampling (default: 5000).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffle and sampling.")
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "ontology_train.jsonl"
    val_path = out_dir / "ontology_val.jsonl"

    rng = random.Random(args.seed)
    limit = max(1, int(args.limit))
    per_level_cap = max(1, limit // 5)

    graph = TermGraph(grounding_mode="string")
    try:
        raw = _dedupe_by_concept_id(_fetch_rows(graph))
    finally:
        graph.close()

    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped_level = 0
    for row in raw:
        lv = _level_int(row["level"])
        if lv is None:
            skipped_level += 1
            continue
        ex = _row_to_example(row)
        if ex is None:
            continue
        by_level[lv].append(ex)

    # Stratified sample: at most per_level_cap per level 1–5
    sampled: list[dict[str, str]] = []
    counts_selected: dict[int, int] = {}
    counts_available: dict[int, int] = {L: len(by_level[L]) for L in range(1, 6)}
    for L in range(1, 6):
        pool = by_level[L]
        if len(pool) <= per_level_cap:
            chosen = list(pool)
        else:
            chosen = rng.sample(pool, per_level_cap)
        counts_selected[L] = len(chosen)
        sampled.extend(chosen)

    rng.shuffle(sampled)
    split = int(len(sampled) * 0.9)
    train = sampled[:split]
    val = sampled[split:]

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Neo4j unique concepts (FR+EN+level, optional parent): {len(raw)}", file=sys.stderr)
    print(f"Skipped (level not in 1–5): {skipped_level}", file=sys.stderr)
    print("Available per level (before cap): " + ", ".join(f"L{L}={counts_available[L]}" for L in range(1, 6)), file=sys.stderr)
    print("Selected per level: " + ", ".join(f"L{L}={counts_selected[L]}" for L in range(1, 6)), file=sys.stderr)
    print(f"Total selected: {len(sampled)}  →  train: {len(train)}  val: {len(val)}", file=sys.stderr)
    print(f"Wrote {train_path}", file=sys.stderr)
    print(f"Wrote {val_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
