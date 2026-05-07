#!/usr/bin/env python3
"""Compare ambiguous French MedDRA keys across two NER segment files (Neo4j string cache).

``TermGraph`` logs warnings like ``Ambiguous FR grounding: 'fatigue' maps to 2 distinct MedDRA concepts``
when the normalized French label for a NER span matches **multiple** concept nodes in the graph.
This script loads the same ambiguity sets (via the string grounding cache), counts how many NER
spans hit each ambiguous key for **two** JSONLs (e.g. BioMistral prompt vs Unsloth FT), and checks
whether Stage-3 **planning locks** exist for the exact surface ``word`` strings.

Requires Neo4j (same as ``evaluate.py``). Run from repo root::

    PYTHONPATH=. python tools/error_analysis/report_ambiguous_grounding.py \\
      --segments-a data/section48/segments_ner_biollm.jsonl \\
      --label-a ner_biollm \\
      --segments-b data/section48/segments_ner_unsloth.jsonl \\
      --label-b ner_biollm_finetuned \\
      --locks data/section48/planning_locks.json \\
      --out-csv error_analysis/ambiguous_grounding_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.graph import TermGraph, normalize_fr_for_grounding
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids


def _term_keys(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, set[str]], dict[str, str]]:
    """normalized_key -> count; normalized_key -> example surface forms; normalized_key -> one fr snippet."""
    counts: dict[str, int] = defaultdict(int)
    surfaces: dict[str, set[str]] = defaultdict(set)
    snippet: dict[str, str] = {}
    for row in rows:
        fr_ctx = (row.get("fr") or "").strip().replace("\n", " ")[:400]
        for t in row.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = normalize_fr_for_grounding(w)
            if not key:
                continue
            counts[key] += 1
            surfaces[key].add(w)
            snippet.setdefault(key, fr_ctx)
    return dict(counts), {k: v for k, v in surfaces.items()}, snippet


def _load_locks(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out = {str(k): str(v) for k, v in data.items() if k != "_segments_mtime"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare ambiguous French MedDRA keys across two NER segment JSONLs; "
            "optional planning_locks coverage (Neo4j required)."
        )
    )
    ap.add_argument(
        "--segments-a",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner_biollm.jsonl",
    )
    ap.add_argument("--label-a", default="ner_biollm", help="Short name for condition A (CSV column prefix).")
    ap.add_argument(
        "--segments-b",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner_unsloth.jsonl",
    )
    ap.add_argument("--label-b", default="ner_biollm_finetuned", help="Short name for condition B.")
    ap.add_argument(
        "--locks",
        type=Path,
        default=ROOT / "data" / "section48" / "planning_locks.json",
        help="Optional planning_locks.json (French surface -> English lock).",
    )
    ap.add_argument("--out-csv", type=Path, default=ROOT / "error_analysis" / "ambiguous_grounding_report.csv")
    ap.add_argument("--grounding-mode", default="string", choices=("string", "vector", "vector_llm"))
    ap.add_argument("--exclude-segment-ids", default="", help="Comma-separated segment ids to skip.")
    args = ap.parse_args()

    exclude = parse_exclude_segment_ids(args.exclude_segment_ids)
    rows_a = load_all_segments(args.segments_a, exclude_segment_ids=exclude)
    rows_b = load_all_segments(args.segments_b, exclude_segment_ids=exclude)

    graph = TermGraph(grounding_mode=args.grounding_mode)
    try:
        graph.ground("__termplan_cache_warm__", context=None)
        ambiguous_keys = frozenset(graph._ambiguous_norms)  # noqa: SLF001
        n_concepts = dict(graph._ambiguous_n_concepts)  # noqa: SLF001
    finally:
        graph.close()

    counts_a, surfaces_a, snip_a = _term_keys(rows_a)
    counts_b, surfaces_b, snip_b = _term_keys(rows_b)

    locks = _load_locks(args.locks)

    keys_union = sorted(set(counts_a) | set(counts_b))
    amb_a_only = sum(v for k, v in counts_a.items() if k in ambiguous_keys)
    amb_b_only = sum(v for k, v in counts_b.items() if k in ambiguous_keys)
    uniq_a = sum(1 for k in counts_a if k in ambiguous_keys)
    uniq_b = sum(1 for k in counts_b if k in ambiguous_keys)

    print("Ambiguous grounding report (normalized FR key matches multiple MedDRA concepts in graph cache)")
    print(f"  Graph mode: {args.grounding_mode}")
    print(f"  Ambiguous keys in graph: {len(ambiguous_keys)}")
    print(f"  Condition A ({args.label_a}): {args.segments_a}")
    print(f"    NER spans on ambiguous keys: {amb_a_only}  (unique keys touched: {uniq_a})")
    print(f"  Condition B ({args.label_b}): {args.segments_b}")
    print(f"    NER spans on ambiguous keys: {amb_b_only}  (unique keys touched: {uniq_b})")
    print(f"  Planning locks loaded: {len(locks)} entries from {args.locks}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    la, lb = args.label_a, args.label_b
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "normalized_key",
                "n_meddra_concepts",
                f"span_count_{la}",
                f"span_count_{lb}",
                "example_surface_a",
                "example_surface_b",
                f"has_planning_lock_{la}",
                f"has_planning_lock_{lb}",
                "fr_snippet_a",
                "fr_snippet_b",
            ]
        )
        for key in keys_union:
            if key not in ambiguous_keys:
                continue
            ca, cb = counts_a.get(key, 0), counts_b.get(key, 0)
            if ca == 0 and cb == 0:
                continue
            exa = next(iter(surfaces_a.get(key, [])), "")
            exb = next(iter(surfaces_b.get(key, [])), "")
            lock_a = any((s in locks) for s in surfaces_a.get(key, ()))
            lock_b = any((s in locks) for s in surfaces_b.get(key, ()))
            w.writerow(
                [
                    key,
                    n_concepts.get(key, 0),
                    ca,
                    cb,
                    exa,
                    exb,
                    int(lock_a),
                    int(lock_b),
                    snip_a.get(key, ""),
                    snip_b.get(key, ""),
                ]
            )

    print(f"Wrote {args.out_csv} (one row per ambiguous key with any NER span in A or B).")


if __name__ == "__main__":
    main()
