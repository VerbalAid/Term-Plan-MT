#!/usr/bin/env python3
"""List NER spans whose normalized French key is ambiguous in Neo4j (multiple concepts).

Writes up to ``--n`` example rows with **segment_id** and exact **term** (surface ``word``)
for manual MedDRA lookup. Requires Neo4j (same as ``evaluate.py``).

Example::

    PYTHONPATH=. python tools/error_analysis/list_ambiguous_spans.py \\
      --segments data/section48/segments_ner_biollm.jsonl \\
      --n 10 \\
      --out-md error_analysis/ambiguous_spans_top10.md \\
      --out-csv error_analysis/ambiguous_spans_top10.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.graph import TermGraph, normalize_fr_for_grounding
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="List segment_id + NER term for ambiguous FR grounding keys.")
    ap.add_argument(
        "--segments",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner_biollm.jsonl",
    )
    ap.add_argument("--n", type=int, default=10, help="Max rows to emit (default 10).")
    ap.add_argument("--grounding-mode", default="string", choices=("string", "vector", "vector_llm"))
    ap.add_argument("--exclude-segment-ids", default="", help="Comma-separated segment ids to skip.")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "error_analysis" / "ambiguous_spans_top10.csv")
    ap.add_argument("--out-md", type=Path, default=ROOT / "error_analysis" / "ambiguous_spans_top10.md")
    args = ap.parse_args()

    seg_path = args.segments if args.segments.is_absolute() else ROOT / args.segments
    if not seg_path.is_file():
        raise SystemExit(f"Missing segments file: {seg_path}")

    exclude = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    rows = load_all_segments(seg_path, exclude_segment_ids=exclude)

    graph = TermGraph(grounding_mode=args.grounding_mode)
    try:
        graph.ground("__termplan_cache_warm__", context=None)
        ambiguous_keys = frozenset(graph._ambiguous_norms)  # noqa: SLF001
        n_concepts = dict(graph._ambiguous_n_concepts)  # noqa: SLF001
    finally:
        graph.close()

    picked: list[tuple[str, str, str, int, str]] = []
    seen_pair: set[tuple[str, str]] = set()
    used_norm: set[str] = set()

    def _try_pick(require_new_key: bool) -> None:
        for row in rows:
            sid = str(row.get("id") or "").strip()
            fr = (row.get("fr") or "").strip().replace("\n", " ")
            excerpt = fr[:240] + ("…" if len(fr) > 240 else "")
            for t in row.get("terms") or []:
                w = (t.get("word") or "").strip()
                if not w or not sid:
                    continue
                key = normalize_fr_for_grounding(w)
                if not key or key not in ambiguous_keys:
                    continue
                if require_new_key and key in used_norm:
                    continue
                pair = (sid, w)
                if pair in seen_pair:
                    continue
                seen_pair.add(pair)
                used_norm.add(key)
                picked.append((sid, w, key, int(n_concepts.get(key, 0)), excerpt))
                if len(picked) >= args.n:
                    return

    _try_pick(require_new_key=True)
    if len(picked) < args.n:
        _try_pick(require_new_key=False)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "term_surface", "normalized_key", "n_meddra_concepts", "fr_excerpt"])
        for sid, surface, key, nc, ex in picked:
            writer.writerow([sid, surface, key, nc, ex])

    lines = [
        "# Ambiguous grounding — sample NER spans",
        "",
        f"Segments file: `{seg_path.relative_to(ROOT)}`",
        f"Grounding mode: `{args.grounding_mode}`",
        f"Ambiguous keys in graph cache: **{len(ambiguous_keys)}**",
        "",
        "Each row is one **(segment_id, term)** pair where the normalized French string maps to **more than one** MedDRA concept in the string-grounding cache. Use the MedDRA browser to see competing PTs/LLTs.",
        "",
        "| # | segment_id | term (NER surface) | normalized_key | # concepts | French excerpt |",
        "|---|------------|--------------------|----------------|------------|----------------|",
    ]
    for i, (sid, surface, key, nc, ex) in enumerate(picked, start=1):
        ex_esc = ex.replace("|", "\\|")
        lines.append(f"| {i} | `{sid}` | `{surface}` | `{key}` | {nc} | {ex_esc} |")
    if not picked:
        lines.append("\n*(No ambiguous-key spans in this JSONL under current graph cache.)*\n")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_csv} and {args.out_md} ({len(picked)} rows).")


if __name__ == "__main__":
    main()
