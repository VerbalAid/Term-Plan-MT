#!/usr/bin/env python3
"""Deterministic terminology analyses (no GPU): drift, levels, missing terms, consistency.

Reads existing pipeline JSONLs under ``--results-dir`` and a segment JSONL (``--segments``).
Writes CSVs under ``<results_dir>/figures/``. All four reports use Neo4j for French→MedDRA grounding;
``--no-graph`` skips them (no embedded grounding in segment JSONL).

Example::

    PYTHONPATH=. python tools/error_analysis/analyse_terminology.py \\
      --results-dir results/ner_biollm \\
      --segments data/section48/segments_ner_biollm.jsonl

    PYTHONPATH=. python tools/error_analysis/analyse_terminology.py --no-graph \\
      --results-dir results/ner_biollm \\
      --segments data/section48/segments_ner_biollm.jsonl
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.graph import TermGraph
from pipeline.metrics.eval_io import load_results_jsonl
from pipeline.metrics.eval_manifest import EVAL_FILES
from pipeline.metrics.htm import (
    _ground_cached,
    _ref_from_concept,
    _rendering_strings,
)
from pipeline.metrics.matching import normalize_text, phrase_in_text
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _figures_dir(results_dir: Path) -> Path:
    d = _resolve(results_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _id_to_segment(segments_path: Path, exclude_segment_ids: frozenset[str] | None) -> dict[str, dict[str, Any]]:
    rows = load_all_segments(_resolve(segments_path), exclude_segment_ids=exclude_segment_ids)
    return {str(r["id"]): r for r in rows}


def _fr_term_segment_index(
    id_to_segment: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """French surface (exact ``word``) -> sorted unique segment ids containing it."""
    term_segs: dict[str, set[str]] = defaultdict(set)
    for sid, row in id_to_segment.items():
        seen_cf: set[str] = set()
        for t in row.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            k = w.casefold()
            if k in seen_cf:
                continue
            seen_cf.add(k)
            term_segs[w].add(sid)
    return {w: sorted(sids) for w, sids in term_segs.items()}


def _agreement_norm(s: str) -> str:
    """Case-fold, NFKC-ish spacing from :func:`normalize_text`, then drop punctuation (consistency_score)."""
    base = normalize_text(s)
    return re.sub(r"[^\w\s]", "", base)


def _first_rendering_in_hyp(hyp: str, graph: TermGraph, concept: dict[str, Any]) -> str | None:
    for r in _rendering_strings(graph, concept):
        if phrase_in_text(hyp, r):
            return r
    return None


def _hyp_match_level(
    hyp: str,
    graph: TermGraph,
    ref: dict[str, Any],
    found_render: str | None,
) -> int | None:
    """MedDRA ``level`` of the English node used like :func:`_score_htm_alignment`, or ``None``."""
    if not found_render:
        return None
    node = graph.get_by_name(ref["en_label"]) or graph.get_by_name(found_render)
    if not node:
        return None
    lv = node.get("level")
    if lv is None:
        return None
    try:
        return int(lv)
    except (TypeError, ValueError):
        return None


def _source_level(concept: dict[str, Any]) -> int | None:
    lv = concept.get("level")
    if lv is None:
        return None
    try:
        return int(lv)
    except (TypeError, ValueError):
        return None


def term_drift_report(
    results_dir: Path,
    segments_path: Path,
    *,
    graph: TermGraph | None = None,
    grounding_mode: str = "string",
    exclude_segment_ids: frozenset[str] | None = None,
    min_segments: int = 3,
    segment_by_id: dict[str, dict[str, Any]] | None = None,
    term_to_segs: dict[str, list[str]] | None = None,
) -> Path:
    """French terms in ``min_segments``+ segments: distinct matched MedDRA renderings in ``hyp`` per system.

    ``drift_score`` = ``(distinct_renderings - 1) / max(segments_with_term - 1, 1)`` so 0 = one stable
    rendering across segments and 1 = a different signature in every segment (for ``segments_with_term > 1``).
    """
    own_graph = graph is None
    g = graph or TermGraph(grounding_mode=grounding_mode)
    try:
        sidmap = segment_by_id if segment_by_id is not None else _id_to_segment(segments_path, exclude_segment_ids)
        tmap = term_to_segs if term_to_segs is not None else _fr_term_segment_index(sidmap)
        rd = _resolve(results_dir)
        out_path = _figures_dir(rd) / "term_drift.csv"
        ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}

        frequent = {w: sids for w, sids in tmap.items() if len(sids) >= min_segments}
        rows_out: list[dict[str, Any]] = []

        for sys_label, fname in EVAL_FILES:
            p = rd / fname
            hyp_by_id = {str(r["id"]): (r.get("hyp") or "") for r in load_results_jsonl(p)}
            for fr_term, sids in sorted(frequent.items(), key=lambda x: (-len(x[1]), x[0].casefold())):
                signatures: list[str] = []
                for sid in sids:
                    seg = sidmap.get(sid)
                    if not seg:
                        continue
                    hyp = hyp_by_id.get(sid, "")
                    fr_ctx = (seg.get("fr") or "").strip() or None
                    concept = _ground_cached(g, ground_cache, fr_term, fr_ctx)
                    if not concept:
                        signatures.append("__UNGROUNDED__")
                        continue
                    found = _first_rendering_in_hyp(hyp, g, concept)
                    signatures.append(normalize_text(found) if found else "__NONE__")
                distinct = len(set(signatures))
                nseg = len(sids)
                drift = (distinct - 1) / max(nseg - 1, 1) if nseg > 1 else 0.0
                rows_out.append(
                    {
                        "fr_term": fr_term,
                        "system": sys_label,
                        "segments_with_term": nseg,
                        "distinct_renderings": distinct,
                        "drift_score": round(drift, 6),
                    }
                )

        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["fr_term", "system", "segments_with_term", "distinct_renderings", "drift_score"],
            )
            w.writeheader()
            w.writerows(rows_out)
        return out_path
    finally:
        if own_graph:
            g.close()


def level_distribution_report(
    results_dir: Path,
    segments_path: Path,
    graph: TermGraph,
    *,
    exclude_segment_ids: frozenset[str] | None = None,
    segment_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Per system: counts of (source_level, hyp_level); flattening when source_level > hyp_level."""
    sidmap = segment_by_id if segment_by_id is not None else _id_to_segment(segments_path, exclude_segment_ids)
    rd = _resolve(results_dir)
    out_path = _figures_dir(rd) / "level_distribution.csv"
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    pair_rows: list[dict[str, Any]] = []
    summary_by_system: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {"src": [], "hyp": [], "flat": 0, "n": 0}
    )

    for sys_label, fname in EVAL_FILES:
        p = rd / fname
        if not p.is_file():
            continue
        for res in sorted(load_results_jsonl(p), key=lambda x: str(x.get("id"))):
            sid = str(res.get("id") or "")
            seg = sidmap.get(sid)
            if not seg:
                continue
            hyp = res.get("hyp") or ""
            fr_ctx = (seg.get("fr") or "").strip() or None
            seen_cf: set[str] = set()
            for t in seg.get("terms") or []:
                w = (t.get("word") or "").strip()
                if not w:
                    continue
                ck = w.casefold()
                if ck in seen_cf:
                    continue
                seen_cf.add(ck)
                concept = _ground_cached(graph, ground_cache, w, fr_ctx)
                if not concept:
                    continue
                ref = _ref_from_concept(concept)
                src_lv = _source_level(concept)
                found = _first_rendering_in_hyp(hyp, graph, concept)
                hyp_lv = _hyp_match_level(hyp, graph, ref, found)
                if src_lv is None or hyp_lv is None:
                    continue
                flat = 1 if src_lv > hyp_lv else 0
                pair_rows.append(
                    {
                        "system": sys_label,
                        "segment_id": sid,
                        "fr_term": w,
                        "source_level": src_lv,
                        "hyp_level": hyp_lv,
                        "flattening": flat,
                    }
                )
                sb = summary_by_system[sys_label]
                sb["src"].append(float(src_lv))
                sb["hyp"].append(float(hyp_lv))
                sb["flat"] += flat
                sb["n"] += 1

    agg: dict[tuple[str, int, int], int] = defaultdict(int)
    for r in pair_rows:
        key = (r["system"], int(r["source_level"]), int(r["hyp_level"]))
        agg[key] += 1

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "source_level", "hyp_level", "count", "flattening"])
        for (sys, sl, hl), c in sorted(agg.items()):
            w.writerow([sys, sl, hl, c, int(sl > hl)])

    summary_table: list[dict[str, Any]] = []
    for sys in sorted(summary_by_system):
        sb = summary_by_system[sys]
        n = int(sb["n"])
        if n == 0:
            summary_table.append(
                {
                    "system": sys,
                    "mean_source_level": float("nan"),
                    "mean_hyp_level": float("nan"),
                    "flattening_rate": float("nan"),
                    "n_spans": 0,
                }
            )
            continue
        summary_table.append(
            {
                "system": sys,
                "mean_source_level": round(statistics.mean(sb["src"]), 4),
                "mean_hyp_level": round(statistics.mean(sb["hyp"]), 4),
                "flattening_rate": round(sb["flat"] / n, 6),
                "n_spans": n,
            }
        )

    print("\nLevel distribution summary (grounded spans with source and hypothesis levels):")
    print(f"{'system':<12} {'mean_src':>10} {'mean_hyp':>10} {'flat_rate':>10} {'n':>8}")
    for row in summary_table:
        ms = row["mean_source_level"]
        mh = row["mean_hyp_level"]
        fr = row["flattening_rate"]
        print(
            f"{row['system']:<12} "
            f"{(ms if isinstance(ms, float) and not math.isnan(ms) else '--'):>10} "
            f"{(mh if isinstance(mh, float) and not math.isnan(mh) else '--'):>10} "
            f"{(fr if isinstance(fr, float) and not math.isnan(fr) else '--'):>10} "
            f"{row['n_spans']:>8}"
        )

    return out_path, summary_table


def missing_term_rate(
    results_dir: Path,
    segments_path: Path,
    graph: TermGraph,
    *,
    exclude_segment_ids: frozenset[str] | None = None,
    segment_by_id: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Share of grounded NER spans with no MedDRA rendering substring in ``hyp``."""
    sidmap = segment_by_id if segment_by_id is not None else _id_to_segment(segments_path, exclude_segment_ids)
    rd = _resolve(results_dir)
    out_path = _figures_dir(rd) / "missing_terms.csv"
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    rows_out: list[dict[str, Any]] = []

    for sys_label, fname in EVAL_FILES:
        p = rd / fname
        if not p.is_file():
            continue
        n_grounded = 0
        n_missing = 0
        for res in sorted(load_results_jsonl(p), key=lambda x: str(x.get("id"))):
            sid = str(res.get("id") or "")
            seg = sidmap.get(sid)
            if not seg:
                continue
            hyp = res.get("hyp") or ""
            fr_ctx = (seg.get("fr") or "").strip() or None
            seen_cf: set[str] = set()
            for t in seg.get("terms") or []:
                w = (t.get("word") or "").strip()
                if not w:
                    continue
                ck = w.casefold()
                if ck in seen_cf:
                    continue
                seen_cf.add(ck)
                concept = _ground_cached(graph, ground_cache, w, fr_ctx)
                if not concept:
                    continue
                n_grounded += 1
                found = _first_rendering_in_hyp(hyp, graph, concept)
                if not found:
                    n_missing += 1
        rate = (n_missing / n_grounded) if n_grounded else float("nan")
        rows_out.append(
            {
                "system": sys_label,
                "grounded_spans": n_grounded,
                "missing_spans": n_missing,
                "missing_rate": rate if not math.isnan(rate) else "",
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["system", "grounded_spans", "missing_spans", "missing_rate"],
        )
        w.writeheader()
        for row in rows_out:
            r = dict(row)
            if isinstance(r["missing_rate"], float):
                r["missing_rate"] = round(r["missing_rate"], 6)
            w.writerow(r)
    return out_path


def consistency_score(
    results_dir: Path,
    segments_path: Path,
    *,
    graph: TermGraph | None = None,
    grounding_mode: str = "string",
    exclude_segment_ids: frozenset[str] | None = None,
    min_segments: int = 3,
    segment_by_id: dict[str, dict[str, Any]] | None = None,
    term_to_segs: dict[str, list[str]] | None = None,
) -> Path:
    """Among French terms in ``min_segments``+ segments: modal fraction of identical matched rendering.

    Agreement uses :func:`_agreement_norm` on the first MedDRA rendering hit in ``hyp`` (case-fold + strip punctuation).
    """
    own_graph = graph is None
    g = graph or TermGraph(grounding_mode=grounding_mode)
    try:
        sidmap = segment_by_id if segment_by_id is not None else _id_to_segment(segments_path, exclude_segment_ids)
        tmap = term_to_segs if term_to_segs is not None else _fr_term_segment_index(sidmap)
        rd = _resolve(results_dir)
        out_path = _figures_dir(rd) / "consistency.csv"
        ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
        frequent = {w: sids for w, sids in tmap.items() if len(sids) >= min_segments}

        rows_out: list[dict[str, Any]] = []

        for sys_label, fname in EVAL_FILES:
            p = rd / fname
            hyp_by_id = {str(r["id"]): (r.get("hyp") or "") for r in load_results_jsonl(p)}
            scores: list[float] = []
            for fr_term, sids in frequent.items():
                sigs: list[str] = []
                for sid in sids:
                    seg = sidmap.get(sid)
                    if not seg:
                        continue
                    hyp = hyp_by_id.get(sid, "")
                    fr_ctx = (seg.get("fr") or "").strip() or None
                    concept = _ground_cached(g, ground_cache, fr_term, fr_ctx)
                    if not concept:
                        sigs.append("__UNGROUNDED__")
                        continue
                    found = _first_rendering_in_hyp(hyp, g, concept)
                    sigs.append(_agreement_norm(found) if found else "__NONE__")
                if not sigs:
                    continue
                counts: dict[str, int] = defaultdict(int)
                for s in sigs:
                    counts[s] += 1
                mode_freq = max(counts.values())
                frac = mode_freq / len(sigs)
                scores.append(frac)
                rows_out.append(
                    {
                        "fr_term": fr_term,
                        "system": sys_label,
                        "n_segments": len(sigs),
                        "modal_agreement_rate": round(frac, 6),
                    }
                )

            mean_cons = statistics.mean(scores) if scores else float("nan")
            rows_out.append(
                {
                    "fr_term": "__MEAN__",
                    "system": sys_label,
                    "n_segments": len(scores),
                    "modal_agreement_rate": round(mean_cons, 6) if scores else "",
                }
            )

        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["fr_term", "system", "n_segments", "modal_agreement_rate"],
            )
            w.writeheader()
            w.writerows(rows_out)
        return out_path
    finally:
        if own_graph:
            g.close()


def run_all(
    results_dir: Path,
    segments_path: Path,
    *,
    use_graph: bool = True,
    grounding_mode: str = "string",
    exclude_segment_ids: frozenset[str] | None = None,
) -> None:
    """Run analyses; print a one-page summary to stdout."""
    id_to_segment = _id_to_segment(segments_path, exclude_segment_ids)
    term_to_segs = _fr_term_segment_index(id_to_segment)
    rd = _resolve(results_dir)
    fig = _figures_dir(rd)

    print("=== Terminology analysis (deterministic) ===")
    print(f"results_dir: {rd}")
    print(f"segments:    {_resolve(segments_path)}")
    print(f"outputs:     {fig}/term_drift.csv, level_distribution.csv, missing_terms.csv, consistency.csv")
    print()

    graph: TermGraph | None = None
    if use_graph:
        graph = TermGraph(grounding_mode=grounding_mode)
    try:
        if graph is None:
            print(
                "All analyses skipped (--no-graph): French→MedDRA grounding requires Neo4j.",
                file=sys.stderr,
            )
            print()
        else:
            p1 = term_drift_report(
                results_dir,
                segments_path,
                graph=graph,
                exclude_segment_ids=exclude_segment_ids,
                segment_by_id=id_to_segment,
                term_to_segs=term_to_segs,
            )
            print(f"[1] term_drift_report  -> {p1}")

            p2, _ = level_distribution_report(
                results_dir,
                segments_path,
                graph,
                exclude_segment_ids=exclude_segment_ids,
                segment_by_id=id_to_segment,
            )
            print(f"[2] level_distribution_report -> {p2}")
            p3 = missing_term_rate(
                results_dir,
                segments_path,
                graph,
                exclude_segment_ids=exclude_segment_ids,
                segment_by_id=id_to_segment,
            )
            print(f"[3] missing_term_rate -> {p3}")

            p4 = consistency_score(
                results_dir,
                segments_path,
                graph=graph,
                exclude_segment_ids=exclude_segment_ids,
                segment_by_id=id_to_segment,
                term_to_segs=term_to_segs,
            )
            print(f"[4] consistency_score -> {p4}")
        print()

        if graph is not None:
            print("Missing-term rate (grounded spans with no rendering substring in hyp):")
            mp = _resolve(results_dir) / "figures" / "missing_terms.csv"
            if mp.is_file():
                with mp.open(encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i < 8:
                            print(line.rstrip())
            print()

            print("Consistency (__MEAN__ rows = mean modal agreement over frequent French terms):")
            cp = _resolve(results_dir) / "figures" / "consistency.csv"
            if cp.is_file():
                with cp.open(encoding="utf-8") as f:
                    for line in f:
                        if "__MEAN__" in line:
                            print(line.rstrip())
    finally:
        if graph is not None:
            graph.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic terminology reports (drift, levels, missing, consistency).")
    ap.add_argument("--results-dir", type=Path, required=True, help="e.g. results/ner_biollm")
    ap.add_argument(
        "--segments",
        type=Path,
        required=True,
        help="Segment JSONL with terms[] (e.g. data/section48/segments_ner_biollm.jsonl)",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip all reports (French grounding needs Neo4j / TermGraph).",
    )
    ap.add_argument(
        "--grounding-mode",
        default="string",
        choices=("string", "vector", "vector_llm"),
        help="Neo4j grounding mode when graph is used (default: string).",
    )
    ap.add_argument("--exclude-segment-ids", type=str, default="", help="Comma-separated segment ids to skip.")
    args = ap.parse_args()
    ex = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    run_all(
        args.results_dir,
        args.segments,
        use_graph=not args.no_graph,
        grounding_mode=args.grounding_mode,
        exclude_segment_ids=ex,
    )


if __name__ == "__main__":
    main()
