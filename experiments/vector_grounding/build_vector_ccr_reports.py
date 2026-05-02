#!/usr/bin/env python3
"""Write vector_ccr_threshold_sweep.json and vector_ccr_all_models.json for section 4.8."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_SECTION = ROOT / "data" / "section48"

# Import compare_neo4j_grounding_ccr (shared helpers) from french_medical_ner
_spec = importlib.util.spec_from_file_location(
    "compare_neo4j_grounding_ccr",
    ROOT / "experiments" / "french_medical_ner" / "compare_neo4j_grounding_ccr.py",
)
_compare = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_compare)
require_vector_index = _compare.require_vector_index
vector_ccr_stats_with_graph = _compare.vector_ccr_stats_with_graph
run_ccr_modes = _compare.run_ccr_modes

from pipeline.graph import TermGraph
from pipeline.systems.data_io import load_all_segments

# Paper-facing segment lists only (see archive/data/section48/ for retired extractors).
NER_FILES = [
    "segments_ner_biollm.jsonl",
    "segments_ner_unsloth.jsonl",
    "segments_ner_unsloth_full.jsonl",
]

SWEEP_THRESHOLDS = (0.70, 0.75, 0.80)
# Search for an informative mid-range threshold (cosine on span queries).
GRID_LOW_HIGH = (0.25, 0.60)


def _grid_thresholds() -> list[float]:
    """Candidate cosine thresholds in [0.25, 0.60] for cross-model spread search."""
    return [0.25, 0.35, 0.45, 0.55, 0.60]


def _pick_threshold_vector_ccr(
    paths: list[Path],
) -> tuple[float, str, dict[str, Any]]:
    """Pick a cosine threshold using one reference NER file (default: biollm).

    Full multi-file × multi-threshold grids are prohibitively slow (each pass embeds
    every span and hits Neo4j). Instead we scan the grid on ``segments_ner_biollm.jsonl``
    (or the first available path) and choose the threshold whose vector CCR is closest
    to a mid-band target (~0.82) while staying away from trivial 0 or 1 plateaus.
    """
    ref_path = next((p for p in paths if p.name == "segments_ner_biollm.jsonl"), paths[0])
    rows = load_all_segments(ref_path)
    grid = _grid_thresholds()
    g = TermGraph(grounding_mode="vector", vector_score_threshold=grid[0])
    scores: dict[float, float] = {}
    try:
        for thr in grid:
            scores[thr] = float(vector_ccr_stats_with_graph(rows, thr, g)["ccr"])
    finally:
        g.close()

    target = 0.82
    candidates = [
        thr
        for thr in grid
        if 0.05 < scores[thr] < 0.995 and not (scores[thr] <= 0.0 or scores[thr] >= 1.0)
    ]
    pool = candidates if candidates else list(grid)
    best_thr = min(pool, key=lambda t: abs(scores[t] - target))
    rationale = (
        f"reference={ref_path.name}; target vector CCR≈{target}; "
        f"closest at cos≥{best_thr:.2f} (CCR={scores[best_thr]:.4f})"
    )
    meta = {
        "reference_file": ref_path.name,
        "grid": [f"{t:.2f}" for t in grid],
        "vector_ccr_by_threshold": {f"{t:.2f}": scores[t] for t in grid},
    }
    return best_thr, rationale, meta


def _modes_payload(
    rows: list[dict[str, Any]],
    thr: float,
    ambiguous_scratch: Path,
    modes: tuple[str, ...] = ("string", "vector", "vector_llm"),
) -> dict[str, Any]:
    results, _ = run_ccr_modes(
        rows,
        vector_threshold=thr,
        ambiguous_out=ambiguous_scratch,
        modes=modes,
    )
    out: dict[str, Any] = {}
    for mode, ccr, amb, fb, mc, n_ext, n_gr in results:
        row: dict[str, Any] = {
            "ccr": round(float(ccr), 6),
            "grounded": int(n_gr),
            "total": int(n_ext),
        }
        if mode == "string":
            row["ambiguous_warnings"] = amb
        else:
            row["fallbacks"] = fb
            row["mean_candidates"] = round(mc, 6)
        out[mode] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Do not rewrite data/section48/vector_ccr_threshold_sweep.json (faster resume).",
    )
    ap.add_argument(
        "--vector-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Override cosine threshold; skips biollm grid selection (e.g. 0.55).",
    )
    args = ap.parse_args()

    skip_llm = os.environ.get("VECTOR_CCR_SKIP_LLM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    print("vector_ccr_reports: checking Neo4j vector index...", flush=True)
    require_vector_index()

    paths = []
    missing: list[str] = []
    for name in NER_FILES:
        p = _SECTION / name
        if p.is_file():
            paths.append(p)
        else:
            missing.append(name)

    if not paths:
        raise SystemExit(f"No NER segment files found under {_SECTION}")

    # --- (1) Reference sweep on biollm (matches compare_neo4j_grounding_ccr default) ---
    ref = _SECTION / "segments_ner_biollm.jsonl"
    if not ref.is_file():
        ref = paths[0]
    SWEEP_PATH = _SECTION / "vector_ccr_threshold_sweep.json"
    if not args.skip_sweep:
        rows_ref = load_all_segments(ref)
        sweep_obj: dict[str, dict[str, Any]] = {}
        g_sweep = TermGraph(grounding_mode="vector", vector_score_threshold=float(SWEEP_THRESHOLDS[0]))
        try:
            for thr in SWEEP_THRESHOLDS:
                sweep_obj[f"{thr:.2f}"] = vector_ccr_stats_with_graph(rows_ref, thr, g_sweep)
        finally:
            g_sweep.close()
        SWEEP_PATH.write_text(json.dumps(sweep_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {SWEEP_PATH}")
    else:
        print(f"Skipped sweep (--skip-sweep); keeping existing {SWEEP_PATH}", flush=True)

    # --- (2) Threshold selection ---
    if args.vector_threshold is not None:
        thr_sel = float(args.vector_threshold)
        rationale = f"CLI override --vector-threshold={thr_sel}"
        grid_meta = {"cli_override": True}
        print(f"vector_ccr_reports: threshold_selected={thr_sel} (override)", flush=True)
    else:
        print("vector_ccr_reports: selecting threshold (biollm reference grid)...", flush=True)
        thr_sel, rationale, grid_meta = _pick_threshold_vector_ccr(paths)
        print(f"vector_ccr_reports: threshold_selected={thr_sel}", flush=True)

    amb_scratch = _SECTION / ".ambiguous_scratch_build_vector_ccr.txt"
    report: dict[str, Any] = {
        "threshold_selected": thr_sel,
        "threshold_rationale": rationale,
        "threshold_search": {
            "range": list(GRID_LOW_HIGH),
            "grid": [f"{t:.2f}" for t in _grid_thresholds()],
        },
        "grid_exploration": grid_meta,
        "missing_segment_files": missing,
        "vector_llm_included": not skip_llm,
        "models": {},
    }

    for p in paths:
        print(f"vector_ccr_reports: evaluating {p.name} ...", flush=True)
        rows = load_all_segments(p)
        if skip_llm:
            modes = ("string", "vector")
            report["models"][p.name] = _modes_payload(rows, thr_sel, amb_scratch, modes=modes)
        else:
            try:
                report["models"][p.name] = _modes_payload(rows, thr_sel, amb_scratch)
            except Exception as e:
                report["models"][p.name] = {"error": str(e), "type": type(e).__name__}

    ALL_PATH = _SECTION / "vector_ccr_all_models.json"
    ALL_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ALL_PATH} (threshold={thr_sel})")
    if amb_scratch.exists():
        try:
            amb_scratch.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
