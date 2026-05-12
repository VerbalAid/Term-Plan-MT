#!/usr/bin/env python3
"""Call ``evaluate.py`` then ``plot_figures.py`` for each ``EVAL_RERUN_PROFILES`` row.

Used by ``rerun_all.sh``. Skips missing dirs, empty pipeline outputs, or missing segment files.

Env (optional): ``EVAL_GROUNDING_MODES``, ``HTM_VECTOR_THRESHOLDS``, ``EXTRA_EVAL_FLAGS``,
``SKIP_EVAL_PROFILES``, ``PLOT_COMET``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from metrics import (
    EVAL_RERUN_PROFILES,
    condition_name_from_results_subdir,
    eval_files_for_set,
    unpack_eval_rerun_profile,
)


def _has_pipeline_outputs(results_dir: Path, eval_file_set: str = "standard") -> bool:
    return any((results_dir / fn).is_file() for _, fn in eval_files_for_set(eval_file_set))


def _resolve_segments(root: Path, rels: tuple[str, ...]) -> Path | None:
    for rel in rels:
        p = root / rel
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root (default: repo root).",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        default=os.environ.get("EXCLUDE_SEGMENT_IDS", "48_028"),
        help="Segment ids to exclude (default: env EXCLUDE_SEGMENT_IDS or 48_028).",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    py = sys.executable
    eval_script = root / "tools" / "eval" / "evaluate.py"
    plot_script = root / "tools" / "eval" / "plot_figures.py"
    if not eval_script.is_file() or not plot_script.is_file():
        raise SystemExit(f"Missing {eval_script} or {plot_script}")

    modes = os.environ.get("EVAL_GROUNDING_MODES", "string").split()
    htm_vec = os.environ.get("HTM_VECTOR_THRESHOLDS", "").strip()
    hvf: list[str] = []
    if htm_vec:
        hvf = ["--htm-vector-thresholds", htm_vec]
    extra: list[str] = []
    raw_extra = os.environ.get("EXTRA_EVAL_FLAGS", "").strip()
    if raw_extra:
        extra = shlex.split(raw_extra)

    _skip_raw = os.environ.get("SKIP_EVAL_PROFILES", "").strip()
    _skip_profiles = {x.strip() for x in _skip_raw.split(",") if x.strip()}

    for prof in EVAL_RERUN_PROFILES:
        results_sub, seg_rels, excl_ov, eval_file_set = unpack_eval_rerun_profile(prof)
        cond = condition_name_from_results_subdir(results_sub)
        if cond in _skip_profiles:
            print(
                f"[run_eval_plot_matrix] skip (SKIP_EVAL_PROFILES): {results_sub}",
                file=sys.stderr,
            )
            continue
        rd = root / results_sub
        if not rd.is_dir():
            print(f"[run_eval_plot_matrix] skip (no dir): {results_sub}", file=sys.stderr)
            continue
        if not _has_pipeline_outputs(rd, eval_file_set):
            print(f"[run_eval_plot_matrix] skip (no pipeline JSONLs): {results_sub}", file=sys.stderr)
            continue
        seg = _resolve_segments(root, seg_rels)
        if seg is None:
            tried = ", ".join(seg_rels)
            print(
                f"[run_eval_plot_matrix] skip (no segment file for {results_sub}; tried {tried})",
                file=sys.stderr,
            )
            continue
        excl_arg = args.exclude_segment_ids if excl_ov is None else excl_ov

        for gm in modes:
            gm = gm.strip()
            if not gm:
                continue
            out_sub = "figures" if gm == "string" else f"figures_{gm}"
            od = rd / out_sub
            print("=" * 72, file=sys.stderr)
            print(f"[run_eval_plot_matrix] eval+plot  {results_sub}  grounding={gm}", file=sys.stderr)
            print(f"  segments: {seg.relative_to(root)}", file=sys.stderr)
            print(f"  exclude-segment-ids: {excl_arg!r}  eval-file-set: {eval_file_set}", file=sys.stderr)
            print("=" * 72, file=sys.stderr)

            ev_cmd = [
                py,
                str(eval_script),
                "--grounding-mode",
                gm,
                "--results-dir",
                str(rd.relative_to(root)),
                "--segments",
                str(seg.relative_to(root)),
                "--exclude-segment-ids",
                str(excl_arg),
                "--eval-file-set",
                eval_file_set,
                *hvf,
                *extra,
            ]
            subprocess.run(ev_cmd, cwd=str(root), check=True)
            pl_cmd = [
                py,
                str(plot_script),
                "--grounding-mode",
                gm,
                "--results-dir",
                str(rd.relative_to(root)),
                "--segments",
                str(seg.relative_to(root)),
                "--out-dir",
                str(od.relative_to(root)),
                "--exclude-segment-ids",
                str(excl_arg),
                "--eval-file-set",
                eval_file_set,
                *hvf,
                *extra,
            ]
            if os.environ.get("PLOT_COMET", "0") == "1":
                pl_cmd.append("--comet")
            subprocess.run(pl_cmd, cwd=str(root), check=True)


if __name__ == "__main__":
    main()
