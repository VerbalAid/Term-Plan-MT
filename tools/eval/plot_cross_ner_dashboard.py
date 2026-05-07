#!/usr/bin/env python3
"""Aggregate ``scores_summary.csv`` from several ``results/ner_*`` trees into cross-condition figures.

Reads per-condition outputs produced by ``tools/eval/plot_figures.py`` (default: ``{results-dir}/figures/``).
Each CSV row is one system; ``ccr_dataset`` is identical across rows for that condition.

Unless ``--no-graph`` is set, dataset CCR for the bar chart is **recomputed** from each
pipeline’s segment JSONL (same paths as ``rerun_all.sh``) so bars reflect differing NER
extractions; ``scores_summary.csv`` values are kept only as a fallback when Neo4j is down.

If every CSV still carries the same ``ccr_dataset`` or Neo4j is unavailable, values are read
from ``ccr_snapshot.json`` beside the output (committed defaults can be replaced after a live
Neo4j run — successful recomputation overwrites that file).

Example::

    PYTHONPATH=. python tools/eval/plot_cross_ner_dashboard.py \\
        --out-dir results/cross_ner_comparison

Use ``--figures-subdir figures_vector`` after a vector-grounding rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

from pipeline.metrics.ccr import compute_ccr
from pipeline.metrics.eval_manifest import (
    EVAL_FILES,
    EVAL_RERUN_PROFILES,
    condition_name_from_results_subdir,
)
from pipeline.graph import TermGraph
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids

try:
    from neo4j.exceptions import ServiceUnavailable
except ImportError:

    class ServiceUnavailable(Exception):
        """Placeholder if neo4j is not installed."""


_NEO4J_CONN_ERRORS: tuple[type[BaseException], ...] = (
    ServiceUnavailable,
    ConnectionError,
    TimeoutError,
)

SYSTEM_ORDER = [label for label, _ in EVAL_FILES]

# Directory name under results/ → short legend label (keep in sync with ``EVAL_RERUN_PROFILES``).
CONDITION_LABELS: dict[str, str] = {
    "ner_biollm": "BioMistral prompt",
    "ner_biollm_finetuned": "FT BioMistral NER",
    "ner_baseline": "CamemBERT baseline",
    "ner_finetuned": "CamemBERT fine-tuned",
}

DEFAULT_CONDITION_ORDER = [
    condition_name_from_results_subdir(sub) for sub, _rels in EVAL_RERUN_PROFILES
]

# Colorblind-friendly NER-pipeline colours (extend when adding ``EVAL_RERUN_PROFILES`` rows).
CONDITION_COLORS: dict[str, str] = {
    "ner_biollm": "#CC79A7",  # reddish purple — BioMistral prompt
    "ner_biollm_finetuned": "#009E73",  # bluish green — FT BioMistral NER
    "ner_baseline": "#E69F00",  # orange
    "ner_finetuned": "#0072B2",  # blue
}

# Same mapping as ``tools/eval/run_eval_plot_matrix.py`` / ``rerun_all.sh``.
SEGMENTS_REL_FOR_CONDITION: dict[str, tuple[str, ...]] = {
    condition_name_from_results_subdir(sub): rels for sub, rels in EVAL_RERUN_PROFILES
}


def _resolve_segments_for_condition(cond: str) -> Path | None:
    rels = SEGMENTS_REL_FOR_CONDITION.get(cond)
    if not rels:
        return None
    for rel in rels:
        p = ROOT / rel
        if p.is_file():
            return p
    return None


def _ccr_from_segments_file(
    seg_path: Path,
    grounding_mode: str,
    exclude_segment_ids: frozenset[str],
) -> float | None:
    """Recompute dataset CCR for the NER spans in ``seg_path`` (needs live Neo4j)."""
    rows = load_all_segments(seg_path, exclude_segment_ids=exclude_segment_ids)
    graph = TermGraph(grounding_mode=grounding_mode)
    try:
        try:
            return float(compute_ccr(rows, graph))
        except _NEO4J_CONN_ERRORS:
            return None
    finally:
        graph.close()


def _ccr_values_collapsed(ccr_by_cond: dict[str, float], *, tol: float = 1e-4) -> bool:
    vals = [float(v) for v in ccr_by_cond.values()]
    if len(vals) < 2:
        return False
    return max(vals) - min(vals) < tol


def _load_ccr_snapshot(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k in DEFAULT_CONDITION_ORDER:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out[str(k)] = fv
            except (TypeError, ValueError):
                pass
    return out or None


def _write_ccr_snapshot(path: Path, ccr_by_cond: dict[str, float]) -> None:
    snap = {k: round(float(ccr_by_cond[k]), 6) for k in DEFAULT_CONDITION_ORDER if k in ccr_by_cond}
    if not snap:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_ccr_from_segment_files(
    ccr_by_cond: dict[str, float],
    *,
    by_cond_keys: set[str],
    grounding_mode: str,
    no_graph: bool,
    exclude_segment_ids: frozenset[str],
) -> tuple[dict[str, float], int]:
    """Overwrite CCR with values from each condition’s segment file when Neo4j is available."""
    if no_graph:
        return ccr_by_cond, 0
    out = dict(ccr_by_cond)
    n_ok = 0
    for cond in DEFAULT_CONDITION_ORDER:
        if cond not in by_cond_keys:
            continue
        seg = _resolve_segments_for_condition(cond)
        if seg is None:
            continue
        v = _ccr_from_segments_file(seg, grounding_mode, exclude_segment_ids)
        if v is None:
            continue
        out[cond] = v
        n_ok += 1
    return out, n_ok


def _y_axis_truncation_break(ax: plt.Axes) -> None:
    """Double-slash when y-axis lower bound excludes zero (matches plot_figures.py)."""
    y0, _y1 = ax.get_ylim()
    if y0 <= 0.02:
        return
    xlim = ax.get_xlim()
    sx = xlim[1] - xlim[0]
    sy = ax.get_ylim()[1] - y0
    dx = 0.012 * sx
    dy = 0.018 * sy
    x0 = xlim[0] - 0.008 * sx
    for ox in (0.0, dx * 0.65):
        ax.plot(
            [x0 + ox, x0 + ox + dx],
            [y0, y0 + dy],
            color="#333333",
            lw=1.05,
            clip_on=False,
            zorder=200,
            solid_capstyle="butt",
        )


def _parse_float(x: str | None) -> float | None:
    if x is None or x.strip() == "" or x.strip().lower() == "nan":
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def _load_scores_csv(path: Path) -> tuple[list[dict[str, object]], float | None]:
    rows: list[dict[str, object]] = []
    ccr_ds: float | None = None
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            label = row.get("label", "").strip()
            if not label:
                continue
            bleu = _parse_float(row.get("bleu"))
            chrf = _parse_float(row.get("chrf"))
            comet = _parse_float(row.get("comet"))
            htm = _parse_float(row.get("htm"))
            mean_s = _parse_float(row.get("mean_s"))
            p95_s = _parse_float(row.get("p95_s"))
            if ccr_ds is None:
                ccr_ds = _parse_float(row.get("ccr_dataset"))
            rows.append(
                {
                    "label": label,
                    "bleu": bleu,
                    "chrf": chrf,
                    "comet": comet,
                    "htm": htm,
                    "mean_s": mean_s,
                    "p95_s": p95_s,
                }
            )
    return rows, ccr_ds


def _matrix(
    by_cond: dict[str, list[dict[str, object]]],
    metric: str,
) -> tuple[np.ndarray, list[str]]:
    """Shape (n_systems, n_conditions present); NaN where missing."""
    keys = [k for k in DEFAULT_CONDITION_ORDER if k in by_cond and by_cond[k]]
    M = np.full((len(SYSTEM_ORDER), len(keys)), np.nan, dtype=float)
    for j, ck in enumerate(keys):
        lab_to_row = {str(r["label"]): r for r in by_cond[ck]}
        for i, sys in enumerate(SYSTEM_ORDER):
            if sys in lab_to_row:
                v = lab_to_row[sys].get(metric)
                if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and math.isnan(v)
                ):
                    M[i, j] = float(v)
                elif v is not None:
                    try:
                        M[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass
    return M, keys


def _plot_metric_bars_on_ax(
    ax: plt.Axes,
    *,
    by_cond: dict[str, list[dict[str, object]]],
    metric_key: str,
    ylim: tuple[float, float] | None,
    show_xticklabels: bool,
) -> None:
    """Draw grouped bars (bar colours = NER pipeline)."""
    M, cond_keys = _matrix(by_cond, metric_key)
    if not cond_keys:
        return
    x = np.arange(len(SYSTEM_ORDER), dtype=float)
    n_b = len(cond_keys)
    total_w = 0.75
    w = total_w / max(n_b, 1)
    for j, ck in enumerate(cond_keys):
        offset = (j - (n_b - 1) / 2.0) * w
        color = CONDITION_COLORS.get(ck, "#333333")
        ax.bar(
            x + offset,
            M[:, j],
            width=w * 0.92,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(SYSTEM_ORDER, rotation=0, ha="center", fontsize=9)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", alpha=0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_metric_grid(
    by_cond: dict[str, list[dict[str, object]]],
    out_base: Path,
    dpi: int,
    fmt: str,
) -> None:
    """Cross-NER grid: colours = **NER pipeline** (segment source); x-axis = **MT system** (s1…s5_mistral).

    This differs from per-condition trade-off plots in ``plot_figures.py``, where colours = systems.
    Here dark blue bars are **Baseline CamemBERT** NER — not “S1 NLLB”. S1 NLLB is the **first cluster** on the x-axis.
    """
    metrics_spec = [
        ("bleu", "BLEU", None),
        ("chrf", "chrF++", None),
        ("comet", "COMET", None),
        ("htm", "HTM", (0.0, 1.05)),
        ("mean_s", "Mean inference (s/segment)", None),
    ]
    present: list[tuple[str, str, tuple[float, float] | None]] = []
    for key, title, lim in metrics_spec:
        M, cond_keys = _matrix(by_cond, key)
        if not cond_keys:
            continue
        if key == "comet" and np.all(np.isnan(M)):
            continue
        present.append((key, title, lim))

    if not present:
        return

    keys_only = [p[0] for p in present]
    has_comet = "comet" in keys_only
    has_mean = "mean_s" in keys_only

    panel_ratios: list[float] = [1.0]
    if has_comet:
        panel_ratios.append(1.0)
    if "htm" in keys_only:
        panel_ratios.append(1.0)
    if has_mean:
        panel_ratios.append(1.18)

    legend_patches = [
        Patch(
            facecolor=CONDITION_COLORS[k],
            edgecolor="white",
            linewidth=0.6,
            label=CONDITION_LABELS[k],
        )
        for k in DEFAULT_CONDITION_ORDER
        if k in by_cond
    ]

    # Rows: [ legend span ] [ BLEU | chrF++ ] [ COMET span? ] [ HTM span ] [ mean span ]
    legend_ratio = 0.38
    height_ratios = [legend_ratio] + panel_ratios
    fig_h = 3.05 * len(panel_ratios) + 1.05
    fig = plt.figure(figsize=(11.5, fig_h))
    gs = fig.add_gridspec(
        len(height_ratios),
        2,
        height_ratios=height_ratios,
        hspace=0.42,
        wspace=0.22,
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )

    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    if legend_patches:
        ncol_leg = min(4, len(legend_patches))
        ax_leg.legend(
            handles=legend_patches,
            loc="center",
            ncol=ncol_leg,
            fontsize=9,
            title="NER pipeline (segment source)",
            title_fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
        )

    row = 1
    ylim_htm_fallback = (0.0, 1.05)
    # BLEU | chrF++
    if "bleu" in keys_only:
        ax_b = fig.add_subplot(gs[row, 0])
        _plot_metric_bars_on_ax(
            ax_b,
            by_cond=by_cond,
            metric_key="bleu",
            ylim=None,
            show_xticklabels=False,
        )
        ax_b.set_title("BLEU", fontsize=11, fontweight="bold")
        ax_b.set_ylabel("BLEU")
    if "chrf" in keys_only:
        ax_c = fig.add_subplot(gs[row, 1])
        _plot_metric_bars_on_ax(
            ax_c,
            by_cond=by_cond,
            metric_key="chrf",
            ylim=None,
            show_xticklabels=False,
        )
        ax_c.set_title("chrF++", fontsize=11, fontweight="bold")
        ax_c.set_ylabel("chrF++")
    row += 1

    if has_comet:
        ax_co = fig.add_subplot(gs[row, :])
        _plot_metric_bars_on_ax(
            ax_co,
            by_cond=by_cond,
            metric_key="comet",
            ylim=None,
            show_xticklabels=False,
        )
        ax_co.set_title("COMET", fontsize=11, fontweight="bold")
        ax_co.set_ylabel("COMET")
        row += 1

    if "htm" in keys_only:
        ax_h = fig.add_subplot(gs[row, :])
        _plot_metric_bars_on_ax(
            ax_h,
            by_cond=by_cond,
            metric_key="htm",
            ylim=None,
            show_xticklabels=False,
        )
        ax_h.set_title("HTM (0–1; y-axis zoomed to bar range)", fontsize=11, fontweight="bold")
        ax_h.set_ylabel("HTM")
        M_h, _ = _matrix(by_cond, "htm")
        vmin_h = float(np.nanmin(M_h))
        vmax_h = float(np.nanmax(M_h))
        if math.isnan(vmin_h) or math.isnan(vmax_h):
            ax_h.set_ylim(ylim_htm_fallback)
        else:
            span_h = max(vmax_h - vmin_h, 1e-9)
            pad_h = max(0.02, span_h * 0.22, vmin_h * 0.05)
            lo = max(0.0, vmin_h - pad_h)
            hi = min(1.05, vmax_h + pad_h)
            if hi - lo < 0.07:
                hi = min(1.05, lo + 0.10)
            ax_h.set_ylim(lo, hi)
            if lo > 0.02:
                _y_axis_truncation_break(ax_h)
        row += 1

    if has_mean:
        ax_m = fig.add_subplot(gs[row, :])
        _plot_metric_bars_on_ax(
            ax_m,
            by_cond=by_cond,
            metric_key="mean_s",
            ylim=None,
            show_xticklabels=True,
        )
        ax_m.set_title("Mean inference time", fontsize=11, fontweight="bold")
        ax_m.set_ylabel("Seconds / segment")

    fig.suptitle(
        "Performance metrics by NER pipeline and MT system",
        fontsize=13,
        y=0.97,
    )

    for ext in fmt.split(","):
        e = ext.strip()
        if not e:
            continue
        fig.savefig(f"{out_base}_metrics_grid.{e}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_ccr_bars(
    ccr_by_cond: dict[str, float],
    out_path: Path,
    dpi: int,
) -> None:
    if not ccr_by_cond:
        return
    keys = [k for k in DEFAULT_CONDITION_ORDER if k in ccr_by_cond]
    if not keys:
        return
    labels = [CONDITION_LABELS.get(k, k) for k in keys]
    vals = [ccr_by_cond[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
    x = np.arange(len(keys))
    colors = [CONDITION_COLORS.get(k, "#4c72b0") for k in keys]
    ax.bar(x, vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CCR (dataset — NER spans grounded in MedDRA)")
    ax.set_title(
        "Dataset CCR by NER pipeline\n(same for every MT system within a pipeline)",
        fontsize=11,
        pad=10,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.35)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-NER comparison figures from scores_summary.csv files.")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Directory containing ner_* subfolders (default: results/).",
    )
    ap.add_argument(
        "--figures-subdir",
        default="figures",
        help="Subfolder under each ner_* dir holding scores_summary.csv (default: figures).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: {results-root}/cross_ner_comparison).",
    )
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--format", default="png,pdf", help="Comma-separated figure formats.")
    ap.add_argument(
        "--grounding-mode",
        choices=["string", "vector", "vector_llm"],
        default="string",
        help="Neo4j grounding when recomputing CCR from segment JSONLs (default: string).",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="Keep ccr_dataset from scores_summary.csv only (no Neo4j CCR refresh).",
    )
    ap.add_argument(
        "--ccr-snapshot",
        type=Path,
        default=None,
        help=(
            "JSON map ner_* → CCR float. Used when Neo4j refresh fails or every CSV carries the "
            "same ccr_dataset (default: {out-dir}/ccr_snapshot.json if present)."
        ),
    )
    ap.add_argument(
        "--no-ccr-snapshot-fallback",
        action="store_true",
        help="Do not merge ccr_snapshot.json when recomputation fails or values are collapsed.",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids excluded from CCR recomputation (match pipeline/evaluate). Example: 48_028",
    )
    args = ap.parse_args()

    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = results_root / "cross_ner_comparison"
    else:
        out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cond: dict[str, list[dict[str, object]]] = {}
    ccr_by_cond: dict[str, float] = {}

    exclude_seg = parse_exclude_segment_ids(args.exclude_segment_ids or None)

    if not results_root.is_dir():
        raise SystemExit(f"Results root not found: {results_root}")

    for p in sorted(results_root.glob("ner_*")):
        if not p.is_dir():
            continue
        key = p.name
        csv_path = p / args.figures_subdir / "scores_summary.csv"
        if not csv_path.is_file():
            continue
        rows, ccr = _load_scores_csv(csv_path)
        if not rows:
            continue
        by_cond[key] = rows
        if ccr is not None:
            ccr_by_cond[key] = float(ccr)

    ccr_by_cond, n_ccr_refresh = _refresh_ccr_from_segment_files(
        ccr_by_cond,
        by_cond_keys=set(by_cond.keys()),
        grounding_mode=args.grounding_mode,
        no_graph=args.no_graph,
        exclude_segment_ids=exclude_seg,
    )
    if not args.no_graph and n_ccr_refresh == 0:
        print(
            "[plot_cross_ner_dashboard] CCR bars use scores_summary.csv only "
            "(Neo4j unreachable or segment JSONLs missing). "
            "Per-pipeline CCR needs live Neo4j + data/section48/segments_ner_*.jsonl.",
            file=sys.stderr,
        )
    elif n_ccr_refresh > 0:
        print(
            f"[plot_cross_ner_dashboard] Recomputed dataset CCR from segment JSONLs ({n_ccr_refresh} pipeline(s)).",
            file=sys.stderr,
        )

    snap_path = args.ccr_snapshot
    if snap_path is None:
        snap_path = out_dir / "ccr_snapshot.json"
    else:
        snap_path = snap_path if snap_path.is_absolute() else ROOT / snap_path

    if n_ccr_refresh > 0 and ccr_by_cond:
        _write_ccr_snapshot(snap_path, ccr_by_cond)

    use_snap = (
        not args.no_ccr_snapshot_fallback
        and snap_path.is_file()
        and (
            n_ccr_refresh == 0
            or _ccr_values_collapsed(ccr_by_cond)
        )
    )
    if use_snap:
        loaded = _load_ccr_snapshot(snap_path)
        if loaded:
            for k in DEFAULT_CONDITION_ORDER:
                if k in by_cond and k in loaded:
                    ccr_by_cond[k] = loaded[k]
            print(
                f"[plot_cross_ner_dashboard] Applied CCR values from {snap_path} "
                "(Neo4j offline or identical CSV ccr_dataset across pipelines).",
                file=sys.stderr,
            )

    if not by_cond:
        raise SystemExit(
            f"No scores_summary.csv files found under {results_root}/*/ {args.figures_subdir}/ — "
            "run rerun_all.sh or plot_figures.py first."
        )

    base = out_dir / "cross_ner"
    plot_metric_grid(by_cond, base, args.dpi, args.format)
    plot_ccr_bars(ccr_by_cond, out_dir / "cross_ner_ccr_dataset.png", args.dpi)

    # Small manifest for provenance
    man = out_dir / "sources.txt"
    lines = [
        f"figures_subdir={args.figures_subdir}",
        f"ccr_snapshot_fallback={snap_path.name} (used when Neo4j/CSV give identical CCR)",
        "conditions_loaded:",
    ]
    for k in DEFAULT_CONDITION_ORDER:
        if k in by_cond:
            lines.append(f"  {k}")
    man.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir} (cross_ner_metrics_grid.*, cross_ner_ccr_dataset.png, sources.txt)")


if __name__ == "__main__":
    main()
