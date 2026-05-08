#!/usr/bin/env python3
"""Figures + ``scores_summary.csv`` for one ``--results-dir``.

Run ``evaluate.py`` first (same ``--segments`` / ``--grounding-mode``). Writes PNG/PDF under
``--out-dir`` (default: ``<results-dir>/figures/``) and a small markdown summary table.

Optional COMET: ``--comet`` (slow). Without Neo4j, pass ``--no-graph`` like eval.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from matplotlib.patches import Patch

# Non-interactive backend first — avoids Qt / QSocketNotifier noise on Fedora (wayland/QtAgg).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.comet_score import corpus_comet_da as _try_comet
from pipeline.metrics.corpus_scores import corpus_bleu, corpus_chrf
from pipeline.metrics.eval_io import align_src_hyp_ref, load_results_jsonl
from pipeline.metrics.eval_manifest import EVAL_FILES
from pipeline.metrics.eval_table import (
    NEO4J_CONN_ERRORS,
    collect_system_metric_rows,
    display_label_for_system as _label,
    neo4j_connection_help,
    write_scores_summary_csv,
)
from pipeline.metrics.htm import (
    htm_vector_column_key,
    parse_cosine_thresholds_csv,
)
from pipeline.graph import TermGraph
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids
from pipeline.systems.inference_timing import inference_mean_p95


def _htm_unavailable(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        isinstance(r.get("htm"), float) and math.isnan(r["htm"]) for r in rows
    )


def _parse_csv_float(x: str | None) -> float | None:
    if x is None or str(x).strip() == "" or str(x).strip().lower() == "nan":
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def scan_global_scatter_limits(scan_root: Path) -> dict[str, tuple[float, float]]:
    """Collect chrF, HTM, and mean inference ranges from primary NER conditions only.

    Uses ``ner_biollm`` and ``ner_biollm_finetuned`` ``scores_summary.csv`` (same reproduce path
    as ``rerun_all.sh``). Other ``ner_*`` trees can carry legacy or snapshot metrics that would
    distort shared trade-off / bubble axes when merged.
    """
    chrfs: list[float] = []
    htms: list[float] = []
    times: list[float] = []
    if not scan_root.is_dir():
        return {}
    for rel in ("ner_biollm/figures/scores_summary.csv", "ner_biollm_finetuned/figures/scores_summary.csv"):
        csv_path = scan_root / rel
        if not csv_path.is_file():
            continue
        try:
            with csv_path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    c = _parse_csv_float(row.get("chrf"))
                    h = _parse_csv_float(row.get("htm"))
                    t = _parse_csv_float(row.get("mean_s"))
                    if c is not None:
                        chrfs.append(c)
                    if h is not None:
                        htms.append(h)
                    if t is not None:
                        times.append(t)
        except OSError:
            continue
    out: dict[str, tuple[float, float]] = {}
    if chrfs:
        pad_x = 0.03 * (max(chrfs) - min(chrfs) + 1e-9)
        out["chrf"] = (min(chrfs) - pad_x, max(chrfs) + pad_x)
    if htms:
        pad_y = 0.03 * (max(htms) - min(htms) + 1e-9)
        out["htm"] = (max(0.0, min(htms) - pad_y), min(1.05, max(htms) + pad_y))
    if times:
        pad_t = 0.04 * (max(times) - min(times) + 1e-9)
        out["time"] = (max(0.0, min(times) - pad_t), max(times) + pad_t)
    return out


def _tighten_axis_to_points(
    vals: np.ndarray,
    lim: tuple[float, float],
    *,
    hard_lo: float,
    hard_hi: float,
    min_span_ratio: float = 2.35,
) -> tuple[float, float]:
    """If axis span is much wider than the point spread, shrink limits (fairer scatter view)."""
    v0, v1 = float(np.min(vals)), float(np.max(vals))
    span_pt = v1 - v0
    lo, hi = float(lim[0]), float(lim[1])
    span_ax = hi - lo
    if span_pt <= 1e-12 or span_ax <= span_pt * min_span_ratio:
        return (lo, hi)
    pad = max(span_ax * 0.02, span_pt * 0.14, 0.01)
    n_lo = max(hard_lo, v0 - pad)
    n_hi = min(hard_hi, v1 + pad)
    if n_hi - n_lo < span_pt * 1.2:
        cx = (v0 + v1) / 2.0
        half = max(span_pt * 0.75, (n_hi - n_lo) / 2.0)
        n_lo = max(hard_lo, cx - half)
        n_hi = min(hard_hi, cx + half)
    return (n_lo, n_hi)


def _merge_limits_with_rows(
    glob_lim: dict[str, tuple[float, float]],
    rows: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Expand global scan bounds so the current ``rows`` are always inside."""
    merged = dict(glob_lim)
    chrfs = [float(r["chrf"]) for r in rows if r.get("chrf") is not None]
    htms = [
        float(r["htm"])
        for r in rows
        if r.get("htm") is not None and not (isinstance(r["htm"], float) and math.isnan(r["htm"]))
    ]
    ts = [float(r["mean_s"]) for r in rows if r.get("mean_s") is not None]
    if chrfs:
        lo, hi = merged.get("chrf", (min(chrfs), max(chrfs)))
        pad = 0.03 * (max(chrfs + [hi]) - min(chrfs + [lo]) + 1e-9)
        merged["chrf"] = (min(lo, min(chrfs)) - pad, max(hi, max(chrfs)) + pad)
    if htms:
        lo, hi = merged.get("htm", (min(htms), max(htms)))
        pad = 0.03 * (max(htms + [hi]) - min(htms + [lo]) + 1e-9)
        merged["htm"] = (
            max(0.0, min(lo, min(htms)) - pad),
            min(1.05, max(hi, max(htms)) + pad),
        )
    if ts:
        lo, hi = merged.get("time", (min(ts), max(ts)))
        pad = 0.04 * (max(ts + [hi]) - min(ts + [lo]) + 1e-9)
        merged["time"] = (max(0.0, min(lo, min(ts)) - pad), max(hi, max(ts)) + pad)
    return merged


def _draw_y_axis_break_warning(ax: plt.Axes) -> None:
    """Double-slash on the y-axis when the lower limit is above zero (truncated scale)."""
    y0, y1 = ax.get_ylim()
    if y0 <= 0.02:
        return
    xlim = ax.get_xlim()
    sx = xlim[1] - xlim[0]
    sy = y1 - y0
    dx = 0.012 * sx
    dy = 0.018 * sy
    x0 = xlim[0] - 0.008 * sx
    # Two parallel slashes at the bottom of the axis
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


# ---------------------------------------------------------------------------
# Display names (short labels reduce axis clutter)
# ---------------------------------------------------------------------------

# Left-aligned labels for horizontal charts (ASCII only — avoids clipping odd glyphs).
# Display strings shared with ``pipeline.metrics.eval_table`` (imported as ``_label``).
def _scatter_legend_label(row: dict[str, Any]) -> str:
    """Short legend text: S1 … SN, or S{k} Mistral for *_mistral labels."""
    lab = str(row.get("label") or "").strip()
    if not lab:
        return str(row.get("display") or "?")
    low = lab.lower()
    m_mistral = re.fullmatch(r"s(\d+)_mistral", low)
    if m_mistral:
        return f"S{int(m_mistral.group(1))} Mistral"
    m_sn = re.fullmatch(r"s(\d+)", low)
    if m_sn:
        return f"S{int(m_sn.group(1))}"
    return str(row.get("display") or lab)


def _distinct_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 4))
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, n))
    if n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, n))
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    return np.array([cols[i % 20] for i in range(n)], dtype=float)


def _scatter_figure_with_legend_column(
    *,
    figsize: tuple[float, float],
    width_ratios: tuple[float, float] = (4.35, 1.42),
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.set_axis_off()
    return fig, ax, ax_leg


def _draw_right_legend(ax_leg: plt.Axes, rows: list[dict[str, Any]], colours: np.ndarray) -> None:
    handles = [
        Patch(
            facecolor=colours[i],
            edgecolor="#2d2d2d",
            linewidth=0.85,
            label=_scatter_legend_label(rows[i]),
        )
        for i in range(len(rows))
    ]
    ax_leg.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper left",
        frameon=False,
        fontsize=9,
        borderaxespad=0,
        handlelength=1.2,
        handleheight=0.95,
        labelspacing=0.55,
    )


def _height_for_labels(n: int, base: float, per_row: float) -> float:
    """Ensure enough vertical space for tick labels."""
    return base + per_row * max(n, 1)


def _metric_for_plot(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) or math.isinf(v) else v


def plot_grouped(rows: list[dict[str, Any]], out_base: Path, fmt: str, dpi: int) -> None:
    if not rows:
        return
    labels = [r["display"] for r in rows]
    n = len(labels)
    y = np.arange(n, dtype=float)
    # Paired horizontal bars per system (labels read cleanly on the y-axis).
    gap = 0.22
    bar_h = 0.36
    bleu = [r["bleu"] for r in rows]
    chrf = [r["chrf"] for r in rows]
    htm = [_metric_for_plot(r.get("htm")) for r in rows]

    hfig = _height_for_labels(n, base=5.2, per_row=0.62)

    def _style_y_axis(ax) -> None:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.35)
        ax.set_axisbelow(True)

    if _htm_unavailable(rows):
        fig, ax0 = plt.subplots(
            1,
            1,
            figsize=(11.2, hfig + 0.35),
            gridspec_kw={"left": 0.30, "right": 0.78, "top": 0.92, "bottom": 0.14},
        )

        ax0.barh(y - gap, bleu, height=bar_h, label="BLEU", color="#0173B2")
        ax0.barh(y + gap, chrf, height=bar_h, label="chrF++", color="#DE8F05")
        _style_y_axis(ax0)
        ax0.set_xlabel("Score")
        ax0.set_title("Overlap with English reference (BLEU · chrF++; HTM needs Neo4j)", fontsize=11)
        ax0.set_xlim(left=0)

        handles_all = [
            Patch(facecolor="#0173B2", edgecolor="#cccccc", linewidth=0.6, label="BLEU"),
            Patch(facecolor="#DE8F05", edgecolor="#cccccc", linewidth=0.6, label="chrF++"),
        ]
    else:
        hfig2 = _height_for_labels(n, base=6.1, per_row=0.56)
        fig, (ax0, ax_htm) = plt.subplots(
            2,
            1,
            figsize=(11.2, hfig2 + 0.85),
            gridspec_kw={
                "hspace": 0.44,
                "height_ratios": [1.12, 0.95],
                "left": 0.30,
                "right": 0.78,
                "top": 0.94,
                "bottom": 0.13,
            },
        )

        ax0.barh(y - gap, bleu, height=bar_h, label="BLEU", color="#0173B2")
        ax0.barh(y + gap, chrf, height=bar_h, label="chrF++", color="#DE8F05")
        ax0.set_title("Overlap with English reference (BLEU · chrF++)", fontsize=11)
        _style_y_axis(ax0)
        ax0.set_xlabel("Score")
        ax0.set_xlim(left=0)

        ax_htm.barh(y, htm, height=0.5, label="HTM", color="#029E73")
        htm_max = max(htm) if htm else 0.0
        span = max(htm) - min(htm) if htm else 0.0
        pad = max(1e-6, span * 0.35, htm_max * 0.12)
        x_hi = min(1.0, max(0.055, htm_max + pad))
        ax_htm.set_xlim(0, x_hi)
        ax_htm.set_title(
            "HTM — NER-anchored hierarchy check on English hyp (graph; panel zooms when values are small)",
            fontsize=11,
        )
        _style_y_axis(ax_htm)
        ax_htm.set_xlabel("Score (theoretical range 0–1; xmax may be < 1 for visibility)")

        handles_all = [
            Patch(facecolor="#0173B2", edgecolor="#cccccc", linewidth=0.6, label="BLEU"),
            Patch(facecolor="#DE8F05", edgecolor="#cccccc", linewidth=0.6, label="chrF++"),
            Patch(facecolor="#029E73", edgecolor="#cccccc", linewidth=0.6, label="HTM"),
        ]

    fig.legend(
        handles_all,
        [h.get_label() for h in handles_all],
        loc="upper center",
        ncol=min(3, len(handles_all)),
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.02),
        framealpha=0.95,
    )

    for ext in fmt.split(","):
        fig.savefig(
            f"{out_base}_metrics_grouped.{ext.strip()}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.35,
        )
    plt.close(fig)


def _htm_vec_col_title(key: str) -> str:
    if not key.startswith("htm_vector_"):
        return key
    suf = key.removeprefix("htm_vector_")
    try:
        return f"HTMv{int(suf) / 100:.2f}"
    except ValueError:
        return key


def plot_heatmap(rows: list[dict[str, Any]], ccr: float, out_base: Path, fmt: str, dpi: int) -> None:
    if not rows:
        return
    if _htm_unavailable(rows):
        cols = ["BLEU", "chrF"]
        M = np.array([[r["bleu"], r["chrf"]] for r in rows], dtype=float)
    else:
        vec_keys = sorted(k for k in rows[0] if str(k).startswith("htm_vector_")) if rows else []
        cols = ["BLEU", "chrF", "HTM (lex)"] + [_htm_vec_col_title(k) for k in vec_keys]
        M = np.array(
            [
                [r["bleu"], r["chrf"], _metric_for_plot(r.get("htm"))]
                + [_metric_for_plot(r.get(k)) for k in vec_keys]
                for r in rows
            ],
            dtype=float,
        )
    col_min = M.min(axis=0)
    col_max = M.max(axis=0)
    denom = np.where(col_max > col_min, col_max - col_min, 1.0)
    Mn = (M - col_min) / denom

    nrows = len(rows)
    hfig = _height_for_labels(nrows, base=4.2, per_row=0.52)
    wfig = max(8.5, 1.35 * len(cols) + 5.0)

    fig = plt.figure(figsize=(wfig, hfig + 1.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.14], hspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    foot = fig.add_subplot(gs[1, 0])
    foot.axis("off")

    im = ax.imshow(Mn, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([r["display"] for r in rows], fontsize=10)
    ax.set_title(
        "Relative standing per metric (column min–max normalisation)",
        fontsize=11,
        pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalised rank within column", fontsize=10)

    ccr_note = (
        f"CCR = {ccr:.3f}  ·  dataset grounding coverage only (not a per-system score)"
        if not (isinstance(ccr, float) and math.isnan(ccr))
        else "CCR not computed (--no-graph or Neo4j unavailable)"
    )
    rhtm = rows[0].get("htm_en_ref_dataset") if rows else None
    if rhtm is not None and not (isinstance(rhtm, float) and math.isnan(float(rhtm))):
        ccr_note += (
            f"  ·  rHTM = {float(rhtm):.3f} (same HTM-style check on gold en_ref; dataset-level)"
        )
    foot.text(
        0.5,
        0.5,
        ccr_note,
        ha="center",
        va="center",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f4f4f4",
            "edgecolor": "#c8c8c8",
            "linewidth": 0.9,
        },
    )

    for ext in fmt.split(","):
        fig.savefig(f"{out_base}_heatmap.{ext.strip()}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(
    rows: list[dict[str, Any]],
    out_base: Path,
    fmt: str,
    dpi: int,
    *,
    axis_limits: dict[str, tuple[float, float]] | None = None,
) -> None:
    if len(rows) < 2:
        return
    if any(isinstance(r.get("htm"), float) and math.isnan(r["htm"]) for r in rows):
        return

    n = len(rows)
    colours = _distinct_colors(n)

    fig, ax, ax_leg = _scatter_figure_with_legend_column(figsize=(9.35, 6.35))

    xs = np.array([float(r["chrf"]) for r in rows], dtype=float)
    ys = np.array([float(r["htm"]) for r in rows], dtype=float)

    if axis_limits and "chrf" in axis_limits and "htm" in axis_limits:
        xlim = axis_limits["chrf"]
        ylim = axis_limits["htm"]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.margins(0.10)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    xlim = _tighten_axis_to_points(xs, xlim, hard_lo=0.0, hard_hi=200.0)
    ylim = _tighten_axis_to_points(ys, ylim, hard_lo=0.0, hard_hi=1.05)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    mx = (xlim[0] + xlim[1]) / 2
    my = (ylim[0] + ylim[1]) / 2
    ax.axhspan(my, ylim[1], facecolor="#e8f5e9", alpha=0.28, zorder=0)

    draw_order = np.argsort(xs + ys)[::-1]
    for ii in draw_order:
        r = rows[int(ii)]
        i = int(ii)
        c = colours[i]
        xi, yi = float(r["chrf"]), float(r["htm"])
        ax.scatter(
            xi,
            yi,
            s=220,
            c=[c],
            alpha=0.65,
            edgecolors="#2d2d2d",
            linewidths=1.0,
            zorder=3,
        )

    ax.set_xlabel("chrF++ (fluency; overlap with reference)", fontsize=11)
    ax.set_ylabel("HTM (terminology match with graph-aware hierarchy)", fontsize=11)
    ax.set_title("Fluency vs terminology (prefer top-right)", fontsize=11.5, pad=14)
    ax.grid(alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))

    _draw_y_axis_break_warning(ax)
    _draw_right_legend(ax_leg, rows, colours)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.13)
    fig.text(
        0.5,
        0.02,
        "HTM is defined on [0, 1]; axes shrink when padding is much wider than the plotted points.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )

    for ext in fmt.split(","):
        fig.savefig(
            f"{out_base}_tradeoff_chrF_vs_HTM.{ext.strip()}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.32,
        )
    plt.close(fig)


def plot_bubble_chrF_htm_time(
    rows: list[dict[str, Any]],
    out_base: Path,
    fmt: str,
    dpi: int,
    *,
    axis_limits: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Fluency (x) vs terminology (y); bubble area scales with mean wall time (prefer top-right, small bubbles)."""
    rows_b = [r for r in rows if r.get("mean_s") is not None]
    if len(rows_b) < 2:
        return
    if any(isinstance(r.get("htm"), float) and math.isnan(r["htm"]) for r in rows_b):
        return

    n = len(rows_b)
    colours = _distinct_colors(n)
    times = np.array([float(r["mean_s"]) for r in rows_b], dtype=float)

    if axis_limits and "time" in axis_limits:
        t_lo_g, t_hi_g = axis_limits["time"]
    else:
        t_lo_g, t_hi_g = float(times.min()), float(times.max())
    span_t = max(t_hi_g - t_lo_g, 1e-9)
    norm_t = np.clip((times - t_lo_g) / span_t, 0.0, 1.0)

    s_min, s_max = 140.0, 2400.0
    areas = s_min + norm_t * (s_max - s_min)

    xs = np.array([float(r["chrf"]) for r in rows_b], dtype=float)
    ys = np.array([float(r["htm"]) for r in rows_b], dtype=float)

    fig, ax, ax_leg = _scatter_figure_with_legend_column(figsize=(9.35, 6.95))

    if axis_limits and "chrf" in axis_limits and "htm" in axis_limits:
        xlim = axis_limits["chrf"]
        ylim = axis_limits["htm"]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.margins(0.10)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    xlim = _tighten_axis_to_points(xs, xlim, hard_lo=0.0, hard_hi=200.0)
    ylim = _tighten_axis_to_points(ys, ylim, hard_lo=0.0, hard_hi=1.05)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    mx = (xlim[0] + xlim[1]) / 2
    my = (ylim[0] + ylim[1]) / 2

    ax.axvspan(mx, xlim[1], facecolor="#e8f5e9", alpha=0.22, zorder=0)
    ax.axhspan(my, ylim[1], facecolor="#e8f5e9", alpha=0.22, zorder=0)

    for idx in np.argsort(-areas):
        i = int(idx)
        r = rows_b[i]
        c = colours[i]
        xi, yi = float(r["chrf"]), float(r["htm"])
        ax.scatter(
            xi,
            yi,
            s=areas[i],
            c=[c],
            alpha=0.65,
            edgecolors="#2d2d2d",
            linewidths=1.0,
            zorder=3,
        )

    ax.set_xlabel("chrF++ (fluency)", fontsize=11)
    ax.set_ylabel("HTM (terminology)", fontsize=11)
    ax.set_title(
        "Fluency vs terminology (bubble size ∝ mean seconds per segment)",
        fontsize=11.5,
        pad=14,
    )
    ax.grid(alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))

    _draw_y_axis_break_warning(ax)
    _draw_right_legend(ax_leg, rows_b, colours)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.11)

    for ext in fmt.split(","):
        fig.savefig(
            f"{out_base}_bubble_chrF_HTM_time.{ext.strip()}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.32,
        )
    plt.close(fig)


def plot_timing(rows: list[dict[str, Any]], out_base: Path, fmt: str, dpi: int) -> None:
    rows_t = [r for r in rows if r.get("mean_s") is not None]
    if not rows_t:
        return
    labels = [r["display"] for r in rows_t]
    means = [float(r["mean_s"]) for r in rows_t]
    p95s = [float(r["p95_s"]) if r["p95_s"] is not None else float(r["mean_s"]) for r in rows_t]
    err = [max(0.0, p - m) for p, m in zip(p95s, means)]

    n = len(rows_t)
    hfig = _height_for_labels(n, base=4.5, per_row=0.48)
    fig, ax = plt.subplots(figsize=(10.0, hfig), constrained_layout=True)
    y = np.arange(n)
    # Upper tail to p95 only (symmetric xerr would extend left of the mean and hit negative x).
    xerr = np.vstack([np.zeros(n, dtype=float), np.asarray(err, dtype=float)])
    ax.barh(y, means, xerr=xerr, capsize=4, color="#787878", ecolor="#2a2a2a", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per segment (mean; error bar = upper tail to p95)", fontsize=11)
    ax.set_title(
        "Inference wall time\n(compare configurations on one machine; not cross-hardware benchmarks)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    for ext in fmt.split(","):
        fig.savefig(f"{out_base}_inference_time.{ext.strip()}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_paper_summary_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Markdown table: fluency, terminology, latency — suitable for paper text."""
    vec_keys = sorted(k for k in rows[0] if str(k).startswith("htm_vector_")) if rows else []
    header_cells = ["System", "chrF++", "HTM (lex)"] + [_htm_vec_col_title(k) for k in vec_keys]
    header_cells += ["BLEU", "doc-BLEU", "doc-chrF", "BLEU†", "Mean s/seg", "p95 s"]
    sep = "| " + " | ".join(["---"] * len(header_cells)) + " |"
    lines = [
        "# Summary metrics",
        "",
        "| " + " | ".join(header_cells) + " |",
        sep,
    ]

    def _fmt_doc_m(v: object) -> str:
        if v is None:
            return "—"
        fv = float(v)
        return "—" if math.isnan(fv) else f"{fv:.2f}"

    for r in rows:
        ms = r.get("mean_s")
        p95 = r.get("p95_s")
        ms_s = f"{float(ms):.2f}" if ms is not None else "—"
        p95_s = f"{float(p95):.2f}" if p95 is not None else "—"
        h = r.get("htm")
        h_s = "—" if (isinstance(h, float) and math.isnan(h)) else f"{float(h):.3f}"
        cells = [r["display"], f"{float(r['chrf']):.2f}", h_s]
        for k in vec_keys:
            hv = r.get(k)
            if isinstance(hv, float) and math.isnan(hv):
                cells.append("—")
            elif hv is None:
                cells.append("—")
            else:
                cells.append(f"{float(hv):.3f}")
        cells += [
            f"{float(r['bleu']):.2f}",
            _fmt_doc_m(r.get("bleu_doc_macro")),
            _fmt_doc_m(r.get("chrf_doc_macro")),
            _fmt_doc_m(r.get("bleu_doc_concat")),
            ms_s,
            p95_s,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "*BLEU† — macro mean over documents of corpus BLEU on a single synthetic line per document "
        "(segment `hyp` / `en_ref` joined with spaces; column `bleu_doc_concat` in CSV).*"
    )
    rh = rows[0].get("htm_en_ref_dataset") if rows else None
    if rh is not None and not (isinstance(rh, float) and math.isnan(float(rh))):
        lines.append("")
        lines.append(
            f"*rHTM (dataset, gold `en_ref` vs grounded MedDRA English): {float(rh):.3f}*"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_comet_optional(rows: list[dict[str, Any]], out_base: Path, fmt: str, dpi: int) -> None:
    vals = [r["comet"] for r in rows if r.get("comet") is not None]
    if not vals:
        return
    sub = [r for r in rows if r.get("comet") is not None]
    n = len(sub)
    hfig = _height_for_labels(n, base=4.0, per_row=0.48)
    fig, ax = plt.subplots(figsize=(10.0, hfig), constrained_layout=True)
    y = np.arange(n)
    ax.barh(y, [float(r["comet"]) for r in sub], color="#56B4E9", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r["display"] for r in sub], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("COMET DA (Unbabel/wmt22-comet-da)", fontsize=11)
    ax.set_title("Neural reference-based metric (optional; requires unbabel-comet)", fontsize=11)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    for ext in fmt.split(","):
        fig.savefig(f"{out_base}_comet.{ext.strip()}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot system comparison figures from results JSONL files.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory with s1.jsonl … (default: results/ad_hoc/). Figures default to {results-dir}/figures/.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: {--results-dir}/figures).",
    )
    p.add_argument("--partial", action="store_true", help="Skip malformed JSONL lines.")
    p.add_argument("--comet", action="store_true", help="Compute COMET (slow; needs unbabel-comet).")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--format", default="png", help="Comma-separated: png, pdf, svg (default: png only)")
    p.add_argument(
        "--no-inference-figure",
        action="store_true",
        help="Skip comparison_inference_time.* (use paper_summary_table.md for latency instead).",
    )
    p.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="JSONL for CCR (default: data/section48/segments_ner.jsonl). Use segments_ner_biollm.jsonl to match a BioMistral NER run.",
    )
    p.add_argument(
        "--grounding-mode",
        choices=["string", "vector", "vector_llm"],
        default="string",
        help="Neo4j grounding for CCR / HTM in summary tables (default: string).",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="Do not connect to Neo4j: skip CCR, HTM, rHTM, and HTM-based figures (BLEU/chrF/COMET kept).",
    )
    p.add_argument(
        "--scatter-scan-root",
        type=Path,
        default=None,
        help="Directory to scan for ner_*/figures/scores_summary.csv when locking trade-off/bubble axes "
        "(default: project results/). Ignored with --no-lock-scatter-axes.",
    )
    p.add_argument(
        "--no-lock-scatter-axes",
        action="store_true",
        help="Let chrF/HTM/time axis limits auto-fit each chart instead of global scan + merge.",
    )
    p.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids omitted from plots/metrics (same as evaluate.py / pipeline). Example: 48_028",
    )
    p.add_argument(
        "--htm-vector-thresholds",
        type=str,
        default="",
        help="Comma-separated cosine thresholds in [0,1] for extra vector HTM columns (e.g. 0.8,0.9). Slow.",
    )
    p.add_argument(
        "--htm-embed-model",
        type=str,
        default=None,
        help="sentence-transformers model id for vector HTM (default: TERMPLAN_EMBED_MODEL env).",
    )
    args = p.parse_args()

    try:
        htm_vec_thr = parse_cosine_thresholds_csv(args.htm_vector_thresholds or "")
    except ValueError as e:
        raise SystemExit(str(e)) from None

    def _under_root(p: Path | None, default: Path) -> Path:
        x = p if p is not None else default
        return x if x.is_absolute() else (ROOT / x)

    seg_path = _under_root(args.segments, ROOT / "data" / "section48" / "segments_ner.jsonl")
    if not seg_path.is_file():
        raise SystemExit(f"Segments file not found: {seg_path}")
    results_dir = _under_root(args.results_dir, ROOT / "results" / "ad_hoc")
    out_dir = _under_root(args.out_dir, results_dir / "figures")

    id_to_ref: dict[str, str] = {}
    id_to_src: dict[str, str] = {}
    exclude_seg = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude_seg)
    keep_ids = frozenset(row["id"] for row in segment_rows)
    for row in segment_rows:
        id_to_ref[row["id"]] = row["en_ref"]
        id_to_src[row["id"]] = row["fr"]
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )

    if args.no_graph:
        print(
            "[no-graph] Skipping Neo4j — CCR, HTM, rHTM, and HTM-based scatter/bubble figures omitted.\n",
            file=sys.stderr,
        )

    rows: list[dict[str, Any]] = []
    ccr = float("nan")
    graph: TermGraph | None = None
    try:
        if not args.no_graph:
            graph = TermGraph(grounding_mode=args.grounding_mode)
        try:
            rows, ccr = collect_system_metric_rows(
                results_dir=results_dir,
                id_to_ref=id_to_ref,
                id_to_src=id_to_src,
                graph=graph,
                segment_rows=segment_rows,
                partial=args.partial,
                with_comet=args.comet,
                keep_segment_ids=keep_ids,
                htm_vector_thresholds=htm_vec_thr if htm_vec_thr else None,
                htm_embed_model=args.htm_embed_model or None,
            )
        except NEO4J_CONN_ERRORS:
            raise SystemExit(neo4j_connection_help()) from None
    finally:
        if graph is not None:
            graph.close()

    if not rows:
        raise SystemExit("No result files found — run tools/pipeline/run_pipeline.py first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "comparison"
    write_scores_summary_csv(rows, ccr, out_dir / "scores_summary.csv")
    write_paper_summary_table(rows, out_dir / "paper_summary_table.md")

    scan_root = _under_root(args.scatter_scan_root, ROOT / "results")
    if args.no_lock_scatter_axes:
        scatter_limits: dict[str, tuple[float, float]] | None = None
    else:
        g = scan_global_scatter_limits(scan_root)
        scatter_limits = _merge_limits_with_rows(g, rows)

    plot_grouped(rows, base, args.format, args.dpi)
    plot_heatmap(rows, ccr, base, args.format, args.dpi)
    plot_tradeoff(rows, base, args.format, args.dpi, axis_limits=scatter_limits)
    plot_bubble_chrF_htm_time(rows, base, args.format, args.dpi, axis_limits=scatter_limits)
    if not args.no_inference_figure:
        plot_timing(rows, base, args.format, args.dpi)
    plot_comet_optional(rows, base, args.format, args.dpi)

    print(f"Wrote figures, {out_dir / 'paper_summary_table.md'}, and {out_dir / 'scores_summary.csv'}")
    if isinstance(ccr, float) and math.isnan(ccr):
        print("CCR (dataset): not computed (--no-graph).")
    else:
        rh = rows[0].get("htm_en_ref_dataset") if rows else None
        rh_s = ""
        if rh is not None and not (isinstance(rh, float) and math.isnan(float(rh))):
            rh_s = f"  rHTM (en_ref vs MedDRA renderings): {float(rh):.4f}."
        print(
            f"CCR (dataset): {ccr:.4f} — also in heatmap footnote; not a per-system bar.{rh_s}"
        )


if __name__ == "__main__":
    main()
