#!/usr/bin/env python3
"""Compute string HTM vs vector HTM at configurable cosine thresholds for every MT system.

Scans ``results/`` for NER directories listed in ``pipeline.metrics.eval_manifest.EVAL_RERUN_PROFILES``
that contain pipeline JSONLs (``EVAL_FILES``), evaluates each system file against the **same**
segment JSONL as that pipeline (NER ``terms[]`` → graph grounding → English ``hyp``), and writes:

  - ``htm_threshold_comparison.csv`` — one row per (NER condition, system)
  - ``htm_threshold_comparison.json`` — nested summary
  - ``htm_threshold_comparison.png`` (optional ``.pdf`` via ``--format``) — **overview**: one panel per MT system; **colour = NER pipeline**, **hatch = HTM variant** (string vs vector thresholds); no numeric bar labels
  - ``htm_threshold_comparison__<metric>.png`` — **per-metric** figures (larger bars, solid fill only) for slides or the appendix
  - ``htm_threshold_comparison__string_vs_vector_panels.png`` — **grid**: one subplot per NER pipeline; at each MT system, **substring and vector thresholds as adjacent bars** (easier to read than hatch-only overview)

Requires Neo4j (same as ``evaluate.py`` HTM) **and** ``sentence-transformers`` for vector HTM.

Example::

    PYTHONPATH=. python tools/eval/compare_htm_vector_thresholds.py \\
      --results-root results --thresholds 0.8,0.9 --out-dir results/htm_vector_comparison \\
      --exclude-segment-ids 48_028
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.eval_io import load_results_jsonl as _load_jsonl
from pipeline.metrics.eval_manifest import (
    EVAL_FILES,
    EVAL_RERUN_PROFILES,
    condition_name_from_results_subdir,
)
from pipeline.metrics.htm import (
    compute_htm,
    compute_htm_vector,
    htm_vector_column_key,
    parse_cosine_thresholds_csv,
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


DISPLAY_NAMES: dict[str, str] = {
    "s1": "S1 NLLB",
    "s2": "S2 Mistral (doc)",
    "s3": "S3 GraphRAG",
    "s4": "S4 rerank",
    "s5": "S5 NLLB + boost",
    "s5_mistral": "S5 Mistral + boost",
}

# Preferred NER pipeline order for grouped bars (see ``EVAL_RERUN_PROFILES``).
PIPELINE_ORDER: list[str] = [
    condition_name_from_results_subdir(s) for s, _rels in EVAL_RERUN_PROFILES
]

_PIPELINE_LEGEND_BASE: dict[str, str] = {
    "ner_biollm": "BioLLM",
    "ner_biollm_finetuned": "BioLLM fine-tuned",
    "ner_baseline": "CamemBERT baseline",
    "ner_finetuned": "CamemBERT fine-tuned",
}

PIPELINE_LEGEND: dict[str, str] = {
    c: _PIPELINE_LEGEND_BASE.get(c, c.replace("_", " ")) for c in PIPELINE_ORDER
}

PIPELINE_COLORS: list[str] = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#56B4E9", "#D55E00"]

SEGMENTS_REL_FOR_CONDITION: dict[str, tuple[str, ...]] = {
    condition_name_from_results_subdir(sub): rels for sub, rels in EVAL_RERUN_PROFILES
}


def _segments_path_for_condition(cond_name: str) -> Path:
    rels = SEGMENTS_REL_FOR_CONDITION.get(cond_name)
    if not rels:
        raise SystemExit(
            f"No segment JSONL mapping for results folder {cond_name!r}. "
            f"Add it to SEGMENTS_REL_FOR_CONDITION in tools/eval/compare_htm_vector_thresholds.py."
        )
    for rel in rels:
        p = ROOT / rel
        if p.is_file():
            return p
    tried = ", ".join(rels)
    raise SystemExit(f"No segment file found for {cond_name!r} (tried: {tried})")


def _discover_ner_dirs(results_root: Path) -> list[Path]:
    """Only paper-facing NER conditions (see ``PIPELINE_ORDER``), not every ``ner_*`` folder."""
    out: list[Path] = []
    for name in PIPELINE_ORDER:
        p = results_root / name
        if not p.is_dir():
            continue
        if any((p / fn).is_file() for _, fn in EVAL_FILES):
            out.append(p)
    return out


def collect_metrics_for_dir(
    results_dir: Path,
    id_to_segment: dict[str, dict[str, Any]],
    keep_segment_ids: frozenset[str],
    graph: TermGraph,
    thresholds: list[float],
    *,
    partial: bool,
    embed_model_name: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, fname in EVAL_FILES:
        path = results_dir / fname
        if not path.is_file():
            continue
        res = _load_jsonl(path, partial=partial)
        res = [r for r in res if str(r.get("id")) in keep_segment_ids]
        if not res:
            continue
        htm_s = compute_htm(res, graph, id_to_segment)
        row: dict[str, Any] = {
            "label": label,
            "display": DISPLAY_NAMES.get(label, label),
            "htm_string": round(float(htm_s), 6),
        }
        for t in thresholds:
            hv = compute_htm_vector(
                res,
                graph,
                id_to_segment,
                similarity_threshold=t,
                embed_model_name=embed_model_name,
            )
            row[htm_vector_column_key(t)] = round(float(hv), 6)
        rows.append(row)
    return rows


def _pipeline_conditions(by_condition: dict[str, list[dict[str, Any]]]) -> list[str]:
    preferred = [c for c in PIPELINE_ORDER if c in by_condition]
    rest = sorted(c for c in by_condition if c not in preferred)
    return preferred + rest


def _system_labels_in_eval_order(by_condition: dict[str, list[dict[str, Any]]]) -> list[str]:
    present: set[str] = set()
    for rows in by_condition.values():
        present.update(r["label"] for r in rows)
    return [lbl for lbl, _ in EVAL_FILES if lbl in present]


def _metric_file_slug(key: str) -> str:
    """Stable filename fragment, e.g. ``htm_vector_080``."""
    return key if key else "metric"


def _plot_single_metric_panel(
    ax: plt.Axes,
    *,
    metric_key: str,
    metric_title: str,
    by_label: dict[str, dict[str, dict[str, Any]]],
    conditions: list[str],
    system_labels: list[str],
    panel_title_fs: float,
    label_fs: float,
    show_xlabels: bool,
) -> float:
    """Draw grouped bars (pipelines) for one HTM column; return global max value for this metric."""
    n_sys = len(system_labels)
    n_pipe = len(conditions)
    x = np.arange(n_sys, dtype=float)
    W = 0.78
    slot = W / n_pipe
    bar_w = slot * 0.88

    max_val = 0.0

    for pi, cond in enumerate(conditions):
        yvals: list[float] = []
        for lbl in system_labels:
            r = by_label[cond].get(lbl)
            v = float(r.get(metric_key, float("nan"))) if r else float("nan")
            if not np.isfinite(v):
                v = 0.0
            yvals.append(v)
            max_val = max(max_val, v)
        left0 = x - W / 2.0
        x_bar = left0 + pi * slot + (slot - bar_w) / 2.0
        color = PIPELINE_COLORS[pi % len(PIPELINE_COLORS)]
        ax.bar(
            x_bar,
            yvals,
            width=bar_w,
            color=color,
            edgecolor="0.22",
            linewidth=0.65,
            zorder=3,
        )

    y_hi = min(0.5, max(0.08, max_val * 1.18))
    ax.set_ylim(0.0, y_hi)
    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(
            [DISPLAY_NAMES.get(lbl, lbl) for lbl in system_labels],
            rotation=24,
            ha="right",
            fontsize=max(8.5, label_fs - 1.0),
        )
        ax.set_xlabel("MT system", fontsize=label_fs)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("HTM", fontsize=label_fs)
    ax.set_title(metric_title, fontsize=panel_title_fs, fontweight="semibold", pad=14)
    ax.grid(axis="y", alpha=0.38, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    return max_val


def _plot_overview_combined_bars(
    ax: plt.Axes,
    *,
    by_label: dict[str, dict[str, dict[str, Any]]],
    conditions: list[str],
    system_labels: list[str],
    metric_keys: list[str],
) -> float:
    """One cluster per MT system: bars are pipeline × metric (hatch distinguishes metric)."""
    n_sys = len(system_labels)
    n_pipe = len(conditions)
    n_met = len(metric_keys)
    styles = ["", "//", "xx", "..", "||"]
    hatch_per_met = [styles[i % len(styles)] for i in range(n_met)]
    n_bar = n_pipe * n_met
    cluster_w = 0.82
    bar_w = cluster_w / max(n_bar, 1) * 0.91
    x_centers = np.arange(n_sys, dtype=float)
    max_val = 0.0
    for si, lbl in enumerate(system_labels):
        k = 0
        for pi, cond in enumerate(conditions):
            r = by_label[cond].get(lbl)
            for mi, mk in enumerate(metric_keys):
                v = 0.0
                if r:
                    vv = r.get(mk)
                    if vv is not None:
                        try:
                            fv = float(vv)
                            if np.isfinite(fv):
                                v = fv
                        except (TypeError, ValueError):
                            pass
                max_val = max(max_val, v)
                offset = (k - (n_bar - 1) / 2.0) * bar_w * 1.06
                ax.bar(
                    x_centers[si] + offset,
                    v,
                    width=bar_w,
                    color=PIPELINE_COLORS[pi % len(PIPELINE_COLORS)],
                    edgecolor="0.18",
                    linewidth=0.55,
                    hatch=hatch_per_met[mi],
                    zorder=3,
                )
                k += 1
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(lbl, lbl) for lbl in system_labels],
        rotation=24,
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel("MT system", fontsize=10.5)
    ax.set_ylabel("HTM", fontsize=10.5)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    y_hi = min(0.5, max(0.08, max_val * 1.22))
    ax.set_ylim(0.0, y_hi)
    return max_val


def _metric_colors_and_labels(thresholds: list[float]) -> tuple[list[str], list[str], list[str]]:
    metric_keys = ["htm_string"] + [htm_vector_column_key(t) for t in thresholds]
    metric_labels = ["Substring match"] + [f"Vector cosine ≥ {t:g}" for t in thresholds]
    # Distinct, colour-blind-friendly hues (string = blue; vectors = green / amber / mauve…).
    palette = ["#4e79a7", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]
    colors = [palette[i % len(palette)] for i in range(len(metric_keys))]
    return metric_keys, metric_labels, colors


def plot_string_vs_vector_panel_grid(
    by_condition: dict[str, list[dict[str, Any]]],
    thresholds: list[float],
    out_base: Path,
    *,
    dpi: int,
    formats: str,
) -> None:
    """One subplot per NER pipeline: grouped bars per MT system (substring next to each vector threshold)."""
    conditions = _pipeline_conditions(by_condition)
    if not conditions:
        return
    system_labels = _system_labels_in_eval_order(by_condition)
    if not system_labels:
        return

    metric_keys, metric_labels, metric_colors = _metric_colors_and_labels(thresholds)
    n_met = len(metric_keys)
    n_sys = len(system_labels)

    by_label_per_cond = {c: {r["label"]: r for r in by_condition[c]} for c in conditions}

    n_panels = len(conditions)
    ncols = 2 if n_panels > 1 else 1
    nrows = (n_panels + ncols - 1) // ncols
    fig_w = 7.8 * ncols + 1.2
    fig_h = 4.55 * nrows + 1.55

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=[0.12] + [1.0] * nrows,
        hspace=0.42,
        wspace=0.26,
        left=0.07,
        right=0.99,
        top=0.94,
        bottom=0.07,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()

    handles = [
        Patch(facecolor=metric_colors[i], edgecolor="0.2", linewidth=0.65, label=metric_labels[i])
        for i in range(n_met)
    ]
    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=min(4, n_met),
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#aaaaaa",
        title="HTM variant",
    )

    x = np.arange(n_sys, dtype=float)
    spacing = min(0.22, 0.78 / max(n_met * 1.15, 1))
    bar_w = spacing * 0.9
    centers = np.linspace(-(n_met - 1) / 2.0 * spacing, (n_met - 1) / 2.0 * spacing, n_met)

    slot_axes: list = []
    for slot in range(nrows * ncols):
        row = 1 + slot // ncols
        col = slot % ncols
        slot_axes.append(fig.add_subplot(gs[row, col]))

    for pi, cond in enumerate(conditions):
        ax = slot_axes[pi]
        bl = by_label_per_cond[cond]
        max_val = 0.0
        for mi, mk in enumerate(metric_keys):
            yvals: list[float] = []
            for lbl in system_labels:
                r = bl.get(lbl)
                v = float(r.get(mk, float("nan"))) if r else float("nan")
                if not np.isfinite(v):
                    v = 0.0
                yvals.append(v)
                max_val = max(max_val, v)
            ax.bar(
                x + centers[mi],
                yvals,
                width=bar_w,
                color=metric_colors[mi],
                edgecolor="0.18",
                linewidth=0.55,
                zorder=3,
            )

        y_hi = min(0.5, max(0.08, max_val * 1.2))
        ax.set_ylim(0.0, y_hi)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_NAMES.get(lbl, lbl) for lbl in system_labels],
            rotation=22,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel("HTM", fontsize=10.5)
        title = PIPELINE_LEGEND.get(cond, cond.replace("_", " "))
        ax.set_title(title, fontsize=11.5, fontweight="semibold", pad=10)
        ax.grid(axis="y", alpha=0.35, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    for pi in range(len(conditions), len(slot_axes)):
        slot_axes[pi].set_visible(False)

    fig.suptitle(
        "Substring HTM vs vector HTM (adjacent bars per MT system)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    base_path = out_base.parent / (out_base.name + "__string_vs_vector_panels")
    for ext in formats.split(","):
        e = ext.strip()
        if e:
            fig.savefig(f"{base_path}.{e}", dpi=dpi, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def plot_grouped_bars(
    by_condition: dict[str, list[dict[str, Any]]],
    thresholds: list[float],
    out_base: Path,
    *,
    dpi: int,
    formats: str,
) -> None:
    """Write one overview figure (combined bars: pipeline × metric hatch) and per-metric figures."""
    metric_keys = ["htm_string"] + [htm_vector_column_key(t) for t in thresholds]
    metric_labels = ["HTM (substring match)"] + [f"Vector HTM (cosine ≥ {t:g})" for t in thresholds]

    conditions = _pipeline_conditions(by_condition)
    if not conditions:
        return
    by_label: dict[str, dict[str, dict[str, Any]]] = {
        c: {r["label"]: r for r in by_condition[c]} for c in conditions
    }
    system_labels = _system_labels_in_eval_order(by_condition)
    if not system_labels:
        return

    n_sys = len(system_labels)

    # Legend handles (pipelines only — one metric per figure is readable without hatch legend).
    pipe_handles = [
        Patch(
            facecolor=PIPELINE_COLORS[i % len(PIPELINE_COLORS)],
            edgecolor="0.22",
            linewidth=0.65,
            label=PIPELINE_LEGEND.get(cond, cond.replace("_", " ")),
        )
        for i, cond in enumerate(conditions)
    ]

    fig_w = min(15.5, 4.2 + 1.55 * n_sys)

    # --- Per-metric figures (clearest for publication slides) ---
    ncol = min(4, len(pipe_handles))
    for metric_key, metric_title in zip(metric_keys, metric_labels, strict=True):
        fig = plt.figure(figsize=(fig_w, 7.0))
        fig.suptitle(
            "HTM by MT system and NER pipeline",
            fontsize=14,
            fontweight="bold",
            y=0.97,
        )
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[0.26, 1],
            hspace=0.35,
            left=0.10,
            right=0.98,
            top=0.89,
            bottom=0.18,
        )
        ax_leg = fig.add_subplot(gs[0, 0])
        ax_leg.set_axis_off()
        ax_leg.legend(
            handles=pipe_handles,
            loc="center",
            ncol=ncol,
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="0.78",
            columnspacing=1.05,
            handlelength=1.2,
        )
        ax = fig.add_subplot(gs[1, 0])
        _plot_single_metric_panel(
            ax,
            metric_key=metric_key,
            metric_title=metric_title,
            by_label=by_label,
            conditions=conditions,
            system_labels=system_labels,
            panel_title_fs=12.5,
            label_fs=11.5,
            show_xlabels=True,
        )
        slug = _metric_file_slug(metric_key)
        base_detail = f"{out_base.parent / (out_base.name + '__' + slug)}"
        for ext in formats.split(","):
            e = ext.strip()
            if e:
                fig.savefig(f"{base_detail}.{e}", dpi=dpi, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    # --- Overview: single axes (colour = pipeline, hatch = metric); no numeric bar labels ---
    metric_hatch_styles = ["", "//", "xx", "..", "||"]
    metric_handles = [
        Patch(
            facecolor="#d5d5d5",
            edgecolor="#1a1a1a",
            linewidth=0.75,
            hatch="",
            label="HTM (string match)",
        )
    ]
    for mi, (mk, mt) in enumerate(zip(metric_keys[1:], metric_labels[1:], strict=True), start=0):
        h = metric_hatch_styles[(mi + 1) % len(metric_hatch_styles)]
        metric_handles.append(
            Patch(
                facecolor="#d5d5d5",
                edgecolor="#1a1a1a",
                linewidth=0.75,
                hatch=h,
                label=mt.replace("Vector HTM ", "Vec ").replace("cosine ≥", "≥"),
            )
        )

    fig_h = 7.85
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "String vs vector HTM (cosine between hypothesis and English renderings)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.32, 1],
        hspace=0.38,
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.20,
    )
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.set_axis_off()
    ncol_pipe = min(4, len(pipe_handles))
    leg_pipe = ax_leg.legend(
        handles=pipe_handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=ncol_pipe,
        fontsize=8.8,
        title="NER pipeline (colour)",
        frameon=True,
        fancybox=False,
        edgecolor="#999999",
        columnspacing=0.9,
    )
    ax_leg.add_artist(leg_pipe)
    ax_leg.legend(
        handles=metric_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        fontsize=8.2,
        title="Metric variant (hatch)",
        frameon=True,
        fancybox=False,
        edgecolor="#999999",
    )
    ax = fig.add_subplot(gs[1, 0])
    _plot_overview_combined_bars(
        ax,
        by_label=by_label,
        conditions=conditions,
        system_labels=system_labels,
        metric_keys=metric_keys,
    )
    fig.text(
        0.5,
        0.03,
        "Bar heights are HTM (terminology agreement), not dataset CCR. "
        "NER-span grounding by pipeline: results/cross_ner_comparison/cross_ner_ccr_dataset.png.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    for ext in formats.split(","):
        e = ext.strip()
        if e:
            fig.savefig(f"{out_base}.{e}", dpi=dpi, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare string HTM vs vector HTM at cosine thresholds.")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Folder containing ner_* result directories (default: results/).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: {results-root}/htm_vector_comparison).",
    )
    ap.add_argument(
        "--thresholds",
        type=str,
        default="0.8,0.9",
        help="Comma-separated cosine thresholds in [0, 1] (default: 0.8,0.9).",
    )
    ap.add_argument("--partial", action="store_true", help="Skip malformed JSONL lines.")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--format", default="png", help="Comma-separated figure formats (default: png only).")
    ap.add_argument(
        "--embed-model",
        default=None,
        help="sentence-transformers model id (default: TERMPLAN_EMBED_MODEL or paraphrase-multilingual-mpnet-base-v2).",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids to omit (match evaluate.py / pipeline). Example: 48_028",
    )
    ap.add_argument(
        "--grounding-mode",
        choices=["string", "vector", "vector_llm"],
        default="string",
        help="Neo4j grounding for HTM (default: string).",
    )
    args = ap.parse_args()

    thresholds = parse_cosine_thresholds_csv(args.thresholds)
    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = results_root / "htm_vector_comparison"
    elif not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_seg = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    ner_dirs = _discover_ner_dirs(results_root)
    if not ner_dirs:
        raise SystemExit(f"No results/ner_* directories with JSONLs under {results_root}")

    graph: TermGraph | None = None
    by_condition: dict[str, list[dict[str, Any]]] = {}
    flat_csv_rows: list[dict[str, Any]] = []

    try:
        graph = TermGraph(grounding_mode=args.grounding_mode)
        for ner_dir in ner_dirs:
            cond = ner_dir.name
            seg_path = _segments_path_for_condition(cond)
            segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude_seg)
            keep_ids = frozenset(str(r["id"]) for r in segment_rows)
            id_to_segment = {str(r["id"]): r for r in segment_rows}
            try:
                rows = collect_metrics_for_dir(
                    ner_dir,
                    id_to_segment,
                    keep_ids,
                    graph,
                    thresholds,
                    partial=args.partial,
                    embed_model_name=args.embed_model,
                )
            except _NEO4J_CONN_ERRORS as e:
                raise SystemExit(
                    "Neo4j is not reachable — start the DB and check .env (NEO4J_URI / USER / PASS).\n"
                    f"Underlying error: {e}"
                ) from e
            by_condition[cond] = rows
            for r in rows:
                flat_csv_rows.append({"ner_condition": cond, **r})
    finally:
        if graph is not None:
            graph.close()

    fieldnames = ["ner_condition", "label", "display", "htm_string"] + [
        htm_vector_column_key(t) for t in thresholds
    ]
    csv_path = out_dir / "htm_threshold_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in flat_csv_rows:
            w.writerow(row)

    json_path = out_dir / "htm_threshold_comparison.json"
    payload = {
        "thresholds": thresholds,
        "columns": {str(t): htm_vector_column_key(t) for t in thresholds},
        "by_condition": by_condition,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_base = out_dir / "htm_threshold_comparison"
    plot_grouped_bars(
        by_condition,
        thresholds,
        out_base,
        dpi=args.dpi,
        formats=args.format,
    )
    plot_string_vs_vector_panel_grid(
        by_condition,
        thresholds,
        out_base,
        dpi=args.dpi,
        formats=args.format,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        f"Wrote {out_dir / 'htm_threshold_comparison.png'} — stacked overview; "
        f"{out_base}__string_vs_vector_panels.png — substring vs vector side-by-side per pipeline"
    )
    for cond, rows in sorted(by_condition.items()):
        if not rows:
            continue
        keys = ["htm_string"] + [htm_vector_column_key(t) for t in thresholds]
        means = {k: float(sum(float(r[k]) for r in rows) / len(rows)) for k in keys}
        print(f"  [{cond}] mean across systems: htm_string={means['htm_string']:.4f}", end="")
        for t in thresholds:
            k = htm_vector_column_key(t)
            print(f", {k}={means[k]:.4f}", end="")
        print()


if __name__ == "__main__":
    main()
