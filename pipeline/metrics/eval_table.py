"""Collect per-system metric rows and write ``scores_summary.csv`` (shared by eval + plot CLIs)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from pipeline.graph import TermGraph
from pipeline.metrics.ccr import compute_ccr
from pipeline.metrics.comet_score import corpus_comet_da as _try_comet
from pipeline.metrics.corpus_scores import (
    corpus_bleu,
    corpus_chrf,
    macro_bleu_doc_concat,
    macro_corpus_metric_by_group,
)
from pipeline.metrics.eval_io import (
    align_hyp_ref_by_doc,
    align_src_hyp_ref,
    load_results_jsonl,
)
from pipeline.metrics.eval_manifest import EVAL_FILES
from pipeline.metrics.htm import (
    compute_htm,
    compute_htm_en_ref,
    compute_htm_hyp_vs_ref,
    compute_htm_vector,
    htm_vector_column_key,
)
from pipeline.systems.inference_timing import inference_mean_p95

try:
    from neo4j.exceptions import ServiceUnavailable
except ImportError:

    class ServiceUnavailable(Exception):
        """Placeholder if neo4j is not installed."""


NEO4J_CONN_ERRORS: tuple[type[BaseException], ...] = (
    ServiceUnavailable,
    ConnectionError,
    TimeoutError,
)


def neo4j_connection_help() -> str:
    return (
        "Neo4j is not reachable (bolt connection refused).\n"
        "  • Start the DB:    docker compose up -d    (project root; see docker-compose.yml)\n"
        "  • Or omit graph metrics:    --no-graph    (BLEU/chrF/COMET only; skips CCR, HTM, and rHTM)\n"
    )


DISPLAY_NAMES: dict[str, str] = {
    "s1": "S1 NLLB",
    "s2": "S2 Mistral (doc)",
    "s3": "S3 GraphRAG",
    "s4": "S4 rerank",
    "s5": "S5 NLLB + boost",
    "s5_mistral": "S5 Mistral + boost",
}


def display_label_for_system(lab: str) -> str:
    """Short display name for a system label (matches figure tables)."""
    return DISPLAY_NAMES.get(lab, lab)


def document_key_for_segment_row(row: dict[str, Any]) -> str:
    """Stable document id for macro corpus metrics.

    Uses ``document_id`` from the segment row when present; otherwise the prefix
    of ``id`` before the first ``_`` (e.g. ``48`` for ``48_001``), or the full
    ``id`` when no underscore is present.
    """
    doc = row.get("document_id")
    if doc is not None and str(doc).strip():
        return str(doc).strip()
    sid = str(row["id"])
    if "_" in sid:
        return sid.split("_", 1)[0]
    return sid


def collect_system_metric_rows(
    *,
    results_dir: Path,
    id_to_ref: dict[str, str],
    id_to_src: dict[str, str],
    graph: TermGraph | None,
    segment_rows: list[dict],
    partial: bool,
    with_comet: bool,
    keep_segment_ids: frozenset[str] | None = None,
    htm_vector_thresholds: list[float] | None = None,
    htm_embed_model: str | None = None,
    n_expected: int | None = None,
    fill_missing: bool = False,
    out_warnings: list[str] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Build one dict per ``EVAL_FILES`` system (fluency, HTM, timing, …).

    When ``fill_missing`` is True (console eval), emit NaN-filled rows for missing or
    unusable JSONLs so the table always has one row per system in manifest order, and
    append human-readable lines to ``out_warnings`` when given.
    """
    htm_vec_thr = htm_vector_thresholds or []
    htm_en_ref = float("nan")
    if graph is None:
        ccr = float("nan")
    else:
        ccr = compute_ccr(segment_rows, graph)
        htm_en_ref = compute_htm_en_ref(segment_rows, graph)
    id_to_segment = {str(row["id"]): row for row in segment_rows}
    id_to_doc = {str(row["id"]): document_key_for_segment_row(row) for row in segment_rows}
    out: list[dict[str, Any]] = []

    def _nan_row(label: str) -> dict[str, Any]:
        row_out: dict[str, Any] = {
            "label": label,
            "display": display_label_for_system(label),
            "bleu": float("nan"),
            "chrf": float("nan"),
            "bleu_doc_macro": float("nan"),
            "bleu_doc_concat": float("nan"),
            "chrf_doc_macro": float("nan"),
            "comet": None,
            "htm": float("nan"),
            "htm_hyp_ref_agreement": float("nan"),
            "htm_en_ref_dataset": htm_en_ref,
            "mean_s": None,
            "p95_s": None,
        }
        for t in htm_vec_thr:
            row_out[htm_vector_column_key(t)] = float("nan")
        return row_out

    for label, fname in EVAL_FILES:
        p = results_dir / fname
        if not p.is_file():
            if fill_missing:
                if out_warnings is not None:
                    out_warnings.append(f"WARNING: missing file {p} — skipping {label} (no row scores).")
                out.append(_nan_row(label))
            continue
        res = load_results_jsonl(p, partial=partial)
        if keep_segment_ids is not None:
            res = [r for r in res if r.get("id") in keep_segment_ids]
        if fill_missing and n_expected is not None and len(res) < n_expected:
            if out_warnings is not None:
                out_warnings.append(
                    f"WARNING: {p} has {len(res)} rows; expected {n_expected} from segments "
                    f"— skipping {label} (partial run)."
                )
            out.append(_nan_row(label))
            continue
        if not res:
            if fill_missing:
                if out_warnings is not None:
                    out_warnings.append(f"WARNING: {p} empty after id filter — skipping {label}.")
                out.append(_nan_row(label))
            continue
        hyps, refs, doc_keys = align_hyp_ref_by_doc(res, id_to_ref, id_to_doc)
        if not hyps:
            if fill_missing:
                if out_warnings is not None:
                    out_warnings.append(f"WARNING: {p} no aligned hyps — skipping {label}.")
                out.append(_nan_row(label))
            continue
        b = corpus_bleu(hyps, refs)
        c = corpus_chrf(hyps, refs)
        b_doc = macro_corpus_metric_by_group(corpus_bleu, hyps, refs, doc_keys)
        b_concat = macro_bleu_doc_concat(hyps, refs, doc_keys)
        c_doc = macro_corpus_metric_by_group(corpus_chrf, hyps, refs, doc_keys)
        comet_v: float | None = None
        if with_comet:
            srcs, h2, r2 = align_src_hyp_ref(res, id_to_ref, id_to_src)
            comet_v = _try_comet(srcs, h2, r2)
        if graph is None:
            htm_v = float("nan")
            hyp_ref_agreement = float("nan")
        else:
            htm_v = compute_htm(res, graph, id_to_segment)
            hyp_ref_agreement = compute_htm_hyp_vs_ref(res, graph, id_to_segment)
        mean_s, p95_s = inference_mean_p95(res) or (None, None)
        row_out: dict[str, Any] = {
            "label": label,
            "display": display_label_for_system(label),
            "bleu": b,
            "chrf": c,
            "bleu_doc_macro": b_doc,
            "bleu_doc_concat": b_concat,
            "chrf_doc_macro": c_doc,
            "comet": comet_v,
            "htm": htm_v,
            "htm_hyp_ref_agreement": hyp_ref_agreement,
            "htm_en_ref_dataset": htm_en_ref,
            "mean_s": mean_s,
            "p95_s": p95_s,
        }
        for t in htm_vec_thr:
            k = htm_vector_column_key(t)
            if graph is None:
                row_out[k] = float("nan")
            else:
                try:
                    row_out[k] = float(
                        compute_htm_vector(
                            res,
                            graph,
                            id_to_segment,
                            similarity_threshold=t,
                            embed_model_name=htm_embed_model,
                        )
                    )
                except Exception:
                    row_out[k] = float("nan")
        out.append(row_out)
    return out, ccr


def write_scores_summary_csv(rows: list[dict[str, Any]], ccr: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = sorted(k for k in rows[0] if str(k).startswith("htm_vector_")) if rows else []
    fieldnames = [
        "label",
        "display",
        "bleu",
        "chrf",
        "bleu_doc_macro",
        "bleu_doc_concat",
        "chrf_doc_macro",
        "comet",
        "htm",
        "htm_hyp_ref_agreement",
        *vec,
        "htm_en_ref_dataset",
        "ccr_dataset",
        "mean_s",
        "p95_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out_row = {k: r.get(k) for k in fieldnames if k != "ccr_dataset"}
            out_row["ccr_dataset"] = ccr
            w.writerow(out_row)
