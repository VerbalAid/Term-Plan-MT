#!/usr/bin/env python3
"""BLEU/chrF/COMET + CCR/HTM/rHTM + htm_hyp_ref_agreement for translation JSONLs (see README).

``--no-graph``: fluency only (skips Neo4j). Otherwise needs Docker Neo4j like ``run_pipeline``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.ccr import compute_ccr_stats
from pipeline.metrics.eval_manifest import EVAL_FILES
from pipeline.metrics.eval_table import (
    NEO4J_CONN_ERRORS,
    collect_system_metric_rows,
    neo4j_connection_help,
)
from pipeline.metrics.htm import htm_vector_column_key, parse_cosine_thresholds_csv
from pipeline.graph import TermGraph
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids


def _fmt_metric(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return f"{'--':>8}"
    return f"{x:8.3f}"


def _resolve_segments_path(segments: Path | None) -> Path:
    default = ROOT / "data" / "section48" / "segments_ner.jsonl"
    p = segments if segments is not None else default
    return p if p.is_absolute() else (ROOT / p)


def _resolve_results_dir(results_dir: Path | None) -> Path:
    default = ROOT / "results" / "ad_hoc"
    p = results_dir if results_dir is not None else default
    return p if p.is_absolute() else (ROOT / p)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate BLEU, chrF, optional COMET, HTM, CCR.")
    ap.add_argument(
        "--ccr-only",
        action="store_true",
        help="Only print dataset CCR (NER span grounding in Neo4j from segments_ner.jsonl). No system metrics.",
    )
    ap.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="JSONL with id/fr/en_ref/terms for CCR / dataset metrics (default: data/section48/segments_ner.jsonl).",
    )
    ap.add_argument(
        "--partial",
        action="store_true",
        help=(
            "Ignore broken JSONL lines; score only rows that load. Use while some systems are still writing."
        ),
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing s1.jsonl … s5_mistral.jsonl (default: results/ad_hoc/).",
    )
    ap.add_argument(
        "--grounding-mode",
        choices=["string", "vector", "vector_llm"],
        default="string",
        help="Neo4j grounding for CCR / HTM (default: string).",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="Do not connect to Neo4j: skip dataset CCR and HTM/rHTM (BLEU/chrF/COMET unchanged).",
    )
    ap.add_argument(
        "--no-comet",
        action="store_true",
        help="Skip COMET (faster; BLEU/chrF/HTM/CCR unchanged when Neo4j is used).",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids to omit from scoring (must match pipeline run). Example: 48_028",
    )
    ap.add_argument(
        "--htm-vector-thresholds",
        type=str,
        default="",
        help=(
            "Optional comma-separated cosine thresholds in [0,1] for vector HTM columns "
            "(e.g. 0.8,0.9). Requires Neo4j + sentence-transformers; slow on first model load."
        ),
    )
    args = ap.parse_args()
    try:
        htm_vec_thr = parse_cosine_thresholds_csv(args.htm_vector_thresholds)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    htm_vec_keys = [htm_vector_column_key(t) for t in htm_vec_thr]
    results_dir = _resolve_results_dir(args.results_dir)
    partial = args.partial
    exclude_seg = parse_exclude_segment_ids(args.exclude_segment_ids or None)

    def _vec_skip_cells() -> str:
        return "".join(f" {'--':>14}" for _ in htm_vec_thr) if htm_vec_thr else ""

    if args.ccr_only:
        if args.no_graph:
            raise SystemExit("--ccr-only requires Neo4j; remove --no-graph.")
        seg_path = _resolve_segments_path(args.segments)
        if not seg_path.is_file():
            raise SystemExit(f"Segments file not found: {seg_path}")
        segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude_seg)
        graph = TermGraph(grounding_mode=args.grounding_mode)
        try:
            try:
                ccr_val, n_ext, n_gr = compute_ccr_stats(segment_rows, graph)
            except NEO4J_CONN_ERRORS:
                raise SystemExit(neo4j_connection_help()) from None
        finally:
            graph.close()
        print(
            f"CCR (dataset): {ccr_val:.4f}  — {n_gr}/{n_ext} NER spans grounded in MedDRA "
            f"({seg_path})"
        )
        if n_ext == 0:
            print(
                "  No spans in segments (all `terms` empty or filtered). "
                "Lower --min-ner-score in prepare_data.py if the new model’s scores are below 0.80."
            )
        return

    if partial:
        print(
            "[partial] Scoring from available result rows; BLEU/chrF are not full-corpus if files are incomplete.\n"
        )

    seg_path = _resolve_segments_path(args.segments)
    if not seg_path.is_file():
        raise SystemExit(f"Segments file not found: {seg_path}")
    id_to_ref: dict[str, str] = {}
    id_to_src: dict[str, str] = {}
    segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude_seg)
    n_expected = len(segment_rows)
    keep_ids = frozenset(row["id"] for row in segment_rows)
    for row in segment_rows:
        id_to_ref[row["id"]] = row["en_ref"]
        id_to_src[row["id"]] = row["fr"]

    use_graph = not args.no_graph
    if args.no_graph:
        print(
            "[no-graph] Skipping Neo4j — HTM, rHTM, and dataset CCR omitted.\n",
            file=sys.stderr,
        )

    graph = TermGraph(grounding_mode=args.grounding_mode) if use_graph else None
    warn_buf: list[str] = []
    try:
        if use_graph:
            try:
                rows, ccr_val = collect_system_metric_rows(
                    results_dir=results_dir,
                    id_to_ref=id_to_ref,
                    id_to_src=id_to_src,
                    graph=graph,
                    segment_rows=segment_rows,
                    partial=partial,
                    with_comet=not args.no_comet,
                    keep_segment_ids=keep_ids,
                    htm_vector_thresholds=htm_vec_thr if htm_vec_thr else None,
                    htm_embed_model=None,
                    n_expected=n_expected,
                    fill_missing=True,
                    out_warnings=warn_buf,
                )
            except NEO4J_CONN_ERRORS:
                raise SystemExit(neo4j_connection_help()) from None
        else:
            ccr_val = float("nan")
            rows, _ = collect_system_metric_rows(
                results_dir=results_dir,
                id_to_ref=id_to_ref,
                id_to_src=id_to_src,
                graph=None,
                segment_rows=segment_rows,
                partial=partial,
                with_comet=not args.no_comet,
                keep_segment_ids=keep_ids,
                htm_vector_thresholds=htm_vec_thr if htm_vec_thr else None,
                htm_embed_model=None,
                n_expected=n_expected,
                fill_missing=True,
                out_warnings=warn_buf,
            )

        for ln in warn_buf:
            print(ln, file=sys.stderr)

        label_w = max(len(t[0]) for t in EVAL_FILES)
        hdr = f"{'System':<{label_w}}"
        vpad = "".join(f" {k:>14}" for k in htm_vec_keys)
        print(
            f"{hdr} {'BLEU':>8} {'chrF':>8} {'BLdoc':>8} {'CFdoc':>8} {'BLcn':>8} {'COMET':>8} {'HTM':>8}{vpad} "
            f"{'HypRefAg':>8} {'rHTM':>8} {'CCR':>8}"
        )
        print(
            "  HypRefAg = htm_hyp_ref_agreement: 1.0 = perfect agreement with human translator "
            "ontology placement (hyp vs en_ref); 0.0 = maximum deviation.",
            file=sys.stderr,
        )
        ds_pad = "".join(f" {'':>14}" for _ in htm_vec_keys)
        htm_en_ref_val = rows[0]["htm_en_ref_dataset"] if rows else float("nan")
        print(
            f"{'(dataset)':<{label_w}} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8}{ds_pad} "
            f"{'--':>8} {_fmt_metric(float(htm_en_ref_val))} {_fmt_metric(ccr_val)}"
        )
        for row in rows:
            name = row["label"]
            bleu_s = float(row["bleu"]) if row.get("bleu") is not None else float("nan")
            chrf_s = float(row["chrf"]) if row.get("chrf") is not None else float("nan")
            bleu_d = float(row["bleu_doc_macro"]) if row.get("bleu_doc_macro") is not None else float("nan")
            bleu_cn = float(row["bleu_doc_concat"]) if row.get("bleu_doc_concat") is not None else float("nan")
            chrf_d = float(row["chrf_doc_macro"]) if row.get("chrf_doc_macro") is not None else float("nan")
            comet_s = row.get("comet")
            htm_v = row.get("htm")
            if isinstance(bleu_s, float) and math.isnan(bleu_s):
                print(
                    f"{name:<{label_w}} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}"
                    f"{_vec_skip_cells()} {'--':>8} {'--':>8} {'--':>8}"
                )
                continue
            vec_cells = ""
            for t in htm_vec_thr:
                k = htm_vector_column_key(t)
                hv = row.get(k)
                if isinstance(hv, float) and math.isnan(hv):
                    vec_cells += f" {'--':>14}"
                else:
                    vec_cells += f" {float(hv):14.6f}"
            comet_str = f"{comet_s:8.3f}" if comet_s is not None else f"{'--':>8}"
            htm_f = float(htm_v) if htm_v is not None else float("nan")
            hyp_ref = row.get("htm_hyp_ref_agreement")
            hyp_ref_f = float(hyp_ref) if hyp_ref is not None else float("nan")
            print(
                f"{name:<{label_w}} {bleu_s:8.2f} {chrf_s:8.2f} {bleu_d:8.2f} {chrf_d:8.2f} {bleu_cn:8.2f} {comet_str} "
                f"{_fmt_metric(htm_f)}{vec_cells} {_fmt_metric(hyp_ref_f)} {_fmt_metric(float(htm_en_ref_val))} {_fmt_metric(ccr_val)}"
            )

        tw = max(label_w, len("System"))
        hdr_t = f"{'System':<{tw}}"
        print()
        print(
            "Inference wall time (s/segment, end-to-end per segment). "
            "Cross-system totals are not comparable when models sit on CPU vs GPU."
        )
        print(f"{hdr_t} {'mean_s':>10} {'p95_s':>10}")
        for row in rows:
            name = str(row["label"])
            mean_s, p95_s = row.get("mean_s"), row.get("p95_s")
            if mean_s is None or p95_s is None:
                print(f"{name:<{tw}} {'--':>10} {'--':>10}")
            else:
                print(f"{name:<{tw}} {float(mean_s):10.3f} {float(p95_s):10.3f}")
    finally:
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
