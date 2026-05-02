#!/usr/bin/env python3
"""BLEU/chrF/COMET and terminology metrics (see README).

CCR: NER spans grounded in Neo4j (dataset coverage).
HTM: optional ``--gold-terms`` audit list + graph checks on **English** ``hyp`` (Neo4j unless ``--no-graph``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.ccr import compute_ccr, compute_ccr_stats
from pipeline.metrics.corpus_scores import corpus_bleu as _bleu_score, corpus_chrf as _chrf_score
from pipeline.metrics.eval_manifest import EVAL_FILES
from pipeline.metrics.htm import compute_htm
from pipeline.graph import TermGraph
from pipeline.systems.data_io import load_all_segments, parse_exclude_segment_ids
from pipeline.systems.inference_timing import inference_mean_p95

from pipeline.metrics.comet_score import corpus_comet_da as _try_comet
from pipeline.metrics.eval_io import align_refs as _align_refs
from pipeline.metrics.eval_io import align_src_hyp_ref as _align_src_hyp_ref
from pipeline.metrics.eval_io import load_results_jsonl as _load_jsonl

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


def _neo4j_help() -> str:
    return (
        "Neo4j is not reachable (bolt connection refused).\n"
        "  • Start the DB:    docker compose up -d    (project root; see docker-compose.yml)\n"
        "  • Or omit graph metrics:    --no-graph    (BLEU/chrF/COMET only; skips CCR and HTM)\n"
    )


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
        help="Do not connect to Neo4j: skip dataset CCR and HTM (BLEU/chrF/COMET unchanged).",
    )
    ap.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        help="Comma-separated ids to omit from scoring (must match pipeline run). Example: 48_028",
    )
    ap.add_argument(
        "--gold-terms",
        type=Path,
        default=None,
        help="Gold terminology JSON for HTM. If omitted with Neo4j, HTM is not scored (nan).",
    )
    args = ap.parse_args()
    results_dir = _resolve_results_dir(args.results_dir)
    partial = args.partial
    exclude_seg = parse_exclude_segment_ids(args.exclude_segment_ids or None)
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
            except _NEO4J_CONN_ERRORS:
                raise SystemExit(_neo4j_help()) from None
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
    gold_terms: list[dict[str, Any]] | None = None
    if args.gold_terms is not None:
        gp = args.gold_terms
        gold_path = gp if gp.is_absolute() else (ROOT / gp)
        if not gold_path.is_file():
            raise SystemExit(f"Gold terms file not found: {gold_path}")
        gold_terms = json.loads(gold_path.read_text(encoding="utf-8"))
    id_to_ref: dict[str, str] = {}
    id_to_src: dict[str, str] = {}
    segment_rows = load_all_segments(seg_path, exclude_segment_ids=exclude_seg)
    n_expected = len(segment_rows)
    keep_ids = {row["id"] for row in segment_rows}
    for row in segment_rows:
        id_to_ref[row["id"]] = row["en_ref"]
        id_to_src[row["id"]] = row["fr"]

    use_graph = not args.no_graph
    if args.no_graph:
        print(
            "[no-graph] Skipping Neo4j — HTM and dataset CCR omitted.\n",
            file=sys.stderr,
        )

    graph = TermGraph(grounding_mode=args.grounding_mode) if use_graph else None
    if use_graph and gold_terms is None:
        print(
            "[no --gold-terms] HTM omitted (nan). Pass --gold-terms PATH.json for FR→EN gold rows.",
            file=sys.stderr,
        )
    try:
        if use_graph:
            try:
                ccr_val = compute_ccr(segment_rows, graph)
            except _NEO4J_CONN_ERRORS:
                raise SystemExit(_neo4j_help()) from None
        else:
            ccr_val = float("nan")

        label_w = max(len(t[0]) for t in EVAL_FILES)
        hdr = f"{'System':<{label_w}}"
        print(f"{hdr} {'BLEU':>8} {'chrF':>8} {'COMET':>8} {'HTM':>8} {'CCR':>8}")
        print(f"{'(dataset)':<{label_w}} {'':>8} {'':>8} {'':>8} {'':>8} {_fmt_metric(ccr_val)}")
        for name, fname in EVAL_FILES:
            path = results_dir / fname
            if not path.is_file():
                print(f"{name:<{label_w}} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}")
                continue
            res = _load_jsonl(path, partial=partial)
            res = [r for r in res if r.get("id") in keep_ids]
            if len(res) < n_expected:
                print(
                    f"WARNING: {path} has {len(res)} rows; expected {n_expected} from segments "
                    f"— skipping {name} (partial run).",
                    file=sys.stderr,
                )
                print(f"{name:<{label_w}} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}")
                continue
            if not res:
                print(f"{name:<{label_w}} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}")
                continue
            hyps, refs = _align_refs(res, id_to_ref)
            if not hyps:
                bleu_s = chrf_s = 0.0
                comet_s = None
            else:
                bleu_s = _bleu_score(hyps, refs)
                chrf_s = _chrf_score(hyps, refs)
                srcs, hyps_c, refs_c = _align_src_hyp_ref(res, id_to_ref, id_to_src)
                comet_s = _try_comet(srcs, hyps_c, refs_c)

            if graph is not None and gold_terms is not None:
                try:
                    htm_v = compute_htm(res, gold_terms, graph)
                except _NEO4J_CONN_ERRORS:
                    raise SystemExit(_neo4j_help()) from None
            else:
                htm_v = float("nan")

            comet_str = f"{comet_s:8.3f}" if comet_s is not None else f"{'--':>8}"
            print(
                f"{name:<{label_w}} {bleu_s:8.2f} {chrf_s:8.2f} {comet_str} "
                f"{_fmt_metric(htm_v)} {_fmt_metric(ccr_val)}"
            )

        tw = max(label_w, len("System"))
        hdr_t = f"{'System':<{tw}}"
        print()
        print(
            "Inference wall time (s/segment, end-to-end per segment). "
            "Cross-system totals are not comparable when models sit on CPU vs GPU."
        )
        print(f"{hdr_t} {'mean_s':>10} {'p95_s':>10}")
        for name, fname in EVAL_FILES:
            path = results_dir / fname
            if not path.is_file():
                print(f"{name:<{tw}} {'--':>10} {'--':>10}")
                continue
            res = _load_jsonl(path, partial=partial)
            res = [r for r in res if r.get("id") in keep_ids]
            if len(res) < n_expected:
                print(
                    f"WARNING: {path} has {len(res)} rows; expected {n_expected} from segments "
                    f"— skipping {name} timing (partial run).",
                    file=sys.stderr,
                )
                print(f"{name:<{tw}} {'--':>10} {'--':>10}")
                continue
            if not res:
                print(f"{name:<{tw}} {'--':>10} {'--':>10}")
                continue
            mean_s, p95_s = inference_mean_p95(res)
            if mean_s is None or p95_s is None:
                print(f"{name:<{tw}} {'--':>10} {'--':>10}")
            else:
                print(f"{name:<{tw}} {mean_s:10.3f} {p95_s:10.3f}")
    finally:
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
