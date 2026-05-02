#!/usr/bin/env python3
"""Compare CCR across Neo4j grounding modes (string / vector / vector_llm)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.ccr import compute_ccr_stats
from pipeline.graph import VECTOR_INDEX_NAME, TermGraph
from pipeline.systems.data_io import load_all_segments


def _resolve(p: Path | None, default: Path) -> Path:
    x = p if p is not None else default
    return x if x.is_absolute() else (ROOT / x)


def require_vector_index() -> None:
    """Exit if Neo4j has no ``meddra_fr_embedding`` vector index (vector modes would lie)."""
    load_dotenv(ROOT / ".env")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASS", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    ok = False
    try:
        with driver.session() as session:
            try:
                for rec in session.run("SHOW INDEXES YIELD name"):
                    if rec.get("name") == VECTOR_INDEX_NAME:
                        ok = True
                        break
            except Exception:
                rec = session.run(
                    "CALL db.indexes() YIELD name WHERE name = $iname RETURN count(*) AS n",
                    iname=VECTOR_INDEX_NAME,
                ).single()
                ok = rec is not None and int(rec["n"]) > 0
    except Exception as e:
        hint = ""
        if "refused" in str(e).lower() or "Couldn't connect" in str(e) or "Failed to establish" in str(e):
            hint = (
                "\n  Hint: start Neo4j (e.g. systemctl start neo4j, or docker), then set "
                "NEO4J_URI / NEO4J_USER / NEO4J_PASS in .env to match the Neo4j instance."
            )
        print(
            f"ERROR: Could not verify vector index (Neo4j query failed): {e}\n"
            f"  Expected index name: {VECTOR_INDEX_NAME!r}. Run experiments/vector_grounding/build_graph_embeddings.py first."
            f"{hint}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    finally:
        driver.close()
    if not ok:
        print(
            "ERROR: Vector index not built. Run experiments/vector_grounding/build_graph_embeddings.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def vector_ccr_only(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Vector grounding only: CCR + grounded/total/fallbacks."""
    g = TermGraph(grounding_mode="vector", vector_score_threshold=threshold)
    try:
        g.reset_grounding_stats()
        ccr, n_ext, n_gr = compute_ccr_stats(rows, g)
        st = g.get_grounding_stats()
        fb = int(st.get("vector_fallbacks", 0))
        return {
            "ccr": round(float(ccr), 6),
            "grounded": int(n_gr),
            "total": int(n_ext),
            "fallbacks": fb,
        }
    finally:
        g.close()


def vector_ccr_stats_with_graph(rows: list[dict[str, Any]], threshold: float, g: TermGraph) -> dict[str, Any]:
    """Reuse an existing vector-mode graph (avoids reloading the sentence encoder)."""
    g.set_vector_threshold(threshold)
    g.reset_grounding_stats()
    ccr, n_ext, n_gr = compute_ccr_stats(rows, g)
    st = g.get_grounding_stats()
    fb = int(st.get("vector_fallbacks", 0))
    return {
        "ccr": round(float(ccr), 6),
        "grounded": int(n_gr),
        "total": int(n_ext),
        "fallbacks": fb,
    }


def run_ccr_modes(
    rows: list[dict[str, Any]],
    *,
    vector_threshold: float,
    ambiguous_out: Path,
    modes: tuple[str, ...] = ("string", "vector", "vector_llm"),
) -> tuple[list[tuple[str, float, int, int, float, int, int]], list[dict[str, Any]]]:
    """Return table rows (mode, ccr, amb, fb, mc, n_ext, n_gr) and vector rejects."""
    results: list[tuple[str, float, int, int, float, int, int]] = []
    vector_rejects: list[dict[str, Any]] = []

    ambiguous_path = ambiguous_out if ambiguous_out.is_absolute() else ROOT / ambiguous_out

    for mode in modes:
        g = TermGraph(grounding_mode=mode, vector_score_threshold=vector_threshold)
        try:
            g.reset_grounding_stats()
            ccr, n_ext, n_gr = compute_ccr_stats(rows, g)
            st = g.get_grounding_stats()
            amb = int(st.get("string_ambiguous_warnings", 0))
            fb = int(st.get("vector_fallbacks", 0))
            mc = float(st.get("mean_candidates", 0.0))
            results.append((mode, float(ccr), amb, fb, mc, n_ext, int(n_gr)))
            if mode == "string":
                g.write_ambiguous_report(ambiguous_path)
            if mode == "vector":
                vector_rejects = g.take_vector_rejected_spans()
        finally:
            g.close()

    return results, vector_rejects


def main() -> None:
    ap = argparse.ArgumentParser(description="CCR comparison for grounding modes.")
    ap.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="NER segments JSONL (default: data/section48/segments_ner_biollm.jsonl).",
    )
    ap.add_argument(
        "--ambiguous-out",
        type=Path,
        default=ROOT / "data" / "section48" / "ambiguous_terms.txt",
        help="File listing ambiguous FR terms under string mode (tab-separated counts).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max segments (debug).")
    ap.add_argument(
        "--vector-threshold",
        type=float,
        default=0.75,
        help="Min Neo4j top-1 cosine similarity for accepting a vector hit; below → ungrounded + Vec FB.",
    )
    ap.add_argument(
        "--vector-rejects-out",
        type=Path,
        default=None,
        help="Write low-confidence vector rejections as JSON array (default: skip).",
    )
    ap.add_argument(
        "--json-sweep-out",
        type=Path,
        default=None,
        help="Write {\"0.70\": {...}, ...} vector-only CCR stats for --sweep-thresholds.",
    )
    ap.add_argument(
        "--sweep-thresholds",
        type=str,
        default="0.70,0.75,0.80",
        help="Comma-separated thresholds for --json-sweep-out (vector mode only).",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["string", "vector", "vector_llm"],
        default=["string", "vector", "vector_llm"],
        help="Grounding modes to evaluate (default: all three).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG logs (vector_llm selections).",
    )
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    seg_path = _resolve(args.segments, ROOT / "data" / "section48" / "segments_ner_biollm.jsonl")
    if not seg_path.is_file():
        raise SystemExit(f"Segments file not found: {seg_path}")

    rows = load_all_segments(seg_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    mode_tuple = tuple(args.modes)
    needs_vector_index = bool(args.json_sweep_out) or any(
        m in ("vector", "vector_llm") for m in mode_tuple
    )
    if needs_vector_index:
        require_vector_index()

    results, vector_rejects = run_ccr_modes(
        rows,
        vector_threshold=args.vector_threshold,
        ambiguous_out=args.ambiguous_out,
        modes=mode_tuple,
    )

    if args.vector_rejects_out is not None:
        rp = args.vector_rejects_out if args.vector_rejects_out.is_absolute() else ROOT / args.vector_rejects_out
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(vector_rejects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Vector rejections ({len(vector_rejects)} spans): {rp}")

    if args.json_sweep_out is not None:
        sweep_path = args.json_sweep_out if args.json_sweep_out.is_absolute() else ROOT / args.json_sweep_out
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [x.strip() for x in args.sweep_thresholds.split(",") if x.strip()]
        out_obj: dict[str, dict[str, Any]] = {}
        g_vec = TermGraph(grounding_mode="vector", vector_score_threshold=float(parts[0]) if parts else 0.75)
        try:
            for t_str in parts:
                thr = float(t_str)
                key = f"{thr:.2f}"
                out_obj[key] = vector_ccr_stats_with_graph(rows, thr, g_vec)
        finally:
            g_vec.close()
        sweep_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Threshold sweep written to: {sweep_path}")

    print()
    hdr = f"{'Grounding mode':<16} | {'CCR':>7} | {'Ambiguous':>10} | {'Vec FB':>8} | {'Mean cand':>10} | {'Spans':>6}"
    print(hdr)
    print("-" * len(hdr))
    for mode, ccr, amb, fb, mc, n_ext, _n_gr in results:
        fb_s = str(fb) if mode != "string" else "--"
        print(f"{mode:<16} | {ccr:7.4f} | {amb:10d} | {fb_s:>8} | {mc:10.4f} | {n_ext:6d}")
    print()
    ambiguous_path = args.ambiguous_out if args.ambiguous_out.is_absolute() else ROOT / args.ambiguous_out
    print(f"Ambiguous terms (string mode) written to: {ambiguous_path}")
    print(
        "Note: vector / vector_llm need experiments/vector_grounding/build_graph_embeddings.py + Neo4j vector index.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
