#!/usr/bin/env python3
"""Run translation pipeline systems S1–S5. See README.md for flags and S5 backends."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cuda_ld_path import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

from pipeline.graph import TermGraph
from pipeline.planning import load_or_compute_locks
from pipeline.systems.data_io import parse_exclude_segment_ids
from pipeline.systems.models import unload_mistral, unload_nllb

_SYS_MODULES = {
    "s1": "pipeline.systems.s1_nllb",
    "s2": "pipeline.systems.s2_mistral_doc",
    "s3": "pipeline.systems.s3_graphrag",
    "s4": "pipeline.systems.s4_rerank",
    "s5": "pipeline.systems.s5_logit",
}


def _resolve_under_root(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _ids_in_file(out_path: Path) -> set[str]:
    if not out_path.is_file():
        return set()
    done: set[str] = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return done


def _open_result(out_path: Path, resume: bool) -> tuple[set[str], str]:
    if resume and out_path.is_file():
        return _ids_in_file(out_path), "a"
    return set(), "w"


def _expand_run_systems(systems: list[str]) -> list[str]:
    """Preserve user order; ``all`` expands to S1–S5 in pipeline order."""
    if any(s == "all" for s in systems):
        return ["s1", "s2", "s3", "s4", "s5"]
    out: list[str] = []
    seen: set[str] = set()
    for s in systems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _run_s5(
    mod,
    segments_path: Path,
    graph,
    locks: dict[str, str],
    limit: int | None,
    backend: str,
    out_path: Path,
    resume: bool,
    exclude_segment_ids,
) -> None:
    skip_ids, mode = _open_result(out_path, resume)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode, encoding="utf-8") as out_f:
        mod.run(
            segments_path,
            out_f,
            graph=graph,
            locks=locks,
            limit=limit,
            s5_backend=backend,
            skip_ids=skip_ids,
            exclude_segment_ids=exclude_segment_ids,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Run translation pipeline systems S1–S5.")
    p.add_argument(
        "--system",
        nargs="+",
        metavar="SYS",
        choices=["s1", "s2", "s3", "s4", "s5", "all"],
        default=["all"],
        help="One or more of s1…s5, or 'all' for the full pipeline (default: all). Example: --system s3 s4 s5",
    )
    p.add_argument(
        "--segments",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner.jsonl",
    )
    p.add_argument("--limit", type=int, default=None, help="Max segments (debug).")
    p.add_argument(
        "--no-planning",
        action="store_true",
        help="Skip planning locks; S3–S5 use graph names only (no global locks).",
    )
    p.add_argument(
        "--recompute-planning",
        action="store_true",
        help="Recompute planning_locks.json even if cache exists.",
    )
    p.add_argument(
        "--s5-backend",
        choices=["nllb", "mistral", "both"],
        default="nllb",
        help="S5: nllb → s5.jsonl, mistral → s5_mistral.jsonl under --results-dir; both → run both.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append; skip segment ids already in each result JSONL.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "ad_hoc",
        help="Directory for s1.jsonl … s5_mistral.jsonl (default: results/ad_hoc/).",
    )
    p.add_argument(
        "--grounding-mode",
        choices=["string", "vector", "vector_llm"],
        default="string",
        help="Neo4j term grounding: string (default), vector index, or vector + LLM rerank.",
    )
    p.add_argument(
        "--s4-candidates",
        type=int,
        default=3,
        metavar="N",
        help="S4: number of stochastic candidates per segment (1–5; default 3).",
    )
    p.add_argument(
        "--exclude-segment-ids",
        type=str,
        default="",
        metavar="IDS",
        help=(
            "Comma-separated segment ids to skip (default: none). "
            "Example: 48_028 omits Section 4.8 Table 2 block."
        ),
    )
    args = p.parse_args()

    if not (1 <= args.s4_candidates <= 5):
        raise SystemExit("--s4-candidates must be between 1 and 5")

    results_dir = _resolve_under_root(args.results_dir)
    segments_path = _resolve_under_root(args.segments)
    if not segments_path.is_file():
        raise SystemExit(f"Missing segments file: {segments_path}")

    exclude_segment_ids = parse_exclude_segment_ids(args.exclude_segment_ids or None)

    graph = TermGraph(grounding_mode=args.grounding_mode)
    try:
        locks: dict[str, str] = {}
        if not args.no_planning:
            locks = load_or_compute_locks(
                segments_path,
                graph,
                recompute=args.recompute_planning,
                exclude_segment_ids=exclude_segment_ids,
            )

        run_list = _expand_run_systems(list(args.system))

        for name in run_list:
            mod = importlib.import_module(_SYS_MODULES[name])
            if name == "s5":
                if args.s5_backend == "both":
                    unload_mistral()
                    _run_s5(
                        mod,
                        segments_path,
                        graph,
                        locks,
                        args.limit,
                        "nllb",
                        results_dir / "s5.jsonl",
                        args.resume,
                        exclude_segment_ids,
                    )
                    unload_nllb()
                    _run_s5(
                        mod,
                        segments_path,
                        graph,
                        locks,
                        args.limit,
                        "mistral",
                        results_dir / "s5_mistral.jsonl",
                        args.resume,
                        exclude_segment_ids,
                    )
                elif args.s5_backend == "mistral":
                    _run_s5(
                        mod,
                        segments_path,
                        graph,
                        locks,
                        args.limit,
                        "mistral",
                        results_dir / "s5_mistral.jsonl",
                        args.resume,
                        exclude_segment_ids,
                    )
                else:
                    unload_mistral()
                    _run_s5(
                        mod,
                        segments_path,
                        graph,
                        locks,
                        args.limit,
                        "nllb",
                        results_dir / "s5.jsonl",
                        args.resume,
                        exclude_segment_ids,
                    )
                continue

            out_path = results_dir / f"{name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            skip_ids, mode = _open_result(out_path, args.resume)
            with out_path.open(mode, encoding="utf-8") as out_f:
                if name == "s4":
                    mod.run(
                        segments_path,
                        out_f,
                        graph=graph,
                        locks=locks,
                        limit=args.limit,
                        skip_ids=skip_ids,
                        s4_candidates=args.s4_candidates,
                        exclude_segment_ids=exclude_segment_ids,
                    )
                else:
                    mod.run(
                        segments_path,
                        out_f,
                        graph=graph,
                        locks=locks,
                        limit=args.limit,
                        skip_ids=skip_ids,
                        exclude_segment_ids=exclude_segment_ids,
                    )
            if name == "s1":
                unload_nllb()
    finally:
        graph.close()


if __name__ == "__main__":
    main()
