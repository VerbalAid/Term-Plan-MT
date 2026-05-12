#!/usr/bin/env python3
"""Run translation pipeline systems S1–S6. See ``tools/README.md`` for flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

from pipeline import TermGraph, load_or_compute_locks
from systems import parse_exclude_segment_ids, unload_mistral, unload_nllb, run_system, load_all_segments

# Systems are dispatched via systems.run_system() — no importlib needed.

DEFAULT_GLOSSARY = ROOT / "data" / "section48" / "gold_glossary.json"


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
    """Preserve user order; ``all`` expands to S1–S6 in pipeline order."""
    if any(s == "all" for s in systems):
        return ["s1", "s2", "s3", "s4", "s5", "s6"]
    out: list[str] = []
    seen: set[str] = set()
    for s in systems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _needs_neo4j(run_list: list[str]) -> bool:
    return any(name in ("s3", "s4", "s5") for name in run_list)



def main() -> None:
    p = argparse.ArgumentParser(description="Run translation pipeline systems S1-S6.")
    p.add_argument("--system", nargs="+", metavar="SYS",
                   choices=["s1","s2","s3","s4","s5","s6","all"], default=["all"])
    p.add_argument("--segments", type=Path,
                   default=ROOT / "data" / "section48" / "segments_ner.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-planning", action="store_true")
    p.add_argument("--recompute-planning", action="store_true")
    p.add_argument("--s5-backend", choices=["nllb","mistral","both"], default="nllb")
    p.add_argument("--s6-backend", choices=["nllb","mistral","both"], default="nllb")
    p.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ad_hoc")
    p.add_argument("--grounding-mode", choices=["string","vector","vector_llm"], default="string")
    p.add_argument("--s4-candidates", type=int, default=3, metavar="N")
    p.add_argument("--exclude-segment-ids", type=str, default="", metavar="IDS")
    args = p.parse_args()

    results_dir   = _resolve_under_root(args.results_dir)
    segments_path = _resolve_under_root(args.segments)
    glossary_path = _resolve_under_root(args.glossary)
    if not segments_path.is_file():
        raise SystemExit(f"Missing segments file: {segments_path}")

    exclude_ids = parse_exclude_segment_ids(args.exclude_segment_ids or None)
    run_list    = _expand_run_systems(list(args.system))

    # Load glossary for S6 (oracle ablation only).
    glossary: dict = {}
    if "s6" in run_list and glossary_path.is_file():
        import json as _json
        glossary = {item["fr"]: item["en"] for item in _json.loads(glossary_path.read_text("utf-8"))
                    if isinstance(item, dict) and item.get("fr") and item.get("en")}

    graph = None
    locks: dict = {}
    needs_neo4j = _needs_neo4j(run_list)
    try:
        if needs_neo4j:
            graph = TermGraph(grounding_mode=args.grounding_mode)
            if not args.no_planning:
                locks = load_or_compute_locks(segments_path, graph, recompute=args.recompute_planning)

        for name in run_list:
            # S5 and S6 each have an NLLB and a Mistral variant.
            if name == "s5":
                variants = []
                if args.s5_backend in ("nllb", "both"):
                    variants.append(("s5", results_dir / "s5.jsonl", True))
                if args.s5_backend in ("mistral", "both"):
                    variants.append(("s5_mistral", results_dir / "s5_mistral.jsonl", False))
            elif name == "s6":
                variants = []
                if args.s6_backend in ("nllb", "both"):
                    variants.append(("s6", results_dir / "s6.jsonl", True))
                if args.s6_backend in ("mistral", "both"):
                    variants.append(("s6_mistral", results_dir / "s6_mistral.jsonl", False))
            else:
                variants = [(name, results_dir / f"{name}.jsonl", None)]

            for sys_key, out_path, _nllb_first in variants:
                if _nllb_first is True:
                    unload_mistral()
                elif _nllb_first is False:
                    unload_nllb()
                skip_ids: set[str] = set()
                if args.resume:
                    skip_ids = _ids_in_file(out_path)
                run_system(
                    sys_key,
                    segments_path,
                    out_path,
                    graph=graph,
                    locks=locks,
                    glossary=glossary,
                    limit=args.limit,
                    skip_ids=skip_ids,
                    exclude_ids=exclude_ids,
                )
    finally:
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
