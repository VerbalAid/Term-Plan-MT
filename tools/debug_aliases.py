#!/usr/bin/env python3
"""Debug MedDRA ancestor chains for grounded French terms (Neo4j required).

Example:
  NEO4J_URI=bolt://127.0.0.1:7687 PYTHONPATH=. python tools/debug_aliases.py \
    --term \"pneumopathie inflammatoire\" --term \"hypothyroïdie\"
"""

from __future__ import annotations

import argparse

from pipeline import TermGraph


def debug_aliases(fr_term: str, graph: TermGraph) -> list[tuple[str, str]]:
    """Return the full SOC-rooted ancestor chain as (concept_id, en_label), broad → narrow."""
    fr = (fr_term or "").strip()
    if not fr:
        return []
    concept = graph.ground(fr, context=None)
    if not concept:
        return []
    cid = str(concept.get("id") or "").strip()
    if not cid:
        return [(str(concept.get("id") or ""), str(concept.get("name") or ""))]
    h = graph.fetch_hierarchy_for_concept(cid)
    chain = list(h.get("chain") or [])
    return [(str(pl.get("id") or ""), str(pl.get("name") or "")) for pl in chain]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", action="append", default=[], help="French term to ground (repeatable).")
    args = ap.parse_args()

    graph = TermGraph()
    for t in args.term:
        chain = debug_aliases(t, graph)
        print(f"\n## {t}")
        if not chain:
            print("(no grounding)")
            continue
        for cid, name in chain:
            print(f"- {cid} -> {name}")

    # Special check requested in analysis: is 'pneumonitis' in ancestor labels for pneumopathie inflammatoire?
    if any((t or "").strip().casefold() == "pneumopathie inflammatoire" for t in args.term):
        chain = debug_aliases("pneumopathie inflammatoire", graph)
        flat = " | ".join((name or "").casefold() for _cid, name in chain)
        if "pneumonitis" not in flat:
            print("\nGRAPH GAP: ancestor labels do not cover professional translation")
        else:
            print("\nOK: pneumonitis appears in ancestor chain")


if __name__ == "__main__":
    main()

