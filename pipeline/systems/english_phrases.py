"""Collect unique English MedDRA strings (locks + graph) for one segment."""

from __future__ import annotations

from typing import Any

from pipeline.graph import TermGraph


def collect_english_phrases(
    graph: TermGraph,
    seg: dict[str, Any],
    locks: dict[str, str] | None,
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for term in seg.get("terms") or []:
        w = (term.get("word") or "").strip()
        if not w:
            continue
        ctx = (seg.get("fr") or "").strip() or None
        concept = graph.ground(w, context=ctx)
        if not concept:
            continue
        en = (locks or {}).get(w) or concept["name"]
        if en and en not in seen:
            seen.add(en)
            phrases.append(en)
    return phrases
