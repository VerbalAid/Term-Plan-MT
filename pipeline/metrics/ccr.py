"""Concept coverage rate (CCR): share of NER spans that ``graph.ground`` resolves."""

from __future__ import annotations

from typing import Any


def compute_ccr_stats(
    segments: list[dict[str, Any]], graph
) -> tuple[float, int, int]:
    """``(ccr, n_spans, n_grounded)`` over non-empty NER words."""
    extracted = 0
    grounded = 0
    for seg in segments:
        ctx = (seg.get("fr") or "").strip() or None
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            extracted += 1
            if graph.ground(w, context=ctx):
                grounded += 1
    rate = grounded / extracted if extracted else 0.0
    return rate, extracted, grounded


def compute_ccr(segments: list[dict[str, Any]], graph) -> float:
    """Scalar CCR only (same denominator as :func:`compute_ccr_stats`)."""
    rate, _, _ = compute_ccr_stats(segments, graph)
    return rate
