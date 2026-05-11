"""HTM alias expansion tiers.

This module extends the *matching* stage of HTM with alias sets derived from the
MedDRA hierarchy in Neo4j. Scoring (1.0 exact level / 0.5 same branch / 0.0 else)
is kept consistent with :mod:`pipeline.metrics.htm`.

Alias tiers:
- T1: grounded concept name only (baseline matching).
- T2: T1 + immediate parent label (one BROADER_THAN hop up).
- T3: T2 + all ancestor labels in the same SOC-rooted chain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pipeline.graph import TermGraph
from pipeline.metrics.htm import _ground_cached, _ref_from_concept
from pipeline.metrics.matching import phrase_in_text

AliasTier = Literal["T1", "T2", "T3"]


def _unique_preserve_order(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        s = (x or "").strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def alias_strings_for_concept(graph: TermGraph, concept: dict[str, Any], tier: AliasTier) -> list[str]:
    """Return alias strings for a grounded concept at the specified tier."""
    name = str(concept.get("name") or "").strip()
    cid = str(concept.get("id") or "").strip()
    if not name:
        return []
    if not cid:
        return [name]

    try:
        h = graph.fetch_hierarchy_for_concept(cid)
        chain: list[dict[str, Any]] = list(h.get("chain") or [])
    except Exception:
        chain = []

    # Chain is broad → narrow. Ensure we can find the anchor node.
    anchor_idx: int | None = None
    for i, pl in enumerate(chain):
        if str(pl.get("id") or "").strip() == cid:
            anchor_idx = i
            break

    parent_name = ""
    if anchor_idx is not None and anchor_idx > 0:
        parent_name = str(chain[anchor_idx - 1].get("name") or "").strip()

    if tier == "T1":
        return [name]
    if tier == "T2":
        return _unique_preserve_order([name, parent_name])
    # T3: all ancestors + self (whole chain).
    all_names = [str(pl.get("name") or "").strip() for pl in chain] if chain else [name]
    return _unique_preserve_order([name, parent_name] + all_names)


def _score_htm_alignment_alias(found_render: str | None, ref: dict[str, Any], graph: TermGraph) -> float:
    """HTM-style 1.0/0.5/0.0 scoring, grounded on the *matched* alias node.

    This ensures T2/T3 alias hits can affect scoring, rather than always scoring
    as if the exact ref node was matched.
    """
    if not found_render:
        return 0.0
    ref_label = str(ref.get("en_label") or "").strip()
    if not ref_label:
        return 0.0

    node = graph.get_by_name(found_render)
    if not node:
        return 0.0

    gl = ref.get("level")
    if gl is not None and node.get("level") == gl:
        return 1.0
    if graph.same_branch(found_render, ref_label):
        return 0.5
    return 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def compute_htm_a(
    tier: AliasTier,
    *,
    segments_jsonl: Path,
    results_jsonl: Path | None,
    graph: TermGraph,
) -> float:
    """Compute HTM score using alias-tier matching.

    - If ``results_jsonl`` is provided, uses each result row's ``hyp`` as the English surface.
    - If ``results_jsonl`` is None, treats each segment row's ``en_ref`` as the hypothesis
      (rHTM-A / dataset ceiling style).
    """
    seg_rows = _load_jsonl(segments_jsonl)
    id_to_segment = {str(r.get("id")): r for r in seg_rows if r.get("id") is not None}

    if results_jsonl is None:
        results = [{"id": r.get("id"), "hyp": r.get("en_ref") or ""} for r in seg_rows]
    else:
        results = _load_jsonl(results_jsonl)

    scores: list[float] = []
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    alias_cache: dict[tuple[str, str], list[str]] = {}

    for res in results:
        sid = res.get("id")
        hyp = res.get("hyp") or ""
        seg = id_to_segment.get(str(sid)) if sid is not None else None
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None

        seen: set[str] = set()
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = w.casefold()
            if key in seen:
                continue
            seen.add(key)

            concept = _ground_cached(graph, ground_cache, w, fr_ctx)
            if not concept:
                scores.append(0.0)
                continue

            cid = str(concept.get("id") or "").strip()
            cache_key = (tier, cid or str(concept.get("name") or ""))
            if cache_key not in alias_cache:
                alias_cache[cache_key] = alias_strings_for_concept(graph, concept, tier)
            aliases = alias_cache[cache_key]

            found: str | None = None
            for a in aliases:
                if phrase_in_text(hyp, a):
                    found = a
                    break
            ref = _ref_from_concept(concept)
            scores.append(_score_htm_alignment_alias(found, ref, graph))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))

