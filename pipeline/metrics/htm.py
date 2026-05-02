"""Hierarchy-aware terminology match (HTM): string or embedding overlap + MedDRA level."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pipeline.graph import DEFAULT_EMBED_MODEL, TermGraph
from pipeline.metrics.matching import all_renderings, phrase_in_hyp

_sentence_encoder_cache: dict[str, Any] = {}


def _get_sentence_encoder(model_name: str | None) -> Any:
    name = (model_name or DEFAULT_EMBED_MODEL).strip()
    if name not in _sentence_encoder_cache:
        from sentence_transformers import SentenceTransformer

        dev = os.environ.get("TERMPLAN_EMBED_DEVICE", "").strip()
        if dev:
            _sentence_encoder_cache[name] = SentenceTransformer(name, device=dev)
        else:
            _sentence_encoder_cache[name] = SentenceTransformer(name)
    return _sentence_encoder_cache[name]


def _score_htm_alignment(found_render: str | None, gold: dict[str, Any], graph: TermGraph) -> float:
    if not found_render:
        return 0.0
    node = graph.get_by_name(gold["en_label"]) or graph.get_by_name(found_render)
    if not node:
        return 0.0
    gl = gold.get("level")
    if gl is not None and node.get("level") == gl:
        return 1.0
    if graph.same_branch(found_render, gold["en_label"]):
        return 0.5
    return 0.0


def _gold_hits_in_segments(
    results: list[dict[str, Any]],
    gold_terms: list[dict[str, Any]],
):
    for res in results:
        fr_seg = res.get("fr", "")
        hyp = res.get("hyp", "")
        for gold in gold_terms:
            gfr = gold.get("fr", "")
            if not gfr or gfr.lower() not in fr_seg.lower():
                continue
            yield gold, hyp


def _hypothesis_sentence_splits(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return [" "]
    parts = re.split(r"\n+|(?<=[.!?;:])\s+", t)
    out = [p.strip() for p in parts if p.strip()]
    base = out if out else [t]
    if t not in base:
        base = base + [t]
    return base


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / n


def compute_htm(results: list[dict[str, Any]], gold_terms: list[dict[str, Any]], graph: TermGraph) -> float:
    """Mean HTM in [0,1] using substring match on English renderings."""
    scores: list[float] = []
    for gold, hyp in _gold_hits_in_segments(results, gold_terms):
        found_render = None
        for r in all_renderings(gold):
            if phrase_in_hyp(hyp, r):
                found_render = r
                break
        scores.append(_score_htm_alignment(found_render, gold, graph))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_htm_vector(
    results: list[dict[str, Any]],
    gold_terms: list[dict[str, Any]],
    graph: TermGraph,
    *,
    similarity_threshold: float,
    embed_model_name: str | None = None,
) -> float:
    """Mean HTM like :func:`compute_htm`, but detect rendering via cosine ≥ threshold on hypothesis chunks."""
    thr = float(similarity_threshold)
    if thr < 0 or thr > 1:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {thr}")

    model = _get_sentence_encoder(embed_model_name or os.environ.get("TERMPLAN_EMBED_MODEL"))
    scores: list[float] = []
    for gold, hyp in _gold_hits_in_segments(results, gold_terms):
        hyp = hyp or ""
        chunks = _hypothesis_sentence_splits(hyp)
        emb_chunks = model.encode(chunks, convert_to_numpy=True)
        if emb_chunks.ndim == 1:
            emb_chunks = emb_chunks.reshape(1, -1)
        emb_chunks_n = _l2_normalize_rows(emb_chunks)

        renderings = [r for r in all_renderings(gold) if str(r).strip()]
        if not renderings:
            scores.append(0.0)
            continue
        emb_r = model.encode(renderings, convert_to_numpy=True)
        if emb_r.ndim == 1:
            emb_r = emb_r.reshape(1, -1)
        emb_r_n = _l2_normalize_rows(emb_r)
        sims = emb_chunks_n @ emb_r_n.T
        best_sim = float(sims.max())
        if best_sim < thr:
            scores.append(0.0)
            continue
        flat = int(np.argmax(sims))
        jj = flat % sims.shape[1]
        found_render = renderings[jj]
        scores.append(_score_htm_alignment(found_render, gold, graph))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
