"""Hierarchy-aware terminology match (HTM).

NER-anchored audit: each French ``terms[].word`` is grounded with
``TermGraph.ground``; English MedDRA-aligned renderings are checked against
either the system **hyp** (:func:`compute_htm`) or the segment **en_ref**
(:func:`compute_htm_en_ref`), with hierarchy consistency (same branch / level
vs the grounded concept). :func:`compute_htm_hyp_vs_ref` compares HTM-style
scores on ``hyp`` vs ``en_ref`` per grounded span (mean agreement).
"""

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
from pipeline.metrics.matching import phrase_in_text

_sentence_encoder_cache: dict[str, Any] = {}


def parse_cosine_thresholds_csv(s: str) -> list[float]:
    """Parse comma-separated cosine thresholds in ``[0, 1]`` (e.g. ``0.8,0.9``)."""
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        v = float(p)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"HTM vector threshold must be in [0, 1]: {p!r}")
        out.append(v)
    return out


def htm_vector_column_key(similarity_threshold: float) -> str:
    """Stable CSV / row dict key for vector HTM at ``similarity_threshold`` (e.g. ``0.8`` → ``htm_vector_080``)."""
    p = int(round(float(similarity_threshold) * 100))
    if not 0 <= p <= 100:
        raise ValueError(f"threshold out of range after percent rounding: {similarity_threshold!r}")
    return f"htm_vector_{p:03d}"


def _ground_cached(
    graph: TermGraph,
    cache: dict[tuple[str, str | None], dict[str, Any] | None],
    word: str,
    context: str | None,
) -> dict[str, Any] | None:
    key = (word.casefold(), context)
    if key in cache:
        return cache[key]
    concept = graph.ground(word, context=context)
    cache[key] = concept
    return concept


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


def _ref_from_concept(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        "en_label": str(concept.get("name") or ""),
        "en_aliases": [],
        "level": concept.get("level"),
        "tier": concept.get("tier"),
    }


def _rendering_strings(graph: TermGraph, concept: dict[str, Any]) -> list[str]:
    out: list[str] = []
    name = str(concept.get("name") or "").strip()
    if name:
        out.append(name)
    try:
        for c in graph.candidate_renderings(concept):
            s = str(c).strip()
            if s and s not in out:
                out.append(s)
    except (TypeError, AttributeError, ValueError):
        pass
    return out


def _score_htm_alignment(found_render: str | None, ref: dict[str, Any], graph: TermGraph) -> float:
    if not found_render:
        return 0.0
    node = graph.get_by_name(ref["en_label"]) or graph.get_by_name(found_render)
    if not node:
        return 0.0
    gl = ref.get("level")
    if gl is not None and node.get("level") == gl:
        return 1.0
    if graph.same_branch(found_render, ref["en_label"]):
        return 0.5
    return 0.0


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


def _scores_htm_segment_terms_on_english(
    fr_ctx: str | None,
    terms: list[Any],
    english_surface: str,
    graph: TermGraph,
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None],
) -> list[float]:
    """One HTM-style score in ``[0, 1]`` per unique ``terms[].word`` against ``english_surface``."""
    scores: list[float] = []
    seen: set[str] = set()
    for t in terms or []:
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
        ref = _ref_from_concept(concept)
        found: str | None = None
        for r in _rendering_strings(graph, concept):
            if phrase_in_text(english_surface, r):
                found = r
                break
        scores.append(_score_htm_alignment(found, ref, graph))
    return scores


def compute_htm(
    results: list[dict[str, Any]],
    graph: TermGraph,
    id_to_segment: dict[str, dict[str, Any]],
) -> float:
    """Mean HTM in [0, 1] over French NER spans: ground each ``word``, match renderings in ``hyp``."""
    scores: list[float] = []
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    for res in results:
        sid = res.get("id")
        hyp = res.get("hyp") or ""
        seg = id_to_segment.get(str(sid)) if sid is not None else None
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None
        scores.extend(
            _scores_htm_segment_terms_on_english(
                fr_ctx, seg.get("terms") or [], hyp, graph, ground_cache
            )
        )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_htm_en_ref(
    segment_rows: list[dict[str, Any]],
    graph: TermGraph,
) -> float:
    """Mean HTM-style score over ``terms[]``, checking renderings in ``en_ref`` (not ``hyp``).

    Same aggregation and hierarchy weighting as :func:`compute_htm`, but the English
    surface is the human reference for each segment. Dataset-level only (independent
    of system output); useful as a ceiling for how often MedDRA-aligned English strings
    literally appear in the gold translation.
    """
    scores: list[float] = []
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    for row in segment_rows:
        fr_ctx = (row.get("fr") or "").strip() or None
        en_ref = row.get("en_ref") or ""
        scores.extend(
            _scores_htm_segment_terms_on_english(
                fr_ctx, row.get("terms") or [], str(en_ref), graph, ground_cache
            )
        )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_htm_hyp_vs_ref(
    results: list[dict[str, Any]],
    graph: TermGraph,
    id_to_segment: dict[str, dict[str, Any]],
) -> float:
    """Mean agreement between hypothesis and reference on ontology-aligned HTM scores.

    For each French ``terms[].word`` that **grounds** in Neo4j, compute the same
    HTM-style score (1.0 / 0.5 / 0.0) via :func:`_score_htm_alignment` against
    ``hyp`` and against segment ``en_ref`` (pipeline JSONL rows do not carry
    ``terms[]`` — pass ``id_to_segment`` like :func:`compute_htm`).

    Per span: ``agreement = 1.0 - abs(hyp_score - ref_score)``. Returns the
    mean over all such grounded spans; ``float('nan')`` if there are none.
    """
    agreements: list[float] = []
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    for res in results:
        sid = res.get("id")
        seg = id_to_segment.get(str(sid)) if sid is not None else None
        if not seg:
            continue
        hyp = res.get("hyp") or ""
        en_ref = str(seg.get("en_ref") or "")
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
                continue
            ref = _ref_from_concept(concept)
            found_hyp: str | None = None
            found_ref: str | None = None
            for r in _rendering_strings(graph, concept):
                if found_hyp is None and phrase_in_text(hyp, r):
                    found_hyp = r
                if found_ref is None and phrase_in_text(en_ref, r):
                    found_ref = r
                if found_hyp is not None and found_ref is not None:
                    break
            hyp_score = _score_htm_alignment(found_hyp, ref, graph)
            ref_score = _score_htm_alignment(found_ref, ref, graph)
            agreements.append(1.0 - abs(hyp_score - ref_score))
    if not agreements:
        return float("nan")
    return sum(agreements) / len(agreements)


def compute_htm_vector(
    results: list[dict[str, Any]],
    graph: TermGraph,
    id_to_segment: dict[str, dict[str, Any]],
    *,
    similarity_threshold: float,
    embed_model_name: str | None = None,
) -> float:
    """Like :func:`compute_htm`, but detect rendering via cosine ≥ threshold on hypothesis chunks."""
    thr = float(similarity_threshold)
    if thr < 0 or thr > 1:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {thr}")

    model = _get_sentence_encoder(embed_model_name or os.environ.get("TERMPLAN_EMBED_MODEL"))
    scores: list[float] = []
    ground_cache: dict[tuple[str, str | None], dict[str, Any] | None] = {}
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
            ref = _ref_from_concept(concept)
            renderings = [r for r in _rendering_strings(graph, concept) if str(r).strip()]
            if not renderings:
                scores.append(0.0)
                continue
            chunks = _hypothesis_sentence_splits(hyp)
            emb_chunks = model.encode(chunks, convert_to_numpy=True)
            if emb_chunks.ndim == 1:
                emb_chunks = emb_chunks.reshape(1, -1)
            emb_chunks_n = _l2_normalize_rows(emb_chunks)
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
            scores.append(_score_htm_alignment(found_render, ref, graph))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
