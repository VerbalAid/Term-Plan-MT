"""Stage 3: global French→English locks from grounded terms and embedding scores."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer, util

from pipeline.systems.data_io import iter_segments_filtered

_PLAN_MODEL: SentenceTransformer | None = None


def _embed_model(name: str | None = None) -> SentenceTransformer:
    global _PLAN_MODEL
    if _PLAN_MODEL is None:
        mid = name or os.environ.get(
            "TERMPLAN_EMBED_MODEL",
            "paraphrase-multilingual-mpnet-base-v2",
        )
        _PLAN_MODEL = SentenceTransformer(mid)
    return _PLAN_MODEL


def _plan_weights(
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
) -> tuple[float, float, float]:
    a = alpha if alpha is not None else float(os.environ.get("PLAN_ALPHA", "0.5"))
    b = beta if beta is not None else float(os.environ.get("PLAN_BETA", "0.3"))
    g = gamma if gamma is not None else float(os.environ.get("PLAN_GAMMA", "0.2"))
    return a, b, g


def _index_fr_terms(segments: list[dict[str, Any]]) -> dict[str, list[int]]:
    term_to_idxs: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            term_to_idxs.setdefault(w, []).append(i)
    return term_to_idxs


def _neighbor_bar_sim(model, util, graph, concept_id: str, cand_emb, cand: str) -> float:
    neigh = [n for n in graph.neighbours(str(concept_id)) if n and n != cand][:16]
    if not neigh:
        return 0.0
    emb_n = model.encode(neigh, convert_to_tensor=True)
    return float(util.cos_sim(cand_emb.unsqueeze(0), emb_n).mean().cpu())


def _hierarchy_penalty(graph, cand: str, src_level: Any) -> int:
    node_c = graph.get_by_name(cand)
    lvl_c = node_c.get("level") if node_c else src_level
    try:
        return abs(int(lvl_c) - int(src_level))
    except (TypeError, ValueError):
        return 0


def _best_candidate_score(
    model,
    util,
    graph,
    concept: dict[str, Any],
    segments: list[dict[str, Any]],
    idxs: list[int],
    alpha: float,
    beta: float,
    gamma: float,
) -> str | None:
    src_level = concept.get("level") or 0
    candidates = graph.candidate_renderings(concept)
    if not candidates:
        return None

    sents = [segments[j]["fr"] for j in idxs]
    emb_sents = model.encode(sents, convert_to_tensor=True)
    best_c = candidates[0]
    best_score = float("-inf")
    cid = str(concept["id"])

    for cand in candidates:
        emb_c = model.encode(cand, convert_to_tensor=True)
        ctx = float(util.cos_sim(emb_sents, emb_c.unsqueeze(0)).squeeze(-1).mean().cpu())
        nb = _neighbor_bar_sim(model, util, graph, cid, emb_c, cand)
        pen = _hierarchy_penalty(graph, cand, src_level)
        score = alpha * ctx + beta * nb - gamma * float(pen)
        if score > best_score:
            best_score = score
            best_c = cand
    return best_c


def compute_global_locks(
    segments: list[dict[str, Any]],
    graph,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    embed_model: str | None = None,
) -> dict[str, str]:
    """Map each distinct NER French surface string to one English rendering (global lock table)."""
    alpha, beta, gamma = _plan_weights(alpha, beta, gamma)
    term_to_idxs = _index_fr_terms(segments)
    model = _embed_model(embed_model)
    locks: dict[str, str] = {}

    for fr_term, idxs in term_to_idxs.items():
        concept = graph.ground(fr_term, context=(segments[idxs[0]].get("fr") or "").strip() or None)
        if not concept:
            continue
        pick = _best_candidate_score(model, util, graph, concept, segments, idxs, alpha, beta, gamma)
        if pick:
            locks[fr_term] = pick

    return locks


def load_or_compute_locks(
    segments_path: Path,
    graph,
    *,
    cache_path: Path | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    embed_model: str | None = None,
    recompute: bool = False,
    exclude_segment_ids: frozenset[str] | None = None,
) -> dict[str, str]:
    """Load `planning_locks.json` if fresh, else compute and write cache."""
    cache = cache_path or segments_path.parent / "planning_locks.json"
    if cache.is_file() and not recompute:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stored_mtime = data.get("_segments_mtime")
                locks_only = {str(k): str(v) for k, v in data.items() if k != "_segments_mtime"}
                if stored_mtime is None and not locks_only:
                    pass
                else:
                    try:
                        cur_mtime = os.path.getmtime(segments_path)
                    except OSError:
                        cur_mtime = None
                    if stored_mtime is None or cur_mtime is None:
                        logging.warning(
                            "planning_locks.json may be stale — delete it or pass recompute=True"
                        )
                    else:
                        try:
                            delta = abs(float(stored_mtime) - float(cur_mtime))
                        except (TypeError, ValueError):
                            delta = 999.0
                        if delta > 1.0:
                            logging.warning(
                                "planning_locks.json may be stale — delete it or pass recompute=True"
                            )
                    return locks_only
        except (json.JSONDecodeError, OSError):
            pass

    rows: list[dict[str, Any]] = list(iter_segments_filtered(segments_path, exclude_segment_ids))

    locks = compute_global_locks(rows, graph, alpha=alpha, beta=beta, gamma=gamma, embed_model=embed_model)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_segments_mtime": os.path.getmtime(segments_path), **locks}
        cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return locks
