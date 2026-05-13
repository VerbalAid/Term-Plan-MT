"""TermPlanMT evaluation metrics.

All functions take plain Python lists and dicts — no custom classes required.

Sections:
  1. Text normalisation helpers
  2. Result-file loading and alignment
  3. Fluency metrics: BLEU, chrF, COMET
  4. Hierarchy-aware terminology match (HTM)
  5. Concept coverage rate (CCR)
  6. Error analysis helpers (missing terms, drift)
  7. Top-level evaluate() and print_table()
  8. Evaluation manifest: which files map to which systems
  9. Evaluation table: collect rows and write scores_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

# ── 1. Text normalisation ──────────────────────────────────────────────────


def _norm(s: str) -> str:
    """Simple NFKC casefold + whitespace collapse (used internally by phrase_in)."""
    s = unicodedata.normalize("NFKC", (s or "").strip()).lower()
    for ch in "‐‑‒–—":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def normalize_text(s: str) -> str:
    """Normalise text for terminology matching: unicode hyphens → space, casefold."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    for ch in "‐‑‒–—−":
        s = s.replace(ch, "-")
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s.strip())


def phrase_in(text: str, phrase: str) -> bool:
    """True when ``phrase`` appears (normalised) anywhere in ``text``."""
    return bool(phrase) and _norm(phrase) in _norm(text)


def phrase_in_text(text: str, phrase: str) -> bool:
    """True when ``phrase`` appears in ``text`` using hyphen-aware normalisation."""
    if not phrase:
        return False
    return normalize_text(phrase) in normalize_text(text)


def phrase_in_hyp(hyp: str, phrase: str) -> bool:
    """Alias for :func:`phrase_in_text` (hypothesis string variant)."""
    return phrase_in_text(hyp, phrase)


# ── 2. Result loading and alignment ───────────────────────────────────────

# Hypothesis value written for segments excluded from scoring (e.g. Table 2 oracle row).
CONTAMINATION_PLACEHOLDER_HYP = "__CONTAMINATED__"


def fluency_hypothesis_text(hyp: object) -> str:
    """Return the hypothesis string for BLEU/chrF, treating contaminated rows as empty."""
    if hyp is None:
        return ""
    s = str(hyp)
    return "" if s == CONTAMINATION_PLACEHOLDER_HYP else s


def load_results_jsonl(path: Path, partial: bool = False) -> list[dict[str, Any]]:
    """Load a results ``.jsonl`` file; return an empty list when the file is missing."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if partial:
                        continue
                    raise
    return rows


def align_hyp_ref_by_doc(
    results: list[dict],
    id_to_ref: dict[str, str],
    id_to_doc: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Sort results by id, drop rows without a reference; return hyps, refs, doc keys."""
    hyps, refs, doc_keys = [], [], []
    for r in sorted(results, key=lambda x: x["id"]):
        rid = str(r["id"])
        if rid not in id_to_ref:
            continue
        hyps.append(fluency_hypothesis_text(r.get("hyp", "")))
        refs.append(id_to_ref[rid])
        doc_keys.append(id_to_doc.get(rid, rid.split("_", 1)[0] if "_" in rid else rid))
    return hyps, refs, doc_keys


def align_src_hyp_ref(
    results: list[dict],
    id_to_ref: dict[str, str],
    id_to_src: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Like :func:`align_hyp_ref_by_doc` but also returns source strings (needed for COMET)."""
    srcs, hyps, refs = [], [], []
    for r in sorted(results, key=lambda x: x["id"]):
        rid = str(r["id"])
        if rid not in id_to_ref or rid not in id_to_src:
            continue
        srcs.append(id_to_src[rid])
        hyps.append(fluency_hypothesis_text(r.get("hyp", "")))
        refs.append(id_to_ref[rid])
    return srcs, hyps, refs


# ── 3. Fluency metrics ─────────────────────────────────────────────────────


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    """Corpus BLEU via sacreBLEU."""
    return BLEU().corpus_score(hyps, [refs]).score


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    """Corpus chrF via sacreBLEU."""
    return CHRF().corpus_score(hyps, [refs]).score


def doc_bleu(hyps: list[str], refs: list[str], groups: list[str]) -> float:
    """Macro-mean corpus BLEU across document groups (one score per document, then average)."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    scores = [corpus_bleu([hyps[j] for j in idxs], [refs[j] for j in idxs])
              for idxs in idx.values()]
    return sum(scores) / len(scores) if scores else float("nan")


def macro_corpus_metric_by_group(
    corpus_fn: Callable[[list[str], list[str]], float],
    hyps: list[str],
    refs: list[str],
    groups: list[str],
) -> float:
    """Apply ``corpus_fn`` to each document group separately, then average the scores."""
    if not hyps or len(hyps) != len(refs) or len(hyps) != len(groups):
        return float("nan")
    idx_by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by[str(g)].append(i)
    scores = [corpus_fn([hyps[j] for j in ix], [refs[j] for j in ix])
              for ix in idx_by.values()]
    return sum(scores) / len(scores) if scores else float("nan")


def macro_bleu_doc_concat(
    hyps: list[str],
    refs: list[str],
    groups: list[str],
    *,
    sep: str = " ",
) -> float:
    """Macro BLEU where each document is concatenated into one synthetic sentence pair.

    This avoids brevity-penalty inflation on very short individual segments.
    """
    if not hyps or len(hyps) != len(refs) or len(hyps) != len(groups):
        return float("nan")
    idx_by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by[str(g)].append(i)
    scores = []
    for indices in idx_by.values():
        doc_h = sep.join(str(hyps[j] or "") for j in indices)
        doc_r = sep.join(str(refs[j] or "") for j in indices)
        scores.append(corpus_bleu([doc_h], [doc_r]))
    return sum(scores) / len(scores) if scores else float("nan")


_comet_model = None


def _get_comet_model():
    global _comet_model
    if _comet_model is None:
        try:
            from comet import download_model, load_from_checkpoint
            _comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        except Exception:
            pass
    return _comet_model


def corpus_comet(srcs: list[str], hyps: list[str], refs: list[str]) -> float | None:
    """COMET-DA system score (wmt22-comet-da). Returns None if comet is not installed."""
    if not srcs or len(srcs) != len(hyps) or len(hyps) != len(refs):
        return None
    try:
        model = _get_comet_model()
        if model is None:
            return None
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        result = model.predict(data, batch_size=16)
        return float(result.system_score)
    except Exception:
        return None


def corpus_comet_da(srcs: list[str], hyps: list[str], refs: list[str]) -> float | None:
    """Alias for :func:`corpus_comet` (backward-compatible name)."""
    return corpus_comet(srcs, hyps, refs)


def inference_mean_p95(rows: list[dict]) -> tuple[float | None, float | None]:
    """Mean and 95th-percentile wall-clock inference time from ``inference_s`` fields."""
    vals = [float(r["inference_s"]) for r in rows if isinstance(r.get("inference_s"), (int, float))]
    if not vals:
        return None, None
    vals.sort()
    mean = statistics.mean(vals)
    idx = min(len(vals) - 1, max(0, math.ceil(0.95 * len(vals)) - 1))
    return mean, vals[idx]


# ── 4. HTM — hierarchy-aware terminology match ────────────────────────────
# HTM measures whether the system hypothesis contains a MedDRA-aligned English
# rendering of each French NER span.  Score per span: 1.0 (exact level match),
# 0.5 (same branch), 0.0 (miss or ungrounded).

_sentence_encoder_cache: dict[str, Any] = {}


def _get_sentence_encoder(model_name: str | None = None) -> Any:
    """Return a cached SentenceTransformer for vector HTM."""
    from pipeline import DEFAULT_EMBED_MODEL
    name = (model_name or os.environ.get("TERMPLAN_EMBED_MODEL") or DEFAULT_EMBED_MODEL).strip()
    if name not in _sentence_encoder_cache:
        from sentence_transformers import SentenceTransformer
        dev = os.environ.get("TERMPLAN_EMBED_DEVICE", "").strip()
        _sentence_encoder_cache[name] = (
            SentenceTransformer(name, device=dev) if dev else SentenceTransformer(name)
        )
    return _sentence_encoder_cache[name]


def parse_cosine_thresholds_csv(s: str) -> list[float]:
    """Parse a comma-separated list of cosine thresholds, e.g. ``"0.8,0.9"``."""
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        v = float(p)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"HTM vector threshold must be in [0, 1]: {p!r}")
        out.append(v)
    return out


def htm_vector_column_key(similarity_threshold: float) -> str:
    """Stable column name for a vector HTM at a given threshold, e.g. ``htm_vector_080``."""
    p = int(round(float(similarity_threshold) * 100))
    if not 0 <= p <= 100:
        raise ValueError(f"Threshold out of range: {similarity_threshold!r}")
    return f"htm_vector_{p:03d}"


def _ground_cached(
    graph: Any,
    cache: dict[tuple[str, str | None], dict | None],
    word: str,
    context: str | None,
) -> dict | None:
    """Ground ``word`` with ``graph``, using ``cache`` to avoid repeated Neo4j calls."""
    key = (word.casefold(), context)
    if key not in cache:
        cache[key] = graph.ground(word, context=context)
    return cache[key]


def _ref_from_concept(concept: dict) -> dict:
    """Convert a raw graph concept payload to the ref dict used by scoring functions."""
    return {
        "en_label": str(concept.get("name") or ""),
        "level": concept.get("level"),
        "tier": concept.get("tier"),
    }


def _rendering_strings(graph: Any, concept: dict) -> list[str]:
    """Collect English surface strings for a concept: canonical name + graph renderings."""
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


def _score_htm_alignment(found_render: str | None, ref: dict, graph: Any) -> float:
    """HTM score for one span: 1.0 / 0.5 / 0.0."""
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


def _scores_htm_on_english(
    fr_ctx: str | None,
    terms: list[Any],
    english_surface: str,
    graph: Any,
    cache: dict,
) -> list[float]:
    """HTM scores for each unique NER term against ``english_surface``."""
    scores: list[float] = []
    seen: set[str] = set()
    for t in terms or []:
        w = (t.get("word") or "").strip()
        if not w or w.casefold() in seen:
            continue
        seen.add(w.casefold())
        concept = _ground_cached(graph, cache, w, fr_ctx)
        if not concept:
            scores.append(0.0)
            continue
        ref = _ref_from_concept(concept)
        renderings = _rendering_strings(graph, concept)
        found = next((r for r in renderings if phrase_in_text(english_surface, r)), None)
        scores.append(_score_htm_alignment(found, ref, graph))
    return scores


# Internal scoring helpers used by compute_htm and compute_hra.

def _score(found: str | None, concept: dict, graph: Any) -> float:
    if not found:
        return 0.0
    node = graph.get_by_name(str(concept.get("name") or "")) or graph.get_by_name(found)
    if not node:
        return 0.0
    if node.get("level") == concept.get("level"):
        return 1.0
    if graph.same_branch(found, str(concept.get("name") or "")):
        return 0.5
    return 0.0


def _renderings(concept: dict, graph: Any) -> list[str]:
    out = [str(concept.get("name") or "")]
    try:
        for c in graph.candidate_renderings(concept):
            s = str(c).strip()
            if s and s not in out:
                out.append(s)
    except (TypeError, AttributeError):
        pass
    return [r for r in out if r.strip()]


def compute_htm(results: list[dict], graph: Any, id_to_seg: dict[str, dict]) -> float:
    """Mean HTM score across all NER spans in the system hypotheses.

    For each French NER term, the graph returns a MedDRA concept; the system
    hypothesis is checked for any English rendering of that concept.
    """
    scores: list[float] = []
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None
        hyp = res.get("hyp") or ""
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = (w, fr_ctx)
            if key not in cache:
                cache[key] = graph.ground(w, context=fr_ctx)
            concept = cache[key]
            if not concept:
                continue
            found = next((r for r in _renderings(concept, graph) if phrase_in(hyp, r)), None)
            scores.append(_score(found, concept, graph))
    return sum(scores) / len(scores) if scores else 0.0


def compute_htm_en_ref(segment_rows: list[dict], graph: Any) -> float:
    """HTM on the human reference translations — a dataset-level ceiling metric."""
    scores: list[float] = []
    cache: dict = {}
    for row in segment_rows:
        fr_ctx = (row.get("fr") or "").strip() or None
        en_ref = str(row.get("en_ref") or "")
        scores.extend(_scores_htm_on_english(fr_ctx, row.get("terms") or [], en_ref, graph, cache))
    return sum(scores) / len(scores) if scores else 0.0


def compute_htm_hyp_vs_ref(
    results: list[dict],
    graph: Any,
    id_to_seg: dict[str, dict],
) -> float:
    """Agreement between hypothesis and reference on their per-span HTM scores.

    For each grounded span: ``agreement = 1 − |hyp_score − ref_score|``.
    Returns the mean; ``nan`` when no spans are grounded.
    """
    agreements: list[float] = []
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        hyp = res.get("hyp") or ""
        en_ref = str(seg.get("en_ref") or "")
        fr_ctx = (seg.get("fr") or "").strip() or None
        seen: set[str] = set()
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w or w.casefold() in seen:
                continue
            seen.add(w.casefold())
            concept = _ground_cached(graph, cache, w, fr_ctx)
            if not concept:
                continue
            ref = _ref_from_concept(concept)
            renderings = _rendering_strings(graph, concept)
            found_hyp = next((r for r in renderings if phrase_in_text(hyp, r)), None)
            found_ref = next((r for r in renderings if phrase_in_text(en_ref, r)), None)
            h_score = _score_htm_alignment(found_hyp, ref, graph)
            r_score = _score_htm_alignment(found_ref, ref, graph)
            agreements.append(1.0 - abs(h_score - r_score))
    return sum(agreements) / len(agreements) if agreements else float("nan")


def _hypothesis_sentence_splits(text: str) -> list[str]:
    """Split a hypothesis into sentence-like chunks for vector HTM."""
    t = (text or "").strip()
    if not t:
        return [" "]
    parts = re.split(r"\n+|(?<=[.!?;:])\s+", t)
    chunks = [p.strip() for p in parts if p.strip()]
    base = chunks if chunks else [t]
    return base if t in base else base + [t]


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / n


def compute_htm_vector(
    results: list[dict],
    graph: Any,
    id_to_seg: dict[str, dict],
    *,
    similarity_threshold: float,
    embed_model_name: str | None = None,
) -> float:
    """HTM variant that detects renderings via cosine similarity ≥ ``similarity_threshold``.

    Uses sentence-transformer embeddings rather than substring matching, so it can
    identify paraphrastic matches (e.g. "hepatic failure" ≈ "liver failure").
    """
    thr = float(similarity_threshold)
    if not 0.0 <= thr <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {thr}")

    model = _get_sentence_encoder(embed_model_name or os.environ.get("TERMPLAN_EMBED_MODEL"))
    scores: list[float] = []
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        hyp = res.get("hyp") or ""
        fr_ctx = (seg.get("fr") or "").strip() or None
        seen: set[str] = set()
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w or w.casefold() in seen:
                continue
            seen.add(w.casefold())
            concept = _ground_cached(graph, cache, w, fr_ctx)
            if not concept:
                scores.append(0.0)
                continue
            ref = _ref_from_concept(concept)
            renderings = [r for r in _rendering_strings(graph, concept) if r.strip()]
            if not renderings:
                scores.append(0.0)
                continue
            chunks = _hypothesis_sentence_splits(hyp)
            emb_chunks = model.encode(chunks, convert_to_numpy=True)
            if emb_chunks.ndim == 1:
                emb_chunks = emb_chunks.reshape(1, -1)
            emb_r = model.encode(renderings, convert_to_numpy=True)
            if emb_r.ndim == 1:
                emb_r = emb_r.reshape(1, -1)
            sims = _l2_normalize_rows(emb_chunks) @ _l2_normalize_rows(emb_r).T
            best_sim = float(sims.max())
            if best_sim < thr:
                scores.append(0.0)
                continue
            jj = int(np.argmax(sims)) % sims.shape[1]
            scores.append(_score_htm_alignment(renderings[jj], ref, graph))
    return sum(scores) / len(scores) if scores else 0.0


def compute_rhtm(segments: list[dict], graph: Any) -> float:
    """HTM evaluated on the human reference — measures ontology coverage in gold data."""
    results = [{"id": s["id"], "hyp": s.get("en_ref") or ""} for s in segments]
    id_to_seg = {str(s["id"]): s for s in segments}
    return compute_htm(results, graph, id_to_seg)


def compute_hra(results: list[dict], graph: Any, id_to_seg: dict[str, dict]) -> float:
    """Fraction of grounded spans where hypothesis and reference agree on their HTM score."""
    agreements: list[float] = []
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None
        hyp = res.get("hyp") or ""
        ref = str(seg.get("en_ref") or "")
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = (w, fr_ctx)
            if key not in cache:
                cache[key] = graph.ground(w, context=fr_ctx)
            concept = cache[key]
            if not concept:
                continue
            rends = _renderings(concept, graph)
            h_found = next((r for r in rends if phrase_in(hyp, r)), None)
            r_found = next((r for r in rends if phrase_in(ref, r)), None)
            agreements.append(1.0 - abs(_score(h_found, concept, graph) - _score(r_found, concept, graph)))
    return sum(agreements) / len(agreements) if agreements else float("nan")


# ── 5. CCR — concept coverage rate ────────────────────────────────────────


def compute_ccr(segments: list[dict], graph: Any) -> float:
    """Fraction of NER spans that ``graph.ground`` resolves to a MedDRA concept."""
    total = grounded = 0
    for seg in segments:
        fr_ctx = (seg.get("fr") or "").strip() or None
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            total += 1
            if graph.ground(w, context=fr_ctx):
                grounded += 1
    return grounded / total if total else 0.0


def compute_ccr_stats(
    segments: list[dict], graph: Any
) -> tuple[float, int, int]:
    """CCR with raw counts: returns ``(rate, n_spans, n_grounded)``."""
    total = grounded = 0
    for seg in segments:
        fr_ctx = (seg.get("fr") or "").strip() or None
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            total += 1
            if graph.ground(w, context=fr_ctx):
                grounded += 1
    return (grounded / total if total else 0.0), total, grounded


# ── 6. Error analysis helpers ──────────────────────────────────────────────


def missing_term_rate(results: list[dict], graph: Any, id_to_seg: dict[str, dict]) -> float:
    """Fraction of grounded NER spans whose English rendering is absent from the hypothesis."""
    total = missing = 0
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None
        hyp = res.get("hyp") or ""
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = (w, fr_ctx)
            if key not in cache:
                cache[key] = graph.ground(w, context=fr_ctx)
            concept = cache[key]
            if not concept:
                continue
            total += 1
            if not any(phrase_in(hyp, r) for r in _renderings(concept, graph)):
                missing += 1
    return missing / total if total else float("nan")


def term_drift(
    results: list[dict],
    graph: Any,
    id_to_seg: dict[str, dict],
    min_segs: int = 3,
) -> dict[str, float]:
    """Consistency score per French NER term: 0 = always the same rendering, 1 = always different."""
    term_hyps: dict[str, list[str]] = defaultdict(list)
    cache: dict = {}
    for res in results:
        seg = id_to_seg.get(str(res.get("id")))
        if not seg:
            continue
        fr_ctx = (seg.get("fr") or "").strip() or None
        hyp = res.get("hyp") or ""
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if not w:
                continue
            key = (w, fr_ctx)
            if key not in cache:
                cache[key] = graph.ground(w, context=fr_ctx)
            concept = cache[key]
            if not concept:
                continue
            found = next((r for r in _renderings(concept, graph) if phrase_in(hyp, r)), "__NONE__")
            term_hyps[w].append(_norm(found))

    out: dict[str, float] = {}
    for fr_term, sigs in term_hyps.items():
        if len(sigs) < min_segs:
            continue
        distinct = len(set(sigs))
        out[fr_term] = (distinct - 1) / max(len(sigs) - 1, 1)
    return out


# ── 7. Full evaluation ─────────────────────────────────────────────────────


def evaluate(
    results_dir: Path,
    segments_path: Path,
    graph: Any,
    *,
    with_comet: bool = False,
    exclude_ids: frozenset[str] | None = None,
) -> dict[str, dict]:
    """Score every ``s*.jsonl`` file in ``results_dir``; return a dict of system → metrics."""
    segments = [json.loads(ln)
                for ln in segments_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if exclude_ids:
        segments = [s for s in segments if s["id"] not in exclude_ids]

    id_to_seg = {str(s["id"]): s for s in segments}
    id_to_ref = {str(s["id"]): s["en_ref"] for s in segments}
    id_to_src = {str(s["id"]): s["fr"] for s in segments}
    id_to_doc = {str(s["id"]): str(s["id"]).split("_")[0] for s in segments}

    ccr  = compute_ccr(segments, graph)
    rhtm = compute_rhtm(segments, graph)

    out: dict[str, dict] = {}
    for fn in sorted(results_dir.glob("s*.jsonl")):
        rows = [json.loads(ln) for ln in fn.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if exclude_ids:
            rows = [r for r in rows if r.get("id") not in exclude_ids]
        if not rows:
            continue
        label = fn.stem
        ordered = sorted(rows, key=lambda x: x["id"])
        hyps   = [r.get("hyp", "") for r in ordered if str(r["id"]) in id_to_ref]
        refs   = [id_to_ref[str(r["id"])] for r in ordered if str(r["id"]) in id_to_ref]
        groups = [id_to_doc[str(r["id"])] for r in ordered if str(r["id"]) in id_to_doc]
        if len(hyps) != len(refs):
            continue
        entry: dict[str, Any] = {
            "bleu":      corpus_bleu(hyps, refs),
            "chrf":      corpus_chrf(hyps, refs),
            "doc_bleu":  doc_bleu(hyps, refs, groups),
            "htm":       compute_htm(rows, graph, id_to_seg),
            "hra":       compute_hra(rows, graph, id_to_seg),
            "miss_rate": missing_term_rate(rows, graph, id_to_seg),
            "ccr":       ccr,
            "rhtm":      rhtm,
        }
        if with_comet:
            srcs = [id_to_src[str(r["id"])] for r in ordered if str(r["id"]) in id_to_src]
            entry["comet"] = corpus_comet(srcs, hyps, refs)
        out[label] = entry
    return out


def print_table(scores: dict[str, dict]) -> None:
    """Print a formatted console table of evaluation scores."""
    cols = ["bleu", "chrf", "doc_bleu", "htm", "hra", "miss_rate"]
    header = f"{'System':<16}" + "".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for sys_label, m in sorted(scores.items()):
        row = f"{sys_label:<16}"
        for c in cols:
            v = m.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                row += f"{'—':>10}"
            else:
                row += f"{v:>10.3f}"
        print(row)


# ── 8. Evaluation manifest ─────────────────────────────────────────────────
# Defines which output files correspond to which systems.  S6 is a glossary-
# oracle ablation: same logit-boost decoding as S5 but phrases come from a
# hand-built FR→EN glossary rather than MedDRA Neo4j.

EVAL_FILES: list[tuple[str, str]] = [
    ("s1",        "s1.jsonl"),
    ("s2",        "s2.jsonl"),
    ("s3",        "s3.jsonl"),
    ("s4",        "s4.jsonl"),
    ("s5",        "s5.jsonl"),
    ("s5_mistral","s5_mistral.jsonl"),
    ("s6",        "s6.jsonl"),
    ("s6_mistral","s6_mistral.jsonl"),
]

# Contamination-filtered versions for Mistral graph outputs (see docs/CANONICAL_METRICS.md).
EVAL_FILES_MISTRAL_CONTAMINATION_CLEAN: list[tuple[str, str]] = [
    ("s1",        "s1.jsonl"),
    ("s2",        "s2.jsonl"),
    ("s3",        "s3_clean.jsonl"),
    ("s4",        "s4_clean.jsonl"),
    ("s5",        "s5.jsonl"),
    ("s5_mistral","s5_mistral_clean.jsonl"),
    ("s6",        "s6.jsonl"),
    ("s6_mistral","s6_mistral.jsonl"),
]

# Profiles for rerun_all.sh and run_eval_plot_matrix.py.
# Each entry: (results_subdir, segment_jsonl_candidates, exclude_ids_override, eval_file_set)
EVAL_RERUN_PROFILES: list[tuple] = [
    (
        "results/ner_biollm",
        ("data/section48/segments_ner_biollm.jsonl",),
        "",           # Include 48_028 (matches workshop paper tables).
        "standard",
    ),
    (
        "results/ner_biollm_finetuned",
        (
            "data/section48/segments_ner_unsloth.jsonl",
            "data/section48/segments_ner_unsloth_full.jsonl",
        ),
        "",           # Include 48_028 to match paper BLEU/chrF.
        "mistral_clean",
    ),
    ("results/ner_baseline",  ("data/section48/segments_ner.jsonl",)),
    ("results/ner_finetuned", ("data/section48/segments_ner.jsonl",)),
]


def unpack_eval_rerun_profile(item: tuple) -> tuple[str, tuple, str | None, str]:
    """Unpack an ``EVAL_RERUN_PROFILES`` entry into its four components."""
    if len(item) == 2:
        return item[0], item[1], None, "standard"
    if len(item) == 4:
        return item[0], item[1], item[2], item[3]
    raise ValueError(f"Invalid EVAL_RERUN_PROFILES entry length {len(item)}: {item!r}")


def eval_files_for_set(name: str) -> list[tuple[str, str]]:
    """Return the file list for an eval set name (``standard`` or ``mistral_clean``)."""
    if name in ("", "standard"):
        return list(EVAL_FILES)
    if name == "mistral_clean":
        return list(EVAL_FILES_MISTRAL_CONTAMINATION_CLEAN)
    raise ValueError(f"Unknown eval file set: {name!r}")


def condition_name_from_results_subdir(results_subdir: str) -> str:
    """Extract the condition name from a results path, e.g. ``results/ner_biollm`` → ``ner_biollm``."""
    p = results_subdir.strip().strip("/")
    return p[len("results/"):] if p.startswith("results/") else p


# ── 9. Evaluation table helpers ────────────────────────────────────────────

try:
    from neo4j.exceptions import ServiceUnavailable as _ServiceUnavailable
except ImportError:
    class _ServiceUnavailable(Exception):  # type: ignore[no-redef]
        pass

NEO4J_CONN_ERRORS: tuple[type[BaseException], ...] = (
    _ServiceUnavailable,
    ConnectionError,
    TimeoutError,
)


def neo4j_connection_help() -> str:
    """Return a human-readable hint when Neo4j is unreachable."""
    return (
        "Neo4j is not reachable (bolt connection refused).\n"
        "  • Start the DB:    docker compose up -d\n"
        "  • Or skip graph metrics:    --no-graph\n"
    )


DISPLAY_NAMES: dict[str, str] = {
    "s1":        "S1 NLLB",
    "s2":        "S2 Mistral (doc)",
    "s3":        "S3 GraphRAG",
    "s4":        "S4 rerank",
    "s5":        "S5 NLLB + boost",
    "s5_mistral":"S5 Mistral + boost",
    "s6":        "S6 NLLB + glossary",
    "s6_mistral":"S6 Mistral + glossary",
}


def display_label_for_system(lab: str) -> str:
    """Human-readable system name for figures and tables."""
    return DISPLAY_NAMES.get(lab, lab)


def document_key_for_segment_row(row: dict) -> str:
    """Derive the document id for macro BLEU grouping."""
    doc = row.get("document_id")
    if doc is not None and str(doc).strip():
        return str(doc).strip()
    sid = str(row["id"])
    return sid.split("_", 1)[0] if "_" in sid else sid


def collect_system_metric_rows(
    *,
    results_dir: Path,
    id_to_ref: dict[str, str],
    id_to_src: dict[str, str],
    graph: Any,
    segment_rows: list[dict],
    partial: bool,
    with_comet: bool,
    keep_segment_ids: frozenset[str] | None = None,
    htm_vector_thresholds: list[float] | None = None,
    htm_embed_model: str | None = None,
    n_expected: int | None = None,
    fill_missing: bool = False,
    out_warnings: list[str] | None = None,
    eval_file_set: str = "standard",
) -> tuple[list[dict[str, Any]], float]:
    """Build one metric dict per system; also return dataset-level CCR.

    When ``fill_missing`` is True, rows for missing or partial files are included
    with NaN values so every system in the manifest appears in the output.
    """
    htm_vec_thr = htm_vector_thresholds or []
    htm_en_ref = float("nan")
    if graph is None:
        ccr = float("nan")
    else:
        ccr = compute_ccr(segment_rows, graph)
        htm_en_ref = compute_htm_en_ref(segment_rows, graph)

    id_to_seg = {str(r["id"]): r for r in segment_rows}
    id_to_doc = {str(r["id"]): document_key_for_segment_row(r) for r in segment_rows}
    out: list[dict[str, Any]] = []

    def _nan_row(label: str) -> dict[str, Any]:
        row_out: dict[str, Any] = {
            "label": label, "display": display_label_for_system(label),
            "bleu": float("nan"), "chrf": float("nan"),
            "bleu_doc_macro": float("nan"), "bleu_doc_concat": float("nan"),
            "chrf_doc_macro": float("nan"), "comet": None,
            "htm": float("nan"), "htm_hyp_ref_agreement": float("nan"),
            "htm_en_ref_dataset": htm_en_ref, "mean_s": None, "p95_s": None,
        }
        for t in htm_vec_thr:
            row_out[htm_vector_column_key(t)] = float("nan")
        return row_out

    for label, fname in eval_files_for_set(eval_file_set):
        p = results_dir / fname
        if not p.is_file():
            if fill_missing:
                if out_warnings is not None:
                    out_warnings.append(f"WARNING: missing {p} — skipping {label}.")
                out.append(_nan_row(label))
            continue
        res = load_results_jsonl(p, partial=partial)
        if keep_segment_ids is not None:
            res = [r for r in res if r.get("id") in keep_segment_ids]
        if fill_missing and n_expected is not None and len(res) < n_expected:
            if out_warnings is not None:
                out_warnings.append(f"WARNING: {p} has {len(res)} rows (expected {n_expected}) — skipping {label}.")
            out.append(_nan_row(label))
            continue
        if not res:
            if fill_missing:
                if out_warnings is not None:
                    out_warnings.append(f"WARNING: {p} empty after id filter — skipping {label}.")
                out.append(_nan_row(label))
            continue
        hyps, refs, doc_keys = align_hyp_ref_by_doc(res, id_to_ref, id_to_doc)
        if not hyps:
            if fill_missing:
                out.append(_nan_row(label))
            continue
        b        = corpus_bleu(hyps, refs)
        c        = corpus_chrf(hyps, refs)
        b_doc    = macro_corpus_metric_by_group(corpus_bleu, hyps, refs, doc_keys)
        b_concat = macro_bleu_doc_concat(hyps, refs, doc_keys)
        c_doc    = macro_corpus_metric_by_group(corpus_chrf, hyps, refs, doc_keys)
        comet_v: float | None = None
        if with_comet:
            srcs, h2, r2 = align_src_hyp_ref(res, id_to_ref, id_to_src)
            comet_v = corpus_comet_da(srcs, h2, r2)
        if graph is None:
            htm_v = hyp_ref_ag = float("nan")
        else:
            htm_v      = compute_htm(res, graph, id_to_seg)
            hyp_ref_ag = compute_htm_hyp_vs_ref(res, graph, id_to_seg)
        mean_s, p95_s = inference_mean_p95(res)
        row_out: dict[str, Any] = {
            "label": label, "display": display_label_for_system(label),
            "bleu": b, "chrf": c,
            "bleu_doc_macro": b_doc, "bleu_doc_concat": b_concat, "chrf_doc_macro": c_doc,
            "comet": comet_v,
            "htm": htm_v, "htm_hyp_ref_agreement": hyp_ref_ag,
            "htm_en_ref_dataset": htm_en_ref,
            "mean_s": mean_s, "p95_s": p95_s,
        }
        for t in htm_vec_thr:
            k = htm_vector_column_key(t)
            if graph is None:
                row_out[k] = float("nan")
            else:
                try:
                    row_out[k] = float(compute_htm_vector(
                        res, graph, id_to_seg,
                        similarity_threshold=t, embed_model_name=htm_embed_model,
                    ))
                except Exception:
                    row_out[k] = float("nan")
        out.append(row_out)
    return out, ccr


def write_scores_summary_csv(rows: list[dict[str, Any]], ccr: float, path: Path) -> None:
    """Write the scores table to ``scores_summary.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = sorted(k for k in rows[0] if str(k).startswith("htm_vector_")) if rows else []
    fieldnames = [
        "label", "display", "bleu", "chrf", "bleu_doc_macro", "bleu_doc_concat",
        "chrf_doc_macro", "comet", "htm", "htm_hyp_ref_agreement",
        *vec, "htm_en_ref_dataset", "ccr_dataset", "mean_s", "p95_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out_row = {k: r.get(k) for k in fieldnames if k != "ccr_dataset"}
            out_row["ccr_dataset"] = ccr
            w.writerow(out_row)
