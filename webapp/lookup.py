"""MedDRA term lookup: exact → fuzzy (RapidFuzz) → semantic (lazy-loaded embeddings)."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from webapp.context_llm import llm_configured, llm_model, resolve_context
from webapp.graph import MeddraGraph, norm_key

log = logging.getLogger(__name__)

MatchKind = Literal["exact", "fuzzy", "semantic", "context_llm", "none"]
Lang = Literal["fr", "en"]

_FR_HINT = re.compile(
    r"[àâäéèêëïîôùûüçœæ]|"
    r"\b(hypothyroïdie|pneumopathie|indésirable|chimiothérapie|médicament)\b",
    re.IGNORECASE,
)


def detect_lang(term: str) -> Lang:
    """Lightweight FR vs EN hint for ``lang=auto``."""
    if _FR_HINT.search(term):
        return "fr"
    return "en"


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


@dataclass
class ConceptView:
    id: str
    name: str
    level: int | None
    tier: str
    fr_label: str | None = None
    en_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "en_label": self.en_label or self.name,
            "level": self.level,
            "tier": self.tier,
            "fr_label": self.fr_label,
        }


@dataclass
class LookupResult:
    query: str
    query_lang: str
    match_type: MatchKind
    score: float | None
    ambiguous: bool
    concept: ConceptView | None
    parents: list[ConceptView] = field(default_factory=list)
    children: list[ConceptView] = field(default_factory=list)
    ancestors: list[ConceptView] = field(default_factory=list)
    alternatives: list[ConceptView] = field(default_factory=list)
    message: str | None = None
    semantic_ready: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_lang": self.query_lang,
            "match_type": self.match_type,
            "score": self.score,
            "ambiguous": self.ambiguous,
            "concept": self.concept.to_dict() if self.concept else None,
            "parents": [p.to_dict() for p in self.parents],
            "children": [c.to_dict() for c in self.children],
            "ancestors": [a.to_dict() for a in self.ancestors],
            "alternatives": [a.to_dict() for a in self.alternatives],
            "message": self.message,
            "semantic_ready": self.semantic_ready,
        }


@dataclass
class ContextLookupResult:
    context_sentence: str
    target_term: str
    query_lang: str
    baseline: LookupResult
    candidates: list[dict[str, Any]]
    llm: dict[str, Any]
    selected: LookupResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_sentence": self.context_sentence,
            "target_term": self.target_term,
            "query_lang": self.query_lang,
            "baseline": self.baseline.to_dict(),
            "candidates": self.candidates,
            "llm": self.llm,
            "selected": self.selected.to_dict() if self.selected else None,
        }


class _SemanticIndex:
    def __init__(self, model_id: str, threshold: float) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.labels: list[str] = []
        self.concepts: list[dict] = []
        self.matrix = None
        self.model = None
        self.ready = False


class MeddraLookupService:
    """Standalone lookup service (no MT pipeline dependency)."""

    def __init__(self) -> None:
        self._graph = MeddraGraph()
        self._embed_model_id = os.environ.get(
            "TERMPLAN_EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._semantic_threshold = float(os.environ.get("LOOKUP_SEMANTIC_MIN", "0.55"))
        self._fuzzy_cutoff = float(os.environ.get("GROUND_FUZZY_CUTOFF", "90"))
        self._candidate_fuzzy_cutoff = float(
            os.environ.get("CONTEXT_FUZZY_CUTOFF", str(max(70.0, self._fuzzy_cutoff - 15)))
        )
        self._candidate_limit = int(os.environ.get("CONTEXT_CANDIDATE_LIMIT", "8"))
        self._semantic: dict[str, _SemanticIndex] = {
            "fr": _SemanticIndex(self._embed_model_id, self._semantic_threshold),
            "en": _SemanticIndex(self._embed_model_id, self._semantic_threshold),
        }
        self._semantic_lock = threading.Lock()
        self._prewarm = _env_bool("PREWARM_SEMANTIC", False)

    def close(self) -> None:
        self._graph.close()

    def semantic_status(self) -> dict[str, bool]:
        return {lang: idx.ready for lang, idx in self._semantic.items()}

    def health(self) -> dict[str, Any]:
        try:
            self._graph._load()
            return {
                "status": "ok",
                "neo4j": "connected",
                "labels_loaded": self._graph.label_count(),
                "semantic_ready": self.semantic_status(),
                "embed_model": self._embed_model_id,
                "prewarm_semantic": self._prewarm,
                "llm_configured": llm_configured(),
                "llm_model": llm_model() if llm_configured() else None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "neo4j": "unavailable",
                "detail": str(exc),
                "semantic_ready": self.semantic_status(),
            }

    def prewarm_semantic(self) -> None:
        """Load embedding model + both language indexes (optional startup)."""
        if not self._prewarm:
            return
        for lang in ("fr", "en"):
            try:
                self._ensure_semantic_index(lang)
            except Exception as exc:
                log.warning("Semantic prewarm (%s) skipped: %s", lang, exc)

    def lookup(self, term: str, *, lang: str = "fr") -> LookupResult:
        raw = (term or "").strip()
        if not raw:
            return LookupResult(
                query=raw,
                query_lang=lang,
                match_type="none",
                score=None,
                ambiguous=False,
                concept=None,
                message="Enter a term to search.",
                semantic_ready=self.semantic_status(),
            )

        lang_norm = (lang or "fr").lower()
        if lang_norm == "auto":
            query_lang = detect_lang(raw)
        elif lang_norm.startswith("en"):
            query_lang = "en"
        else:
            query_lang = "fr"

        if query_lang == "en":
            result = self._lookup_english(raw, query_lang)
        else:
            result = self._lookup_french(raw, query_lang)
        result.semantic_ready = self.semantic_status()
        return result

    def collect_candidates(
        self, term: str, *, lang: str = "auto", limit: int | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Hybrid graph retrieval: exact, ambiguous alts, fuzzy, semantic (deduped)."""
        raw = (term or "").strip()
        cap = limit if limit is not None else self._candidate_limit
        lang_norm = (lang or "auto").lower()
        if lang_norm == "auto":
            query_lang = detect_lang(raw)
        elif lang_norm.startswith("en"):
            query_lang = "en"
        else:
            query_lang = "fr"

        merged: dict[str, dict[str, Any]] = {}

        def add(concept: dict, source: str, score: float | None) -> None:
            anchored = self._graph.resolve_concept(concept)
            cid = str(anchored.get("id", ""))
            if not cid:
                return
            if cid not in merged or (score or 0) > float(merged[cid].get("score") or 0):
                merged[cid] = self._enrich_candidate(anchored, source, score)

        if query_lang == "fr":
            key = norm_key(raw)
            if self._graph.is_ambiguous_fr(raw):
                for row in self._graph.alternatives_fr(key):
                    add(row, "ambiguous", 100.0)
            if exact := self._graph.exact_fr(raw):
                add(exact, "exact", 100.0)
            for concept, score in self._graph.fuzzy_fr_candidates(
                raw, self._candidate_fuzzy_cutoff, limit=5
            ):
                add(concept, "fuzzy", score)
            for concept, score in self._semantic_top_k(raw, "fr", k=5):
                add(concept, "semantic", round(score * 100, 1))
        else:
            if exact := self._graph.exact_en(raw):
                add(exact, "exact", 100.0)
            for concept, score in self._graph.fuzzy_en_candidates(
                raw, self._candidate_fuzzy_cutoff, limit=5
            ):
                add(concept, "fuzzy", score)
            for concept, score in self._semantic_top_k(raw, "en", k=5):
                add(concept, "semantic", round(score * 100, 1))

        ranked = sorted(
            merged.values(),
            key=lambda c: float(c.get("score") or 0),
            reverse=True,
        )[:cap]
        return query_lang, ranked

    def context_lookup(
        self,
        context_sentence: str,
        target_term: str,
        *,
        lang: str = "auto",
    ) -> ContextLookupResult:
        """In-context mode: graph candidates plus OpenRouter disambiguation."""
        ctx = (context_sentence or "").strip()
        term = (target_term or "").strip()
        query_lang, candidates = self.collect_candidates(term, lang=lang)
        baseline = self.lookup(term, lang=lang)
        baseline.semantic_ready = self.semantic_status()

        llm_out: dict[str, Any] = {
            "configured": llm_configured(),
            "model": llm_model() if llm_configured() else None,
        }
        selected: LookupResult | None = None

        if not candidates:
            llm_out.update(
                ok=False,
                error="no_candidates",
                message="No graph candidates for this term.",
            )
        elif llm_configured():
            llm_out.update(resolve_context(ctx, term, candidates))
            if llm_out.get("ok") and not llm_out.get("abstain"):
                cid = llm_out.get("selected_concept_id")
                concept = self._graph.concept_by_id(str(cid)) if cid else None
                if concept:
                    selected = self._result_from_concept(
                        term, query_lang, concept, "context_llm", 100.0
                    )
                    if llm_out.get("fallback"):
                        selected.message = "Resolved via graph fallback (routing unavailable)."
                    else:
                        selected.message = "Resolved from sentence context."
        else:
            llm_out.update(
                ok=False,
                error="llm_not_configured",
                message="Context routing unavailable.",
            )
            if baseline.concept and len(candidates) <= 1:
                selected = baseline

        return ContextLookupResult(
            context_sentence=ctx,
            target_term=term,
            query_lang=query_lang,
            baseline=baseline,
            candidates=candidates,
            llm=llm_out,
            selected=selected,
        )

    def _enrich_candidate(
        self, concept: dict, source: str, score: float | None
    ) -> dict[str, Any]:
        anchored = self._graph.resolve_concept(concept)
        parents = self._graph.parents(anchored)
        ancestors = self._graph.ancestor_chain(anchored)
        parent_names = [str(p.get("name") or "") for p in parents if p.get("name")]
        chain = [str(a.get("name") or "") for a in ancestors if a.get("name")]
        view = self._concept_view(anchored)
        return {
            **view.to_dict(),
            "match_source": source,
            "score": score,
            "parent_names": parent_names,
            "ancestor_summary": " › ".join(chain) if chain else None,
        }

    def _lookup_english(self, raw: str, query_lang: str) -> LookupResult:
        concept = self._graph.exact_en(raw)
        if concept:
            return self._result_from_concept(raw, query_lang, concept, "exact", 100.0)

        fuzzy = self._graph.fuzzy_en(raw, self._fuzzy_cutoff)
        if fuzzy:
            concept, score = fuzzy
            return self._result_from_concept(raw, query_lang, concept, "fuzzy", score)

        sem = self._semantic_match(raw, "en")
        if sem:
            concept, score = sem
            return self._result_from_concept(
                raw, query_lang, concept, "semantic", round(score * 100, 1)
            )

        return LookupResult(
            query=raw,
            query_lang=query_lang,
            match_type="none",
            score=None,
            ambiguous=False,
            concept=None,
            message="No English match. Try French or another spelling.",
        )

    def _lookup_french(self, raw: str, query_lang: str) -> LookupResult:
        key = norm_key(raw)
        self._graph._load()

        if self._graph.is_ambiguous_fr(raw):
            alts_rows = self._graph.alternatives_fr(key)
            if alts_rows:
                res = self._result_from_concept(raw, query_lang, alts_rows[0], "exact", 100.0)
                res.ambiguous = True
                res.alternatives = [self._concept_view(c) for c in alts_rows[1:]]
                res.message = f"Ambiguous French label ({len(alts_rows)} concepts)."
                return res

        concept = self._graph.exact_fr(raw)
        if concept:
            return self._result_from_concept(raw, query_lang, concept, "exact", 100.0)

        fuzzy = self._graph.fuzzy_fr(raw, self._fuzzy_cutoff)
        if fuzzy:
            concept, score = fuzzy
            return self._result_from_concept(raw, query_lang, concept, "fuzzy", score)

        sem = self._semantic_match(raw, "fr")
        if sem:
            concept, score = sem
            return self._result_from_concept(
                raw, query_lang, concept, "semantic", round(score * 100, 1)
            )

        return LookupResult(
            query=raw,
            query_lang=query_lang,
            match_type="none",
            score=None,
            ambiguous=False,
            concept=None,
            message="No match via exact, fuzzy, or semantic search.",
        )

    def _result_from_concept(
        self,
        query: str,
        query_lang: str,
        concept: dict,
        kind: MatchKind,
        score: float,
    ) -> LookupResult:
        anchored = self._graph.resolve_concept(concept)
        view = self._concept_view(anchored)
        parents = [self._concept_view(c) for c in self._graph.parents(anchored)]
        children = [self._concept_view(c) for c in self._graph.children(anchored)]
        ancestors = [self._concept_view(c) for c in self._graph.ancestor_chain(anchored)]
        return LookupResult(
            query=query,
            query_lang=query_lang,
            match_type=kind,
            score=score,
            ambiguous=False,
            concept=view,
            parents=parents,
            children=children,
            ancestors=ancestors,
        )

    def _concept_view(self, concept: dict) -> ConceptView:
        fr = concept.get("fr_label")
        fr_s = str(fr).strip() if fr else None
        name = str(concept.get("name") or "")
        return ConceptView(
            id=str(concept.get("id") or name),
            name=name,
            en_label=name,
            level=int(concept["level"]) if concept.get("level") is not None else None,
            tier=str(concept.get("tier") or ""),
            fr_label=fr_s or None,
        )

    def _ensure_semantic_index(self, lang: str) -> None:
        idx = self._semantic[lang]
        if idx.ready:
            return
        with self._semantic_lock:
            if idx.ready:
                return
            log.info("Loading semantic model %s for %s…", self._embed_model_id, lang)
            from sentence_transformers import SentenceTransformer, util

            corpus = self._graph.semantic_corpus(lang)
            idx.labels = [t for t, _ in corpus]
            idx.concepts = [c for _, c in corpus]
            idx.model = SentenceTransformer(self._embed_model_id)
            idx.matrix = idx.model.encode(
                idx.labels, convert_to_tensor=True, show_progress_bar=False
            )
            idx.ready = True
            log.info("Semantic index ready (%s): %d labels", lang, len(idx.labels))

    def _semantic_match(self, raw: str, lang: str) -> tuple[dict, float] | None:
        try:
            self._ensure_semantic_index(lang)
        except Exception as exc:
            log.warning("Semantic search unavailable (%s): %s", lang, exc)
            return None
        idx = self._semantic[lang]
        assert idx.model is not None and idx.matrix is not None
        from sentence_transformers import util

        q_emb = idx.model.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, idx.matrix)[0]
        best_i = int(scores.argmax())
        best = float(scores[best_i])
        if best < idx.threshold:
            return None
        return idx.concepts[best_i], best

    def _semantic_top_k(
        self, raw: str, lang: str, *, k: int = 5
    ) -> list[tuple[dict, float]]:
        try:
            self._ensure_semantic_index(lang)
        except Exception as exc:
            log.warning("Semantic top-k unavailable (%s): %s", lang, exc)
            return []
        idx = self._semantic[lang]
        assert idx.model is not None and idx.matrix is not None
        from sentence_transformers import util

        q_emb = idx.model.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, idx.matrix)[0]
        floor = idx.threshold * 0.85
        pairs: list[tuple[dict, float]] = []
        seen: set[str] = set()
        for i in scores.argsort(descending=True)[: k * 3]:
            sc = float(scores[i])
            if sc < floor:
                break
            concept = idx.concepts[int(i)]
            cid = str(concept.get("id", ""))
            if cid in seen:
                continue
            seen.add(cid)
            pairs.append((concept, sc))
            if len(pairs) >= k:
                break
        return pairs


_service: MeddraLookupService | None = None


def get_lookup_service() -> MeddraLookupService:
    global _service
    if _service is None:
        _service = MeddraLookupService()
    return _service
