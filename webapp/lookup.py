"""MedDRA term lookup: string match, fuzzy fallback, semantic fallback."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util

from pipeline import TermGraph, _norm

log = logging.getLogger(__name__)

MatchKind = Literal["exact", "fuzzy", "semantic", "none"]

_TIER_ORDER = {"SOC": 1, "HLGT": 2, "HLT": 3, "PT": 4, "LLT": 5}


@dataclass
class ConceptView:
    id: str
    name: str
    level: int | None
    tier: str
    fr_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level,
            "tier": self.tier,
            "fr_label": self.fr_label,
        }


@dataclass
class LookupResult:
    query: str
    match_type: MatchKind
    score: float | None
    ambiguous: bool
    concept: ConceptView | None
    parents: list[ConceptView] = field(default_factory=list)
    children: list[ConceptView] = field(default_factory=list)
    ancestors: list[ConceptView] = field(default_factory=list)
    alternatives: list[ConceptView] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "match_type": self.match_type,
            "score": self.score,
            "ambiguous": self.ambiguous,
            "concept": self.concept.to_dict() if self.concept else None,
            "parents": [p.to_dict() for p in self.parents],
            "children": [c.to_dict() for c in self.children],
            "ancestors": [a.to_dict() for a in self.ancestors],
            "alternatives": [a.to_dict() for a in self.alternatives],
            "message": self.message,
        }


class MeddraLookupService:
    """String grounding via TermGraph; semantic search over French PT/LLT labels."""

    def __init__(self) -> None:
        self._graph = TermGraph(grounding_mode="string")
        self._embed_model_id = os.environ.get(
            "TERMPLAN_EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2"
        )
        self._semantic_threshold = float(os.environ.get("LOOKUP_SEMANTIC_MIN", "0.55"))
        self._fuzzy_cutoff = float(os.environ.get("GROUND_FUZZY_CUTOFF", "90"))
        self._embed_model: SentenceTransformer | None = None
        self._embed_labels: list[str] = []
        self._embed_concepts: list[dict] = []
        self._embed_matrix = None
        self._embed_lock = threading.Lock()
        self._embed_ready = False

    def close(self) -> None:
        self._graph.close()

    def health(self) -> dict[str, Any]:
        try:
            self._graph._load()
            return {"status": "ok", "neo4j": "connected", "labels_loaded": len(self._graph._cache or {})}
        except Exception as exc:
            return {"status": "error", "neo4j": "unavailable", "detail": str(exc)}

    def lookup(self, term: str, *, lang: str = "fr") -> LookupResult:
        raw = (term or "").strip()
        if not raw:
            return LookupResult(
                query=raw,
                match_type="none",
                score=None,
                ambiguous=False,
                concept=None,
                message="Enter a term to search.",
            )

        lang = (lang or "fr").lower()
        if lang.startswith("en"):
            return self._lookup_english(raw)
        return self._lookup_french(raw)

    def _lookup_english(self, raw: str) -> LookupResult:
        node = self._graph.get_by_name(raw)
        if node:
            return self._result_from_concept(raw, node, "exact", 100.0)

        concepts = self._all_concepts_for_fuzzy_en()
        keys = [_norm(c["name"]) for c in concepts]
        hit = process.extractOne(
            _norm(raw),
            keys,
            scorer=fuzz.ratio,
            score_cutoff=self._fuzzy_cutoff,
        )
        if not hit:
            sem = self._semantic_match(raw)
            if sem:
                concept, score = sem
                return self._result_from_concept(raw, concept, "semantic", round(score * 100, 1))
            return LookupResult(
                query=raw,
                match_type="none",
                score=None,
                ambiguous=False,
                concept=None,
                message="No English concept matched. Try French or another spelling.",
            )
        node = concepts[hit[2]]
        return self._result_from_concept(raw, node, "fuzzy", float(hit[1]))

    def _lookup_french(self, raw: str) -> LookupResult:
        key = _norm(raw)
        self._graph._load()
        assert self._graph._cache is not None

        if key in self._graph._ambiguous:
            alts = self._alternatives_for_key(key)
            primary = alts[0] if alts else None
            if primary:
                res = self._result_from_concept(raw, primary, "exact", 100.0)
                res.ambiguous = True
                res.alternatives = alts[1:]
                res.message = f"Ambiguous French label ({len(alts)} concepts)."
                return res

        if key in self._graph._cache:
            return self._result_from_concept(raw, self._graph._cache[key], "exact", 100.0)

        hit = process.extractOne(
            key,
            self._graph._fuzzy_keys,
            scorer=fuzz.ratio,
            score_cutoff=self._fuzzy_cutoff,
        )
        if hit:
            concept = self._graph._fuzzy_vals[hit[2]]
            return self._result_from_concept(raw, concept, "fuzzy", float(hit[1]))

        sem = self._semantic_match(raw)
        if sem:
            concept, score = sem
            return self._result_from_concept(raw, concept, "semantic", round(score * 100, 1))

        return LookupResult(
            query=raw,
            match_type="none",
            score=None,
            ambiguous=False,
            concept=None,
            message="No match via string, fuzzy, or semantic search.",
        )

    def _result_from_concept(
        self,
        query: str,
        concept: dict,
        kind: MatchKind,
        score: float,
    ) -> LookupResult:
        cid = str(concept["id"])
        view = self._concept_view(concept)
        parents = self._direct_relatives(cid, direction="parents")
        children = self._direct_relatives(cid, direction="children")
        ancestors = self._ancestor_chain(cid, view)
        return LookupResult(
            query=query,
            match_type=kind,
            score=score,
            ambiguous=False,
            concept=view,
            parents=parents,
            children=children,
            ancestors=ancestors,
        )

    def _concept_view(self, concept: dict) -> ConceptView:
        cid = str(concept.get("id") or concept.get("name") or "")
        fr = self._fr_label_for_id(cid)
        return ConceptView(
            id=cid,
            name=str(concept.get("name") or ""),
            level=int(concept["level"]) if concept.get("level") is not None else None,
            tier=str(concept.get("tier") or ""),
            fr_label=fr,
        )

    def _fr_label_for_id(self, concept_id: str) -> str | None:
        q = """
        MATCH (c:Concept) WHERE coalesce(c.id, c.name) = $cid
        RETURN c.fr_label AS fr_label LIMIT 1
        """
        with self._graph._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        if not rec:
            return None
        fr = rec.get("fr_label")
        return str(fr).strip() if fr else None

    def _direct_relatives(self, concept_id: str, *, direction: str) -> list[ConceptView]:
        if direction == "parents":
            q = """
            MATCH (p:Concept)-[:BROADER_THAN]->(c:Concept)
            WHERE coalesce(c.id, c.name) = $cid
            RETURN coalesce(p.id, p.name) AS id, p.name AS name,
                   p.level AS level, p.tier AS tier, p.fr_label AS fr_label
            ORDER BY p.level
            """
        else:
            q = """
            MATCH (c:Concept)-[:BROADER_THAN]->(ch:Concept)
            WHERE coalesce(c.id, c.name) = $cid
            RETURN coalesce(ch.id, ch.name) AS id, ch.name AS name,
                   ch.level AS level, ch.tier AS tier, ch.fr_label AS fr_label
            ORDER BY ch.level
            """
        out: list[ConceptView] = []
        with self._graph._driver.session() as s:
            for rec in s.run(q, cid=concept_id):
                out.append(
                    ConceptView(
                        id=str(rec["id"]),
                        name=str(rec["name"] or ""),
                        level=int(rec["level"]) if rec["level"] is not None else None,
                        tier=str(rec["tier"] or ""),
                        fr_label=str(rec["fr_label"]).strip() if rec.get("fr_label") else None,
                    )
                )
        return out

    def _ancestor_chain(self, concept_id: str, current: ConceptView) -> list[ConceptView]:
        """SOC → … → current, ordered broadest-first."""
        q = """
        MATCH (c:Concept) WHERE coalesce(c.id, c.name) = $cid
        OPTIONAL MATCH (p:Concept)-[:BROADER_THAN*1..10]->(c)
        WITH collect(DISTINCT p) AS parents, c AS node
        RETURN parents, node
        """
        nodes: list[ConceptView] = []
        with self._graph._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
            if rec:
                for p in rec["parents"] or []:
                    nodes.append(
                        ConceptView(
                            id=str(p.get("id") or p.get("name") or ""),
                            name=str(p.get("name") or ""),
                            level=int(p["level"]) if p.get("level") is not None else None,
                            tier=str(p.get("tier") or ""),
                            fr_label=str(p.get("fr_label") or "").strip() or None,
                        )
                    )
        nodes.sort(key=lambda n: (_TIER_ORDER.get(n.tier, 99), n.level or 99))
        if not any(n.id == current.id for n in nodes):
            nodes.append(current)
        else:
            nodes = [n for n in nodes if n.id != current.id] + [current]
        return nodes

    def _alternatives_for_key(self, key: str) -> list[ConceptView]:
        q = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
        WITH c, toLower(trim(c.fr_label)) AS k
        WHERE k = $key
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        ORDER BY c.level
        """
        out: list[ConceptView] = []
        with self._graph._driver.session() as s:
            for rec in s.run(q, key=key):
                out.append(
                    ConceptView(
                        id=str(rec["id"]),
                        name=str(rec["name"] or ""),
                        level=int(rec["level"]) if rec["level"] is not None else None,
                        tier=str(rec["tier"] or ""),
                        fr_label=str(rec["fr_label"] or "").strip() or None,
                    )
                )
        return out

    def _all_concepts_for_fuzzy_en(self) -> list[dict]:
        q = """
        MATCH (c:Concept) WHERE c.name IS NOT NULL
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier LIMIT 50000
        """
        with self._graph._driver.session() as s:
            return [dict(r) for r in s.run(q)]

    def _ensure_semantic_index(self) -> None:
        if self._embed_ready:
            return
        with self._embed_lock:
            if self._embed_ready:
                return
            log.info("Building semantic index (%s)…", self._embed_model_id)
            self._graph._load()
            q = """
            MATCH (c:Concept)
            WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
              AND c.tier IN ['PT', 'LLT']
            RETURN coalesce(c.id, c.name) AS id, c.name AS name,
                   c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
            labels: list[str] = []
            concepts: list[dict] = []
            with self._graph._driver.session() as s:
                for rec in s.run(q):
                    fr = str(rec["fr_label"]).strip()
                    if not fr:
                        continue
                    labels.append(fr)
                    concepts.append(
                        {
                            "id": rec["id"],
                            "name": rec["name"],
                            "level": rec["level"],
                            "tier": rec["tier"],
                            "fr_label": fr,
                        }
                    )
            self._embed_labels = labels
            self._embed_concepts = concepts
            self._embed_model = SentenceTransformer(self._embed_model_id)
            self._embed_matrix = self._embed_model.encode(
                labels, convert_to_tensor=True, show_progress_bar=False
            )
            self._embed_ready = True
            log.info("Semantic index ready: %d labels", len(labels))

    def _semantic_match(self, raw: str) -> tuple[dict, float] | None:
        try:
            self._ensure_semantic_index()
        except Exception as exc:
            log.warning("Semantic index unavailable: %s", exc)
            return None
        assert self._embed_model is not None and self._embed_matrix is not None
        q_emb = self._embed_model.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self._embed_matrix)[0]
        best_i = int(scores.argmax())
        best = float(scores[best_i])
        if best < self._semantic_threshold:
            return None
        return self._embed_concepts[best_i], best


_service: MeddraLookupService | None = None


def get_lookup_service() -> MeddraLookupService:
    global _service
    if _service is None:
        _service = MeddraLookupService()
    return _service
