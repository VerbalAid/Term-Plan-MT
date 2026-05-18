"""Neo4j MedDRA graph access for the lookup web app (string / fuzzy only)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

from neo4j import Driver, GraphDatabase
from rapidfuzz import fuzz, process

from webapp.neo4j_config import friendly_connection_error, neo4j_password, neo4j_uri, neo4j_user

_TIER_ORDER = {"SOC": 1, "HLGT": 2, "HLT": 3, "PT": 4, "LLT": 5}
_MEDDRA_CODE = re.compile(r"^\d+$")


def norm_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", (text or "").strip()).casefold()
    return " ".join(s.split())


def is_meddra_code(value: str) -> bool:
    return bool(_MEDDRA_CODE.match(str(value or "").strip()))


class MeddraGraph:
    """French-label cache + English-name index backed by Neo4j.

    Hierarchy follows ``data/build_graph.py``:
    ``(broader:Concept)-[:BROADER_THAN]->(narrower:Concept)`` (SOC → … → LLT).
    """

    def __init__(self) -> None:
        uri = neo4j_uri()
        self._driver: Driver = GraphDatabase.driver(
            uri, auth=(neo4j_user(), neo4j_password())
        )
        self._fr_cache: dict[str, dict] | None = None
        self._fr_fuzzy_keys: list[str] = []
        self._fr_fuzzy_vals: list[dict] = []
        self._en_fuzzy_keys: list[str] = []
        self._en_fuzzy_vals: list[dict] = []
        self._ambiguous: set[str] = set()
        self._ambiguous_counts: dict[str, int] = {}
        self._label_count = 0
        self._load_lock = threading.Lock()
        self._count_cached_at: float = 0.0
        self._count_cached_val: int = -1

    def close(self) -> None:
        self._driver.close()

    @staticmethod
    def _row_to_concept(rec) -> dict:
        return {
            "id": rec["id"],
            "name": rec["name"],
            "level": rec["level"],
            "tier": rec["tier"],
            "fr_label": rec.get("fr_label"),
        }

    @staticmethod
    def _node_props(node) -> dict[str, Any]:
        if node is None:
            return {}
        if hasattr(node, "_properties"):
            return dict(node._properties)
        if hasattr(node, "items"):
            return dict(node.items())
        return dict(node)

    def verify_connectivity(self) -> None:
        with self._driver.session() as s:
            s.run("RETURN 1").consume()

    def quick_label_count(self, *, ttl_seconds: float = 120.0) -> int:
        """COUNT for health checks (cached; does not build fuzzy indexes)."""
        now = time.monotonic()
        if self._count_cached_val >= 0 and now - self._count_cached_at < ttl_seconds:
            return self._count_cached_val
        q = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
          AND c.id =~ '^[0-9]+$'
        RETURN count(c) AS n
        """
        with self._driver.session() as s:
            n = int(s.run(q).single()["n"])
        self._count_cached_at = now
        self._count_cached_val = n
        return n

    def cache_ready(self) -> bool:
        return self._fr_cache is not None and bool(self._en_fuzzy_keys)

    def _load_fr(self) -> None:
        if self._fr_cache is not None:
            return
        with self._load_lock:
            if self._fr_cache is not None:
                return
            log.info("Loading French label cache from Neo4j…")
            buckets: dict[str, list[dict]] = defaultdict(list)
            q_fr = """
            MATCH (c:Concept)
            WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
              AND c.id =~ '^[0-9]+$'
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
            with self._driver.session() as s:
                for r in s.run(q_fr):
                    k = norm_key(r["fr_label"])
                    buckets[k].append(self._row_to_concept(r))

            cache: dict[str, dict] = {}
            for k, items in buckets.items():
                ids = {str(p["id"]) for p in items}
                self._ambiguous_counts[k] = len(ids)
                if len(ids) > 1:
                    self._ambiguous.add(k)
                cache[k] = items[0]

            self._fr_fuzzy_keys = []
            self._fr_fuzzy_vals = []
            q_fuzzy = """
            MATCH (c:Concept)
            WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
              AND c.tier IN ['PT', 'LLT'] AND c.id =~ '^[0-9]+$'
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
            with self._driver.session() as s:
                for r in s.run(q_fuzzy):
                    self._fr_fuzzy_keys.append(norm_key(r["fr_label"]))
                    self._fr_fuzzy_vals.append(self._row_to_concept(r))

            self._fr_cache = cache
            self._label_count = len(cache)
            log.info("French cache ready: %d labels", self._label_count)

    def _load_en(self) -> None:
        if self._en_fuzzy_keys:
            return
        with self._load_lock:
            if self._en_fuzzy_keys:
                return
            log.info("Loading English fuzzy index from Neo4j…")
            self._en_fuzzy_keys = []
            self._en_fuzzy_vals = []
            q_en = """
            MATCH (c:Concept)
            WHERE c.name IS NOT NULL AND trim(c.name) <> ''
              AND c.tier IN ['PT', 'LLT', 'HLT', 'SOC'] AND c.id =~ '^[0-9]+$'
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
            with self._driver.session() as s:
                for r in s.run(q_en):
                    self._en_fuzzy_keys.append(norm_key(r["name"]))
                    self._en_fuzzy_vals.append(self._row_to_concept(r))
            log.info("English fuzzy index ready: %d terms", len(self._en_fuzzy_keys))

    def label_count(self) -> int:
        if self._fr_cache is not None:
            return self._label_count
        return self.quick_label_count()

    def is_ambiguous_fr(self, fr_term: str) -> bool:
        self._load_fr()
        return norm_key(fr_term) in self._ambiguous

    def exact_fr(self, fr_term: str) -> dict | None:
        self._load_fr()
        assert self._fr_cache is not None
        hit = self._fr_cache.get(norm_key(fr_term))
        return dict(hit) if hit else None

    def fuzzy_fr(self, fr_term: str, cutoff: float) -> tuple[dict, float] | None:
        hits = self.fuzzy_fr_candidates(fr_term, cutoff, limit=1)
        return hits[0] if hits else None

    def fuzzy_fr_candidates(
        self, fr_term: str, cutoff: float, *, limit: int = 5
    ) -> list[tuple[dict, float]]:
        self._load_fr()
        hits = process.extract(
            norm_key(fr_term),
            self._fr_fuzzy_keys,
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=limit,
        )
        out: list[tuple[dict, float]] = []
        seen: set[str] = set()
        for _label, score, idx in hits:
            concept = dict(self._fr_fuzzy_vals[idx])
            cid = str(concept.get("id", ""))
            if cid in seen:
                continue
            seen.add(cid)
            out.append((concept, float(score)))
        return out

    def exact_en(self, en_term: str) -> dict | None:
        q = """
        MATCH (c:Concept)
        WHERE toLower(trim(c.name)) = toLower(trim($n)) AND c.id =~ '^[0-9]+$'
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        ORDER BY c.level DESC
        LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, n=en_term).single()
        return self._row_to_concept(rec) if rec else None

    def fuzzy_en(self, en_term: str, cutoff: float) -> tuple[dict, float] | None:
        hits = self.fuzzy_en_candidates(en_term, cutoff, limit=1)
        return hits[0] if hits else None

    def _fuzzy_en_via_cypher(
        self, en_term: str, *, limit: int = 5
    ) -> list[tuple[dict, float]]:
        """Fast path when the in-memory English index is not built yet."""
        needle = (en_term or "").strip()
        if not needle:
            return []
        q = """
        MATCH (c:Concept)
        WHERE c.name IS NOT NULL AND toLower(c.name) CONTAINS toLower($n)
          AND c.tier IN ['PT', 'LLT'] AND c.id =~ '^[0-9]+$'
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        LIMIT $lim
        """
        out: list[tuple[dict, float]] = []
        with self._driver.session() as s:
            for rec in s.run(q, n=needle, lim=limit * 4):
                concept = self._row_to_concept(rec)
                name = str(rec.get("name") or "")
                score = float(fuzz.ratio(norm_key(needle), norm_key(name)))
                out.append((concept, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]

    def fuzzy_en_candidates(
        self, en_term: str, cutoff: float, *, limit: int = 5
    ) -> list[tuple[dict, float]]:
        if not self._en_fuzzy_keys:
            hits = self._fuzzy_en_via_cypher(en_term, limit=limit)
            return [(c, s) for c, s in hits if s >= cutoff][:limit]
        self._load_en()
        hits = process.extract(
            norm_key(en_term),
            self._en_fuzzy_keys,
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=limit,
        )
        out: list[tuple[dict, float]] = []
        seen: set[str] = set()
        for _label, score, idx in hits:
            concept = dict(self._en_fuzzy_vals[idx])
            cid = str(concept.get("id", ""))
            if cid in seen:
                continue
            seen.add(cid)
            out.append((concept, float(score)))
        return out

    def alternatives_fr(self, key: str) -> list[dict]:
        q = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND toLower(trim(c.fr_label)) = $key
          AND c.id =~ '^[0-9]+$'
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        ORDER BY c.level
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, key=key)]

    def resolve_concept(self, concept: dict) -> dict:
        """Re-anchor to a canonical MedDRA-coded node with hierarchy edges when possible."""
        cid = str(concept.get("id") or "").strip()
        name = str(concept.get("name") or "").strip()
        fr = str(concept.get("fr_label") or "").strip()

        if is_meddra_code(cid):
            row = self.concept_by_id(cid)
            if row and self._has_hierarchy(row["id"]):
                return row

        q = """
        MATCH (c:Concept)
        WHERE c.id =~ '^[0-9]+$'
          AND (
            ($cid <> '' AND c.id = $cid)
            OR ($name <> '' AND toLower(c.name) = toLower($name))
            OR ($fr <> '' AND toLower(trim(c.fr_label)) = toLower(trim($fr)))
          )
        OPTIONAL MATCH (p:Concept)-[:BROADER_THAN]->(c)
        OPTIONAL MATCH (c)-[:BROADER_THAN]->(ch:Concept)
        WITH c, count(DISTINCT p) + count(DISTINCT ch) AS rels
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label, rels
        ORDER BY rels DESC, c.level DESC
        LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=cid, name=name, fr=fr).single()
        if rec:
            return self._row_to_concept(rec)
        return dict(concept)

    def concept_by_id(self, concept_id: str) -> dict | None:
        if is_meddra_code(concept_id):
            q = """
            MATCH (c:Concept {id: $cid})
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
            with self._driver.session() as s:
                rec = s.run(q, cid=concept_id).single()
            return self._row_to_concept(rec) if rec else None
        q = """
        MATCH (c:Concept)
        WHERE c.id = $cid OR toLower(c.name) = toLower($cid)
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        return self._row_to_concept(rec) if rec else None

    def _has_hierarchy(self, concept_id: str) -> bool:
        q = """
        MATCH (c:Concept {id: $cid})
        OPTIONAL MATCH (p:Concept)-[:BROADER_THAN]->(c)
        OPTIONAL MATCH (c)-[:BROADER_THAN]->(ch:Concept)
        RETURN count(p) + count(ch) AS n
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        return bool(rec and rec["n"] and int(rec["n"]) > 0)

    def _anchor_id(self, concept: dict) -> str:
        resolved = self.resolve_concept(concept)
        return str(resolved["id"])

    def parents(self, concept: dict) -> list[dict]:
        """Immediate broader concepts: (parent)-[:BROADER_THAN]->(child)."""
        cid = self._anchor_id(concept)
        q = """
        MATCH (c:Concept {id: $cid})
        MATCH (p:Concept)-[:BROADER_THAN]->(c)
        RETURN p.id AS id, p.name AS name, p.level AS level, p.tier AS tier, p.fr_label AS fr_label
        ORDER BY p.level ASC
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, cid=cid)]

    def children(self, concept: dict) -> list[dict]:
        """Immediate narrower concepts: (c)-[:BROADER_THAN]->(child)."""
        cid = self._anchor_id(concept)
        q = """
        MATCH (c:Concept {id: $cid})
        MATCH (c)-[:BROADER_THAN]->(ch:Concept)
        RETURN ch.id AS id, ch.name AS name, ch.level AS level, ch.tier AS tier, ch.fr_label AS fr_label
        ORDER BY ch.level ASC
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, cid=cid)]

    def ancestor_chain(self, concept: dict) -> list[dict]:
        """SOC → … → concept (broadest first). Up to four hops covers SOC…LLT."""
        cid = self._anchor_id(concept)
        q = """
        MATCH (c:Concept {id: $cid})
        OPTIONAL MATCH (ancestor:Concept)-[:BROADER_THAN*1..4]->(c)
        WITH c, [a IN collect(DISTINCT ancestor) WHERE a IS NOT NULL] AS ancestors
        RETURN ancestors, c
        """
        nodes: list[dict] = []
        with self._driver.session() as s:
            rec = s.run(q, cid=cid).single()
            if not rec:
                return nodes
            for node in rec["ancestors"] or []:
                props = self._node_props(node)
                if props.get("name"):
                    nodes.append(
                        {
                            "id": props.get("id") or props.get("name"),
                            "name": props.get("name"),
                            "level": props.get("level"),
                            "tier": props.get("tier"),
                            "fr_label": props.get("fr_label"),
                        }
                    )
            current = self._node_props(rec["c"])
            if current.get("name"):
                nodes.append(
                    {
                        "id": current.get("id") or current.get("name"),
                        "name": current.get("name"),
                        "level": current.get("level"),
                        "tier": current.get("tier"),
                        "fr_label": current.get("fr_label"),
                    }
                )

        def sort_key(c: dict) -> tuple:
            tier = str(c.get("tier") or "")
            lvl = c.get("level")
            return (_TIER_ORDER.get(tier, 99), int(lvl) if lvl is not None else 99)

        nodes.sort(key=sort_key)
        seen: set[str] = set()
        out: list[dict] = []
        for n in nodes:
            nid = str(n.get("id") or "")
            if nid and nid not in seen:
                seen.add(nid)
                out.append(n)
        return out

    def neighborhood(self, concept: dict) -> dict[str, Any]:
        """Debug: parents, children, rel types for a matched concept."""
        resolved = self.resolve_concept(concept)
        cid = str(resolved["id"])
        q = """
        MATCH (c:Concept {id: $cid})
        OPTIONAL MATCH (p:Concept)-[rp:BROADER_THAN]->(c)
        OPTIONAL MATCH (c)-[rc:BROADER_THAN]->(ch:Concept)
        RETURN c.id AS id, c.name AS name, c.tier AS tier, c.fr_label AS fr_label,
               collect(DISTINCT {id: p.id, name: p.name, tier: p.tier}) AS parents,
               collect(DISTINCT {id: ch.id, name: ch.name, tier: ch.tier}) AS children
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=cid).single()
        if not rec:
            return {"resolved": resolved, "error": "node not found after resolve"}
        return {
            "resolved": resolved,
            "parents": [x for x in rec["parents"] if x.get("id")],
            "children": [x for x in rec["children"] if x.get("id")],
        }

    def schema_summary(self) -> dict[str, Any]:
        q = """
        MATCH ()-[r]->()
        RETURN type(r) AS typ, count(r) AS n
        ORDER BY n DESC
        LIMIT 10
        """
        with self._driver.session() as s:
            rels = [dict(r) for r in s.run(q)]
        q2 = """
        MATCH (c:Concept)
        RETURN
          count(c) AS total,
          sum(CASE WHEN c.id =~ '^[0-9]+$' THEN 1 ELSE 0 END) AS meddra_coded,
          sum(CASE WHEN NOT c.id =~ '^[0-9]+$' THEN 1 ELSE 0 END) AS legacy_name_ids
        """
        with self._driver.session() as s:
            stats = dict(s.run(q2).single())
        return {
            "relationship_types": rels,
            "concept_stats": stats,
            "hierarchy": "(broader)-[:BROADER_THAN]->(narrower)",
        }

    def semantic_corpus(self, lang: str) -> list[tuple[str, dict]]:
        """PT/LLT rows with numeric MedDRA codes only (skips orphan seed nodes)."""
        if lang == "fr":
            q = """
            MATCH (c:Concept)
            WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
              AND c.tier IN ['PT', 'LLT'] AND c.id =~ '^[0-9]+$'
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
        else:
            q = """
            MATCH (c:Concept)
            WHERE c.name IS NOT NULL AND trim(c.name) <> ''
              AND c.tier IN ['PT', 'LLT'] AND c.id =~ '^[0-9]+$'
            RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
        out: list[tuple[str, dict]] = []
        with self._driver.session() as s:
            for rec in s.run(q):
                concept = self._row_to_concept(rec)
                if lang == "fr":
                    fr = str(rec["fr_label"] or "").strip()
                    text = f"{fr} | {rec['name']}" if rec.get("name") else fr
                else:
                    text = str(rec["name"] or "").strip()
                if text:
                    out.append((text, concept))
        return out
