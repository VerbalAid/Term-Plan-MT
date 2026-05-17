"""Neo4j MedDRA graph access for the lookup web app (string / fuzzy only)."""

from __future__ import annotations

import os
import unicodedata
from collections import defaultdict

from neo4j import Driver, GraphDatabase
from rapidfuzz import fuzz, process

_TIER_ORDER = {"SOC": 1, "HLGT": 2, "HLT": 3, "PT": 4, "LLT": 5}


def norm_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", (text or "").strip()).casefold()
    return " ".join(s.split())


class MeddraGraph:
    """French-label cache + English-name index backed by Neo4j."""

    def __init__(self) -> None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd = os.environ.get("NEO4J_PASS", "password")
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._fr_cache: dict[str, dict] | None = None
        self._fr_fuzzy_keys: list[str] = []
        self._fr_fuzzy_vals: list[dict] = []
        self._en_fuzzy_keys: list[str] = []
        self._en_fuzzy_vals: list[dict] = []
        self._ambiguous: set[str] = set()
        self._ambiguous_counts: dict[str, int] = {}
        self._label_count = 0

    def close(self) -> None:
        self._driver.close()

    def _load(self) -> None:
        if self._fr_cache is not None:
            return

        buckets: dict[str, list[dict]] = defaultdict(list)
        q_fr = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
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
          AND c.tier IN ['PT', 'LLT']
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        """
        with self._driver.session() as s:
            for r in s.run(q_fuzzy):
                self._fr_fuzzy_keys.append(norm_key(r["fr_label"]))
                self._fr_fuzzy_vals.append(self._row_to_concept(r))

        self._en_fuzzy_keys = []
        self._en_fuzzy_vals = []
        q_en = """
        MATCH (c:Concept)
        WHERE c.name IS NOT NULL AND trim(c.name) <> ''
          AND c.tier IN ['PT', 'LLT', 'HLT', 'SOC']
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        """
        with self._driver.session() as s:
            for r in s.run(q_en):
                self._en_fuzzy_keys.append(norm_key(r["name"]))
                self._en_fuzzy_vals.append(self._row_to_concept(r))

        self._fr_cache = cache
        self._label_count = len(cache)

    @staticmethod
    def _row_to_concept(rec) -> dict:
        return {
            "id": rec["id"],
            "name": rec["name"],
            "level": rec["level"],
            "tier": rec["tier"],
            "fr_label": rec.get("fr_label"),
        }

    def label_count(self) -> int:
        self._load()
        return self._label_count

    def is_ambiguous_fr(self, fr_term: str) -> bool:
        self._load()
        return norm_key(fr_term) in self._ambiguous

    def exact_fr(self, fr_term: str) -> dict | None:
        self._load()
        assert self._fr_cache is not None
        hit = self._fr_cache.get(norm_key(fr_term))
        return dict(hit) if hit else None

    def fuzzy_fr(self, fr_term: str, cutoff: float) -> tuple[dict, float] | None:
        self._load()
        hit = process.extractOne(
            norm_key(fr_term),
            self._fr_fuzzy_keys,
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
        )
        if not hit:
            return None
        return dict(self._fr_fuzzy_vals[hit[2]]), float(hit[1])

    def exact_en(self, en_term: str) -> dict | None:
        q = """
        MATCH (c:Concept)
        WHERE toLower(trim(c.name)) = toLower(trim($n))
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, n=en_term).single()
        return self._row_to_concept(rec) if rec else None

    def fuzzy_en(self, en_term: str, cutoff: float) -> tuple[dict, float] | None:
        self._load()
        hit = process.extractOne(
            norm_key(en_term),
            self._en_fuzzy_keys,
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
        )
        if not hit:
            return None
        return dict(self._en_fuzzy_vals[hit[2]]), float(hit[1])

    def alternatives_fr(self, key: str) -> list[dict]:
        q = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND toLower(trim(c.fr_label)) = $key
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        ORDER BY c.level
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, key=key)]

    def concept_by_id(self, concept_id: str) -> dict | None:
        q = """
        MATCH (c:Concept)
        WHERE coalesce(c.id, c.name) = $cid
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        return self._row_to_concept(rec) if rec else None

    def parents(self, concept_id: str) -> list[dict]:
        """Immediate broader concepts (one BROADER_THAN hop)."""
        q = """
        MATCH (p:Concept)-[:BROADER_THAN]->(c:Concept)
        WHERE coalesce(c.id, c.name) = $cid
        RETURN coalesce(p.id, p.name) AS id, p.name AS name,
               p.level AS level, p.tier AS tier, p.fr_label AS fr_label
        ORDER BY p.level ASC
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, cid=concept_id)]

    def children(self, concept_id: str) -> list[dict]:
        """Immediate narrower concepts."""
        q = """
        MATCH (c:Concept)-[:BROADER_THAN]->(ch:Concept)
        WHERE coalesce(c.id, c.name) = $cid
        RETURN coalesce(ch.id, ch.name) AS id, ch.name AS name,
               ch.level AS level, ch.tier AS tier, ch.fr_label AS fr_label
        ORDER BY ch.level ASC
        """
        with self._driver.session() as s:
            return [self._row_to_concept(r) for r in s.run(q, cid=concept_id)]

    def ancestor_chain(self, concept_id: str) -> list[dict]:
        """SOC → … → concept, broadest first. Works for SOC (empty) and LLT (full chain)."""
        q = """
        MATCH (c:Concept)
        WHERE coalesce(c.id, c.name) = $cid
        OPTIONAL MATCH (ancestor:Concept)-[:BROADER_THAN*1..5]->(c)
        WITH c, collect(DISTINCT ancestor) AS ancestors
        RETURN ancestors, c
        """
        nodes: list[dict] = []
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
            if not rec:
                return nodes
            for node in rec["ancestors"] or []:
                if node:
                    props = dict(node)
                    nodes.append(
                        {
                            "id": props.get("id") or props.get("name"),
                            "name": props.get("name"),
                            "level": props.get("level"),
                            "tier": props.get("tier"),
                            "fr_label": props.get("fr_label"),
                        }
                    )
            current = rec["c"]
            if current:
                nodes.append(self._row_to_concept(dict(current)))

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

    def semantic_corpus(self, lang: str) -> list[tuple[str, dict]]:
        """PT/LLT rows for embedding index (built lazily by lookup service)."""
        if lang == "fr":
            q = """
            MATCH (c:Concept)
            WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
              AND c.tier IN ['PT', 'LLT']
            RETURN coalesce(c.id, c.name) AS id, c.name AS name,
                   c.level AS level, c.tier AS tier, c.fr_label AS fr_label
            """
        else:
            q = """
            MATCH (c:Concept)
            WHERE c.name IS NOT NULL AND trim(c.name) <> ''
              AND c.tier IN ['PT', 'LLT']
            RETURN coalesce(c.id, c.name) AS id, c.name AS name,
                   c.level AS level, c.tier AS tier, c.fr_label AS fr_label
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
