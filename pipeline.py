"""TermPlanMT — graph grounding, terminology planning, and MedDRA I/O.

This module contains everything needed to connect French NER spans to the
MedDRA ontology stored in Neo4j, plan consistent English renderings across
a document, and load the raw MedDRA flat files that build the graph.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util

load_dotenv(Path(__file__).parent / ".env")
log = logging.getLogger(__name__)

# Sentence-transformer used for vector-based grounding and planning.
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


# ── Internal normalisation ─────────────────────────────────────────────────


def _norm(text: str) -> str:
    """Casefold + collapse whitespace for graph lookups."""
    s = unicodedata.normalize("NFKC", text.strip()).casefold()
    return " ".join(s.split())


def normalize_fr_for_grounding(text: str) -> str:
    """Normalise a French surface form for exact / fuzzy graph lookup keys."""
    return _norm(text)


# ── MedDRA flat-file I/O ───────────────────────────────────────────────────
# These helpers are used by data/build_graph.py to import MedDRA into Neo4j.
# MedDRA is not redistributed; obtain a licence from the MSSO.

# ICH standard: MedDRA hierarchy level → tier abbreviation.
_LEVEL_TO_TIER: dict[int, str] = {1: "SOC", 2: "HLGT", 3: "HLT", 4: "PT", 5: "LLT"}
MEDDRA_TIERS = frozenset(_LEVEL_TO_TIER.values())


def read_meddra_asc(path: Path) -> str:
    """Read a MedDRA ``*.asc`` file, auto-detecting encoding (UTF-8, CP1252, or ISO-8859-1)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_meddra_asc_line(line: str) -> list[str]:
    """Split one MedDRA ``*.asc`` record on ``|`` (standard) or ``$`` (legacy)."""
    line = line.rstrip("\n\r")
    if "|" in line:
        return [p.strip() for p in line.split("|")]
    return [p.strip() for p in line.split("$")]


def _mdhier_primary_flag(parts: list[str]) -> str:
    for j in range(len(parts) - 1, -1, -1):
        s = parts[j].strip().upper()
        if s in ("Y", "N"):
            return s
    return "Y"


def parse_mdhier_row(parts: list[str]) -> dict[str, str] | None:
    """Parse one ``mdhier.asc`` record (pipe or dollar format).

    Returns a dict with llt/pt/hlt/hlgt/soc codes and names, or None if the
    record is too short to be valid.
    """
    if len(parts) < 10:
        return None
    p0, p1, p2, p3 = (parts[i].strip() for i in range(4))
    primary = _mdhier_primary_flag(parts)
    # Dollar (legacy) format: first four fields are all numeric codes.
    if (
        len(parts) >= 11
        and all(p.isdigit() for p in (p0, p1, p2, p3))
        and not parts[4].strip().isdigit()
        and primary in ("Y", "N")
    ):
        return {
            "llt_code": p0, "hlt_code": p1, "hlgt_code": p2, "soc_code": p3,
            "llt_name": parts[4].strip(), "hlt_name": parts[5].strip(),
            "hlgt_name": parts[6].strip(), "soc_name": parts[7].strip(),
            "soc_abbrev": parts[8].strip() if len(parts) > 8 else "",
            "pt_soc_dup": parts[10].strip() if len(parts) > 10 else "",
            "primary_soc_fg": primary, "pt_code": "", "pt_name": "",
        }
    # Standard pipe format.
    return {
        "llt_code": parts[0].strip(), "llt_name": parts[1].strip(),
        "pt_code": parts[2].strip(), "pt_name": parts[3].strip(),
        "hlt_code": parts[4].strip(), "hlt_name": parts[5].strip(),
        "hlgt_code": parts[6].strip(), "hlgt_name": parts[7].strip(),
        "soc_code": parts[8].strip(), "soc_name": parts[9].strip(),
        "primary_soc_fg": "Y",
    }


def load_llt_to_parent_pt(llt_asc: Path) -> dict[str, str]:
    """Build a map of ``llt_code → pt_code`` from the English ``llt.asc`` file."""
    out: dict[str, str] = {}
    for line in read_meddra_asc(llt_asc).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) >= 3:
            llt, pt = parts[0].strip(), parts[2].strip()
            if llt.isdigit() and pt.isdigit():
                out[llt] = pt
    return out


def load_pt_names(pt_asc: Path) -> dict[str, str]:
    """Build a map of ``pt_code → English PT name`` from ``pt.asc``."""
    out: dict[str, str] = {}
    for line in read_meddra_asc(pt_asc).splitlines():
        if not line.strip():
            continue
        parts = split_meddra_asc_line(line)
        if len(parts) >= 2:
            cid, name = parts[0].strip(), parts[1].strip()
            if cid.isdigit():
                out[cid] = name
    return out


def enrich_mdhier_row_pt(
    row: dict[str, str],
    llt_pt: dict[str, str],
    pt_names: dict[str, str],
) -> dict[str, str]:
    """Fill ``pt_code`` / ``pt_name`` for legacy dollar-format rows that omit them."""
    if (row.get("pt_code") or "").strip():
        return row
    r = dict(row)
    llt = (r.get("llt_code") or "").strip()
    pt = (llt_pt.get(llt) or "").strip()
    r["pt_code"] = pt
    r["pt_name"] = (pt_names.get(pt) or r.get("llt_name") or "").strip()
    return r


def canonical_meddra_tier(node_like: dict[str, Any]) -> str:
    """Return the MedDRA tier string (SOC/HLGT/HLT/PT/LLT).

    Prefers the explicit ``tier`` field when valid; falls back to the
    numeric ``level`` field (1 = SOC … 5 = LLT).
    """
    t = str(node_like.get("tier") or "").strip().upper()
    if t in MEDDRA_TIERS:
        return t
    try:
        lvl = int(node_like.get("level"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return t
    return _LEVEL_TO_TIER.get(lvl, t)


# ── Neo4j graph: grounding French NER spans to MedDRA concepts ────────────


class TermGraph:
    """Look up and score French NER spans against the MedDRA Neo4j graph.

    Supports three grounding modes (set via ``grounding_mode`` or .env):
    - ``string``:  exact casefold match, then RapidFuzz fuzzy fallback.
    - ``vector``:  sentence-transformer cosine similarity (requires GPU/RAM).
    - ``llm``:     BioMistral zero-shot (slowest; highest recall).
    """

    def __init__(self, grounding_mode: str = "string") -> None:
        uri  = os.environ.get("NEO4J_URI",  "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd  = os.environ.get("NEO4J_PASS", "password")
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.grounding_mode = grounding_mode.lower()
        self._cache: dict[str, dict] | None = None  # French key → concept payload
        self._fuzzy_keys: list[str] = []
        self._fuzzy_vals: list[dict] = []
        self._ambiguous: set[str] = set()
        self._ambiguous_counts: dict[str, int] = {}

    # ── Cache loading (lazy, runs once per process) ─────────────────────────

    def _load(self) -> None:
        if self._cache is not None:
            return
        q = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        """
        from collections import defaultdict
        buckets: dict[str, list[dict]] = defaultdict(list)
        with self._driver.session() as s:
            for r in s.run(q):
                k = _norm(r["fr_label"])
                buckets[k].append({"id": r["id"], "name": r["name"],
                                    "level": r["level"], "tier": r["tier"]})

        cache: dict[str, dict] = {}
        for k, items in buckets.items():
            ids = {str(p["id"]) for p in items}
            self._ambiguous_counts[k] = len(ids)
            if len(ids) > 1:
                self._ambiguous.add(k)
            cache[k] = items[0]

        # Fuzzy index restricted to PT/LLT to avoid false matches at higher levels.
        q2 = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND c.tier IN ['PT','LLT']
        RETURN coalesce(c.id, c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier, c.fr_label AS fr_label
        """
        with self._driver.session() as s:
            for r in s.run(q2):
                self._fuzzy_keys.append(_norm(r["fr_label"]))
                self._fuzzy_vals.append({"id": r["id"], "name": r["name"],
                                          "level": r["level"], "tier": r["tier"]})
        self._cache = cache

    # ── Public API ─────────────────────────────────────────────────────────

    def ground(self, fr_term: str, *, context: str | None = None) -> dict | None:
        """Ground a French surface form to a MedDRA concept ``{id, name, level, tier}``.

        Returns None when no match is found above the fuzzy threshold.
        """
        self._load()
        assert self._cache is not None
        raw = (fr_term or "").strip()
        if not raw:
            return None
        key = _norm(raw)
        if key in self._ambiguous:
            log.debug("Ambiguous grounding: %r (%d concepts)", raw, self._ambiguous_counts[key])
        if key in self._cache:
            return dict(self._cache[key])
        # Fuzzy fallback (PT/LLT only).
        cutoff = int(os.environ.get("GROUND_FUZZY_CUTOFF", "90"))
        hit = process.extractOne(key, self._fuzzy_keys, scorer=fuzz.ratio,
                                 score_cutoff=float(cutoff))
        if hit:
            return dict(self._fuzzy_vals[hit[2]])
        return None

    def is_ambiguous(self, fr_term: str) -> bool:
        """True when multiple distinct MedDRA concepts share this French label."""
        self._load()
        return _norm(fr_term) in self._ambiguous

    def neighbours(self, concept_id: str) -> list[str]:
        """Return English names of concepts within two hops in the hierarchy."""
        q = """
        MATCH (c:Concept) WHERE coalesce(c.id, c.name) = $cid
        OPTIONAL MATCH (p)-[:BROADER_THAN]->(c)
        OPTIONAL MATCH (c)-[:BROADER_THAN]->(ch)
        OPTIONAL MATCH (p)-[:BROADER_THAN]->(sib) WHERE sib <> c
        WITH collect(DISTINCT p.name) + collect(DISTINCT ch.name)
           + collect(DISTINCT sib.name) AS names
        RETURN [x IN names WHERE x IS NOT NULL] AS names
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        return list(dict.fromkeys(rec["names"])) if rec else []

    def same_branch(self, a: str, b: str) -> bool:
        """True when concepts ``a`` and ``b`` are within 5 BROADER_THAN hops."""
        if a.strip().lower() == b.strip().lower():
            return True
        q = """
        MATCH (a:Concept),(b:Concept)
        WHERE toLower(a.name)=toLower($a) AND toLower(b.name)=toLower($b)
        OPTIONAL MATCH p = shortestPath((a)-[:BROADER_THAN*..5]-(b))
        RETURN p IS NOT NULL AS ok
        """
        with self._driver.session() as s:
            rec = s.run(q, a=a.strip(), b=b.strip()).single()
        return bool(rec and rec["ok"])

    def get_by_name(self, en_name: str) -> dict | None:
        """Look up a concept by its English name; returns None when not found."""
        q = """
        MATCH (c:Concept) WHERE toLower(c.name) = toLower($n)
        RETURN coalesce(c.id,c.name) AS id, c.name AS name,
               c.level AS level, c.tier AS tier LIMIT 1
        """
        with self._driver.session() as s:
            rec = s.run(q, n=en_name.strip()).single()
        return dict(rec) if rec else None

    def fetch_hierarchy_for_concept(self, concept_id: str) -> dict:
        """Return the full SOC-to-LLT ancestor chain for ``concept_id``."""
        q = """
        MATCH (c:Concept) WHERE coalesce(c.id, c.name) = $cid
        OPTIONAL MATCH chain = (c)-[:BROADER_THAN*..10]->(ancestor)
        WITH c, collect(ancestor) AS ancestors
        RETURN c, ancestors
        """
        with self._driver.session() as s:
            rec = s.run(q, cid=concept_id).single()
        if not rec:
            return {"chain": []}
        chain = [{"id": str(a.get("id") or a.get("name") or ""),
                  "name": str(a.get("name") or "")}
                 for a in (rec["ancestors"] or [])]
        return {"chain": chain}

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Global terminology planning ────────────────────────────────────────────
# Before translation, we commit each French NER term to one canonical English
# rendering.  This prevents the same term appearing as both "nausea" and
# "Nausea" within a single document.

_EMBED: SentenceTransformer | None = None


def _embedder() -> SentenceTransformer:
    global _EMBED
    if _EMBED is None:
        mid = os.environ.get("TERMPLAN_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        _EMBED = SentenceTransformer(mid)
    return _EMBED


def compute_global_locks(
    segments: list[dict],
    graph: TermGraph,
    *,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> dict[str, str]:
    """Map each French NER surface form to a single canonical English rendering.

    Scores each candidate rendering by:
    - ``alpha``: contextual similarity (how well the English fits the French sentences).
    - ``beta``:  neighbourhood coherence (similarity to sibling concepts).
    - ``gamma``: hierarchy penalty (prefer PT-level over LLT/SOC).
    """
    model = _embedder()

    # Collect all unique French terms and the segment indices where they appear.
    term_idx: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        for t in seg.get("terms") or []:
            w = (t.get("word") or "").strip()
            if w:
                term_idx.setdefault(w, []).append(i)

    locks: dict[str, str] = {}
    for fr_term, idxs in term_idx.items():
        concept = graph.ground(fr_term)
        if not concept or graph.is_ambiguous(fr_term):
            continue  # Skip terms with no clear grounding.

        cands = [concept["name"]] + graph.neighbours(str(concept["id"]))[:16]
        if not cands:
            continue

        src_sents = [segments[j]["fr"] for j in idxs]
        emb_src = model.encode(src_sents, convert_to_tensor=True)
        src_lvl = concept.get("level") or 0

        best, best_score = cands[0], float("-inf")
        for cand in cands:
            emb_c = model.encode(cand, convert_to_tensor=True)
            ctx = float(util.cos_sim(emb_src, emb_c.unsqueeze(0)).squeeze(-1).mean())
            nbrs = graph.neighbours(str(concept["id"]))[:16]
            nb = float(util.cos_sim(
                emb_c.unsqueeze(0),
                model.encode(nbrs, convert_to_tensor=True),
            ).mean()) if nbrs else 0.0
            node = graph.get_by_name(cand)
            hp = abs(int(node["level"]) - int(src_lvl)) if node else 0
            score = alpha * ctx + beta * nb - gamma * float(hp)
            if score > best_score:
                best, best_score = cand, score
        locks[fr_term] = best

    return locks


def load_or_compute_locks(
    segments_path: Path,
    graph: TermGraph,
    *,
    cache_path: Path | None = None,
    recompute: bool = False,
) -> dict[str, str]:
    """Load planning locks from a JSON cache, or compute and save them.

    The cache is invalidated automatically when the segments file is modified.
    Pass ``recompute=True`` to force a fresh run.
    """
    cache = cache_path or segments_path.parent / "planning_locks.json"
    if cache.is_file() and not recompute:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k != "_mtime"}
        except (json.JSONDecodeError, OSError):
            pass

    rows = [json.loads(ln) for ln in segments_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    locks = compute_global_locks(rows, graph)
    try:
        cache.write_text(
            json.dumps({"_mtime": os.path.getmtime(segments_path), **locks},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    return locks
