"""Neo4j TermGraph: ground French NER spans to MedDRA concepts (string, vector, or vector+LLM)."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from rapidfuzz import fuzz, process

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pipeline.cuda_ld_path import ensure_cuda_pip_libs_visible

ensure_cuda_pip_libs_visible()

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
VECTOR_INDEX_NAME = "meddra_fr_embedding"


def normalize_fr_for_grounding(text: str) -> str:
    """Fold French text for exact / fuzzy lookup keys."""
    s = unicodedata.normalize("NFKC", text.strip())
    s = s.casefold()
    return " ".join(s.split())


class TermGraph:
    """Neo4j graph + grounding modes; see ``ground()``."""

    GROUNDING_MODES = frozenset({"string", "vector", "vector_llm"})

    def __init__(
        self,
        grounding_mode: str = "string",
        *,
        vector_score_threshold: float = 0.75,
        vector_top_k: int = 5,
        vector_index_name: str = VECTOR_INDEX_NAME,
        embed_model_name: str | None = None,
        llm_model_id: str | None = None,
    ) -> None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASS", "password")
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

        mode = grounding_mode.strip().lower()
        if mode not in self.GROUNDING_MODES:
            raise ValueError(f"grounding_mode must be one of {sorted(self.GROUNDING_MODES)}, got {grounding_mode!r}")
        self.grounding_mode = mode
        self._vector_threshold = float(vector_score_threshold)
        self._vector_k = int(vector_top_k)
        self._vector_index_name = vector_index_name
        self._embed_model_name = embed_model_name or os.environ.get(
            "TERMPLAN_EMBED_MODEL",
            DEFAULT_EMBED_MODEL,
        )
        self._llm_model_id = llm_model_id or os.environ.get(
            "GROUND_LLM_MODEL",
            "BioMistral/BioMistral-7B",
        )

        self._fr_exact: dict[str, dict[str, Any]] | None = None
        self._fuzzy_norms: list[str] | None = None
        self._fuzzy_payloads: list[dict[str, Any]] | None = None
        self._ambiguous_norms: set[str] = set()
        self._ambiguous_n_concepts: dict[str, int] = {}

        self._embedder: Any | None = None
        self._llm_cache: tuple[Any, Any] | None = None

        self._ambiguous_lines: list[str] = []
        self._vector_rejected_spans: list[dict[str, Any]] = []
        self._hierarchy_cache: dict[str, dict[str, Any]] = {}
        self._hierarchy_index_loaded: bool = False
        self._hierarchy_payloads: dict[str, dict[str, Any]] | None = None
        self._hierarchy_parent_by_child: dict[str, str] | None = None
        self.reset_grounding_stats()

    def set_vector_threshold(self, value: float) -> None:
        """Set vector similarity cutoff (no driver rebuild)."""
        self._vector_threshold = float(value)

    def reset_grounding_stats(self) -> None:
        self._stats = {
            "calls": 0,
            "vector_fallbacks": 0,
            "string_ambiguous_warnings": 0,
            "mean_candidates_sum": 0.0,
        }
        self._ambiguous_lines.clear()
        self._vector_rejected_spans.clear()

    def get_grounding_stats(self) -> dict[str, Any]:
        out = dict(self._stats)
        n = out["calls"]
        out["mean_candidates"] = (out["mean_candidates_sum"] / n) if n else 0.0
        del out["mean_candidates_sum"]
        return out

    def take_vector_rejected_spans(self) -> list[dict[str, Any]]:
        """Pop and return spans rejected by vector grounding."""
        out = list(self._vector_rejected_spans)
        self._vector_rejected_spans.clear()
        return out

    def _vector_query_embedding_text(self, fr_term: str, context: str | None) -> str:
        """Text sent to the embedder for vector index query (span-first, else context)."""
        span = (fr_term or "").strip()
        if span:
            return span
        ctx = (context or "").strip()
        return ctx if ctx else " "

    def _record_vector_reject(
        self,
        fr_term: str,
        context: str | None,
        *,
        reason: str,
        top_score: float | None,
        top_fr_label: str | None,
    ) -> None:
        ctx_snip = ((context or "").strip())[:240].replace("\n", " ")
        self._vector_rejected_spans.append(
            {
                "fr_term": (fr_term or "").strip(),
                "context_snippet": ctx_snip,
                "reason": reason,
                "top_score": top_score,
                "threshold": self._vector_threshold,
                "top_fr_label": top_fr_label,
            }
        )

    def write_ambiguous_report(self, path: Path) -> None:
        """Write ambiguous FR strings (tab-separated counts) to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        uniq = sorted(set(self._ambiguous_lines))
        path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")

    def _row_to_payload(self, rec: Any) -> dict[str, Any]:
        return {
            "id": rec["id"],
            "name": rec["name"],
            "level": rec["level"],
            "tier": rec["tier"],
        }

    def _load_grounding_cache(self) -> None:
        if self._fr_exact is not None:
            return
        q_all = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
        RETURN coalesce(c.id, c.name) AS id, c.name AS name, c.level AS level,
               c.tier AS tier, c.fr_label AS fr_label
        """
        q_pt_llt = """
        MATCH (c:Concept)
        WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''
          AND c.tier IN ['PT', 'LLT']
        RETURN coalesce(c.id, c.name) AS id, c.name AS name, c.level AS level,
               c.tier AS tier, c.fr_label AS fr_label
        """
        from collections import defaultdict

        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        with self._driver.session() as session:
            for rec in session.run(q_all):
                key = normalize_fr_for_grounding(rec["fr_label"])
                if not key:
                    continue
                payload = self._row_to_payload(rec)
                buckets[key].append(payload)

        exact: dict[str, dict[str, Any]] = {}
        self._ambiguous_norms.clear()
        self._ambiguous_n_concepts.clear()
        for key, items in buckets.items():
            ids = {str(p.get("id")) for p in items}
            self._ambiguous_n_concepts[key] = len(ids)
            if len(ids) > 1:
                self._ambiguous_norms.add(key)
            exact[key] = items[0]

        fuzzy_norms: list[str] = []
        fuzzy_payloads: list[dict[str, Any]] = []

        with self._driver.session() as session:
            for rec in session.run(q_pt_llt):
                key = normalize_fr_for_grounding(rec["fr_label"])
                if not key:
                    continue
                fuzzy_norms.append(key)
                fuzzy_payloads.append(self._row_to_payload(rec))

        self._fr_exact = exact
        self._fuzzy_norms = fuzzy_norms
        self._fuzzy_payloads = fuzzy_payloads

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            dev = os.environ.get("TERMPLAN_EMBED_DEVICE", "").strip()
            if dev:
                self._embedder = SentenceTransformer(self._embed_model_name, device=dev)
            else:
                self._embedder = SentenceTransformer(self._embed_model_name)
        return self._embedder

    def _get_llm(self) -> tuple[Any, Any]:
        if self._llm_cache is not None:
            return self._llm_cache
        import gc

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tok = AutoTokenizer.from_pretrained(self._llm_model_id, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        dm_raw = os.environ.get("BIOMISTRAL_DEVICE_MAP", os.environ.get("MISTRAL_DEVICE_MAP", "")).strip().lower()
        if dm_raw == "auto":
            device_map = "auto"
        elif torch.cuda.is_available():
            device_map = {"": 0}
        else:
            device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            self._llm_model_id,
            quantization_config=bnb,
            device_map=device_map,
        )
        self._llm_cache = (tok, model)
        return self._llm_cache

    def _embed_query_text(self, text: str) -> list[float]:
        model = self._get_embedder()
        v = model.encode(text.strip() or " ", convert_to_numpy=True)
        return [float(x) for x in v]

    def _clean_payload(self, p: dict[str, Any]) -> dict[str, Any]:
        out = dict(p)
        out.pop("fr_label", None)
        return out

    def _vector_candidates(self, query_embedding: list[float]) -> list[tuple[dict[str, Any], float]]:
        q = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node AS c, score
        RETURN coalesce(c.id, c.name) AS id, c.name AS name, c.level AS level,
               c.tier AS tier, c.fr_label AS fr_label, score
        """
        out: list[tuple[dict[str, Any], float]] = []
        with self._driver.session() as session:
            try:
                rows = session.run(
                    q,
                    index_name=self._vector_index_name,
                    k=self._vector_k,
                    embedding=query_embedding,
                )
                for rec in rows:
                    payload = {
                        "id": rec["id"],
                        "name": rec["name"],
                        "level": rec["level"],
                        "tier": rec["tier"],
                        "fr_label": rec["fr_label"],
                    }
                    out.append((payload, float(rec["score"])))
            except Exception as e:
                log.warning("Vector query failed (%s). Build embeddings + index first.", e)
        return out

    def _vector_fetch_candidates(self, fr_term: str, context: str | None) -> list[tuple[dict[str, Any], float]]:
        qtext = self._vector_query_embedding_text(fr_term, context)
        vec = self._embed_query_text(qtext)
        cands = self._vector_candidates(vec)
        self._stats["mean_candidates_sum"] += float(len(cands) if cands else self._vector_k)
        return cands

    def _ground_string(self, fr_term: str) -> dict[str, Any] | None:
        raw = fr_term.strip()
        if not raw:
            return None

        self._load_grounding_cache()
        assert self._fr_exact is not None and self._fuzzy_norms is not None and self._fuzzy_payloads is not None

        key = normalize_fr_for_grounding(raw)
        if key in self._ambiguous_norms:
            self._stats["string_ambiguous_warnings"] += 1
            n_c = self._ambiguous_n_concepts.get(key, 0)
            line = f"{raw}\t{n_c}"
            if line not in self._ambiguous_lines:
                self._ambiguous_lines.append(line)
            log.warning(
                "Ambiguous FR grounding: %r maps to %d distinct MedDRA concepts under normalized key.",
                raw,
                n_c,
            )

        if key in self._fr_exact:
            return dict(self._fr_exact[key])

        enable_fuzzy = os.environ.get("GROUND_ENABLE_FUZZY", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        if not enable_fuzzy or not self._fuzzy_norms:
            return None

        try:
            cutoff = int(os.environ.get("GROUND_FUZZY_CUTOFF", "90"))
        except ValueError:
            cutoff = 90
        cutoff = max(70, min(100, cutoff))

        hit = process.extractOne(
            key,
            self._fuzzy_norms,
            scorer=fuzz.ratio,
            score_cutoff=float(cutoff),
        )
        if not hit:
            return None
        _match, _score, idx = hit
        return dict(self._fuzzy_payloads[idx])

    def _ground_vector(self, fr_term: str, context: str | None) -> dict[str, Any] | None:
        cands = self._vector_fetch_candidates(fr_term, context)

        if not cands:
            self._stats["vector_fallbacks"] += 1
            self._record_vector_reject(
                fr_term, context, reason="no_vector_hits", top_score=None, top_fr_label=None
            )
            return None

        top_payload, top_score = cands[0]
        top_lab = str(top_payload.get("fr_label") or "") or None
        if top_score < self._vector_threshold:
            self._stats["vector_fallbacks"] += 1
            self._record_vector_reject(
                fr_term,
                context,
                reason="below_threshold",
                top_score=float(top_score),
                top_fr_label=top_lab,
            )
            return None
        return self._clean_payload(top_payload)

    def _parse_choice_1_to_5(self, raw: str) -> int | None:
        m = re.search(r"\b([1-5])\b", raw.strip())
        if not m:
            return None
        return int(m.group(1))

    def _ground_vector_llm(self, fr_term: str, context: str | None) -> dict[str, Any] | None:
        text_for_embed = (context or "").strip() or fr_term.strip()
        cands = self._vector_fetch_candidates(fr_term, context)

        if not cands:
            self._stats["vector_fallbacks"] += 1
            self._record_vector_reject(
                fr_term, context, reason="no_vector_hits", top_score=None, top_fr_label=None
            )
            return None

        top_payload, top_score = cands[0]
        top_lab = str(top_payload.get("fr_label") or "") or None
        payloads_scored = [(pl, sc) for pl, sc in cands if sc >= self._vector_threshold]
        if not payloads_scored:
            self._stats["vector_fallbacks"] += 1
            self._record_vector_reject(
                fr_term,
                context,
                reason="below_threshold",
                top_score=float(top_score),
                top_fr_label=top_lab,
            )
            return None

        payloads_scored = payloads_scored[: self._vector_k]

        lines = []
        for i, (pl, _sc) in enumerate(payloads_scored, start=1):
            fr_lab = str(pl.get("fr_label") or "")
            lvl = pl.get("level")
            lines.append(
                f"{i}. {pl.get('name')} (MedDRA level: {lvl}, French: {fr_lab})"
            )

        prompt = (
            "You are a medical terminology expert. Given a French medical sentence and a list of "
            "MedDRA concept candidates, select the single most appropriate concept for the highlighted term.\n\n"
            f"Sentence: {text_for_embed}\n"
            f"Term: {fr_term.strip()}\n\n"
            "Candidates:\n"
            + "\n".join(lines)
            + "\n\nReply with only the candidate number (1-5)."
        )

        tok, model = self._get_llm()
        messages = [{"role": "user", "content": prompt}]
        if getattr(tok, "chat_template", None):
            full = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full = prompt

        import torch

        inputs = tok(full, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=getattr(tok, "pad_token_id", None) or tok.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        gen = tok.decode(out_ids[0, in_len:], skip_special_tokens=True)
        choice = self._parse_choice_1_to_5(gen)
        if choice is None:
            log.debug("vector_llm: could not parse choice from %r; using candidate 1", gen[:200])
            choice = 1
        else:
            log.debug("vector_llm: model replied %r → choice %d", gen.strip(), choice)

        idx = max(0, min(len(payloads_scored) - 1, choice - 1))
        return self._clean_payload(payloads_scored[idx][0])

    def ground(self, fr_term: str, *, context: str | None = None) -> dict[str, Any] | None:
        """Return MedDRA payload for ``fr_term`` (uses ``context`` in vector modes)."""
        self._stats["calls"] += 1
        if self.grounding_mode == "string":
            self._stats["mean_candidates_sum"] += 1.0
            return self._ground_string(fr_term)
        if self.grounding_mode == "vector":
            return self._ground_vector(fr_term, context)
        return self._ground_vector_llm(fr_term, context)

    def get_by_name(self, en_name: str) -> dict[str, Any] | None:
        q = """
        MATCH (c:Concept)
        WHERE toLower(c.name) = toLower($name)
        RETURN coalesce(c.id, c.name) AS id, c.name AS name, c.level AS level, c.tier AS tier
        LIMIT 1
        """
        with self._driver.session() as session:
            rec = session.run(q, name=en_name.strip()).single()
        if not rec:
            return None
        return {
            "id": rec["id"],
            "name": rec["name"],
            "level": rec["level"],
            "tier": rec["tier"],
        }

    def neighbours(self, concept_id: str) -> list[str]:
        q = """
        MATCH (c:Concept)
        WHERE coalesce(c.id, c.name) = $cid
        OPTIONAL MATCH (p)-[:BROADER_THAN]->(c)
        OPTIONAL MATCH (c)-[:BROADER_THAN]->(ch)
        OPTIONAL MATCH (p)-[:BROADER_THAN]->(sib)
        WHERE sib <> c
        WITH collect(DISTINCT p.name) AS ps,
             collect(DISTINCT ch.name) AS cs,
             collect(DISTINCT sib.name) AS ss
        RETURN [x IN (ps + cs + ss) WHERE x IS NOT NULL] AS names
        """
        with self._driver.session() as session:
            rec = session.run(q, cid=concept_id).single()
        if not rec:
            return []
        return list(dict.fromkeys(rec["names"]))

    def candidate_renderings(self, concept: dict[str, Any], max_total: int = 32) -> list[str]:
        """English strings for planning: PT name plus graph neighbours."""
        names: list[str] = []
        if concept.get("name"):
            names.append(concept["name"])
        cid = concept.get("id")
        if cid is None:
            return names
        for nb in self.neighbours(str(cid)):
            if nb and nb not in names:
                names.append(nb)
            if len(names) >= max_total:
                break
        return names

    def list_concept_names(self) -> list[str]:
        q = "MATCH (c:Concept) WHERE c.name IS NOT NULL RETURN DISTINCT c.name AS name"
        with self._driver.session() as session:
            rows = session.run(q).values()
        names = [r[0] for r in rows if r[0]]
        names.sort(key=len, reverse=True)
        return names

    def same_branch(self, name_a: str, name_b: str) -> bool:
        if name_a.strip().lower() == name_b.strip().lower():
            return True
        q = """
        MATCH (a:Concept), (b:Concept)
        WHERE toLower(a.name) = toLower($na) AND toLower(b.name) = toLower($nb)
        OPTIONAL MATCH p = shortestPath((a)-[:BROADER_THAN*..5]-(b))
        RETURN p IS NOT NULL AS ok
        """
        with self._driver.session() as session:
            rec = session.run(q, na=name_a.strip(), nb=name_b.strip()).single()
        return bool(rec and rec["ok"])

    @staticmethod
    def _neo_concept_payload(node: Any) -> dict[str, Any]:
        return {
            "id": node.get("id"),
            "name": node.get("name"),
            "level": node.get("level"),
            "tier": node.get("tier"),
        }

    def preload_hierarchy_index(self) -> None:
        """Load every ``:Concept`` payload plus ``BROADER_THAN`` child→parent edges once.

        After this, ``fetch_hierarchy_for_concept`` resolves chains in memory (two Neo4j scans total),
        which is required for full-ontology exports with tens of thousands of concepts.
        """
        if self._hierarchy_index_loaded:
            return
        payloads: dict[str, dict[str, Any]] = {}
        parent_by_child: dict[str, str] = {}
        q_nodes = """
        MATCH (c:Concept)
        WHERE c.id IS NOT NULL AND trim(toString(c.id)) <> ''
        RETURN c.id AS id, c.name AS name, c.level AS level, c.tier AS tier
        """
        q_edges = """
        MATCH (p:Concept)-[:BROADER_THAN]->(c:Concept)
        WHERE c.id IS NOT NULL AND p.id IS NOT NULL
        RETURN toString(c.id) AS child, toString(p.id) AS parent
        """
        with self._driver.session() as session:
            for rec in session.run(q_nodes):
                cid = str(rec["id"]).strip()
                if cid:
                    payloads[cid] = {
                        "id": cid,
                        "name": rec["name"],
                        "level": rec["level"],
                        "tier": rec["tier"],
                    }
            for rec in session.run(q_edges):
                ch = str(rec["child"]).strip()
                pa = str(rec["parent"]).strip()
                if ch and pa:
                    parent_by_child[ch] = pa
        self._hierarchy_payloads = payloads
        self._hierarchy_parent_by_child = parent_by_child
        self._hierarchy_index_loaded = True

    def _hierarchy_from_index(self, cid: str) -> dict[str, Any]:
        """Build chain via parent pointers (requires ``preload_hierarchy_index``)."""
        payloads = self._hierarchy_payloads or {}
        parent_of = self._hierarchy_parent_by_child or {}
        if cid not in payloads:
            return {"chain": [], "by_tier": {}}
        up_ids: list[str] = []
        cur: str | None = cid
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            up_ids.append(cur)
            cur = parent_of.get(cur)
        chain_ids = list(reversed(up_ids))
        chain: list[dict[str, Any]] = []
        for i in chain_ids:
            pl = payloads.get(i)
            if pl:
                chain.append(dict(pl))
        by_tier: dict[str, dict[str, Any]] = {}
        for pl in chain:
            t = str(pl.get("tier") or "").strip().upper()
            if t:
                by_tier[t] = pl
        return {"chain": chain, "by_tier": by_tier}

    def fetch_hierarchy_for_concept(self, concept_id: str, *, use_cache: bool = True) -> dict[str, Any]:
        """Primary MedDRA branch SOC→…→concept for ``concept_id`` (``:Concept.id``).

        Returns ``{"chain": list[payload], "by_tier": dict[tier_code, payload]}`` where each payload
        has keys ``id``, ``name``, ``level``, ``tier``. ``chain`` is ordered broad→narrow.

        Uses the longest SOC-rooted ``BROADER_THAN`` path when multiple matches exist (MedDRA tree
        should yield a single branch). Results are cached per id for export scripts.
        """
        cid = str(concept_id or "").strip()
        if not cid:
            return {"chain": [], "by_tier": {}}
        if use_cache and cid in self._hierarchy_cache:
            return dict(self._hierarchy_cache[cid])
        if self._hierarchy_index_loaded:
            out = self._hierarchy_from_index(cid)
            self._hierarchy_cache[cid] = out
            return dict(out)

        q = """
        MATCH (anchor:Concept {id: $cid})
        OPTIONAL MATCH path = (soc:Concept)-[:BROADER_THAN*0..14]->(anchor)
        WHERE soc.tier = 'SOC'
        WITH anchor, path
        ORDER BY coalesce(length(path), -1) DESC
        LIMIT 1
        RETURN anchor, path
        """
        chain: list[dict[str, Any]] = []
        with self._driver.session() as session:
            rec = session.run(q, cid=cid).single()
        if not rec:
            out = {"chain": [], "by_tier": {}}
            self._hierarchy_cache[cid] = out
            return out

        anchor = rec["anchor"]
        path = rec["path"]
        if path is None:
            chain = [self._neo_concept_payload(anchor)]
        else:
            try:
                nodes = list(path.nodes)
            except Exception:
                nodes = []
            chain = [self._neo_concept_payload(n) for n in nodes]

        by_tier: dict[str, dict[str, Any]] = {}
        for pl in chain:
            t = str(pl.get("tier") or "").strip().upper()
            if t:
                by_tier[t] = pl

        out = {"chain": chain, "by_tier": by_tier}
        self._hierarchy_cache[cid] = out
        return out

    def fetch_concepts_with_fr_labels(self, *, tiers: list[str] | None = None) -> list[dict[str, Any]]:
        """Return ``:Concept`` rows with French labels (for full-ontology SFT export).

        Each dict has keys ``id``, ``name``, ``level``, ``tier``, ``fr_label`` (Neo4j field names).
        """
        params: dict[str, Any] = {}
        tier_clause = ""
        if tiers:
            tier_clause = " AND c.tier IN $tiers"
            params["tiers"] = list(tiers)
        q = (
            "MATCH (c:Concept)\n"
            "WHERE c.fr_label IS NOT NULL AND trim(c.fr_label) <> ''\n"
            "  AND c.name IS NOT NULL AND trim(c.name) <> ''\n"
            "  AND c.level IS NOT NULL\n"
            f"{tier_clause}\n"
            "RETURN coalesce(c.id, c.name) AS id, c.name AS name, c.level AS level, "
            "c.tier AS tier, c.fr_label AS fr_label"
        )
        rows: list[dict[str, Any]] = []
        with self._driver.session() as session:
            for rec in session.run(q, **params):
                rows.append(
                    {
                        "id": rec["id"],
                        "name": rec["name"],
                        "level": rec["level"],
                        "tier": rec["tier"],
                        "fr_label": rec["fr_label"],
                    }
                )
        return rows

    def close(self) -> None:
        self._hierarchy_cache.clear()
        self._hierarchy_index_loaded = False
        self._hierarchy_payloads = None
        self._hierarchy_parent_by_child = None
        self._driver.close()
        emb = self._embedder
        self._embedder = None
        llm = self._llm_cache
        self._llm_cache = None
        try:
            del emb
            del llm
        except Exception:
            pass
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
