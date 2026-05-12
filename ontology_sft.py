"""Alpaca and Mistral-Instruct formatting for MedDRA NER supervised fine-tuning.

Used by the training data export scripts in tools/data/ to generate SFT corpora
from the Neo4j graph, and by the NER training scripts.
"""

from __future__ import annotations

import json
from typing import Any

from pipeline import canonical_meddra_tier

# ── Prompt instructions ────────────────────────────────────────────────────

ONTOLOGY_SFT_INSTRUCTION = (
    "Extract medical terms from the French medical text below. "
    "Return only a JSON list of objects. Each object must have keys: "
    '"fr" (surface form as in the text), '
    '"en" (English MedDRA preferred term), '
    '"level" (integer MedDRA hierarchy level), '
    '"tier" (string, e.g. PT or LLT), '
    '"id" (MedDRA concept id string). '
    "List each distinct medically relevant term once; omit terms you cannot anchor."
)

ONTOLOGY_SFT_INSTRUCTION_HIERARCHICAL = (
    "Extract medical terms from the French medical text below. "
    "Return only a JSON list of objects. Each object must have keys: "
    '"fr" (surface form as in the text), '
    '"en" (canonical English at MedDRA PT level when a PT exists on the branch), '
    '"en_resolved" (English string for the grounded node actually matched), '
    '"tier" (grounded node tier: SOC, HLGT, HLT, PT, or LLT), '
    '"id" (MedDRA concept id for the grounded node), '
    '"level" (integer hierarchy level 1–5 for the grounded node), '
    '"soc", "hlgt", "hlt", "pt", "llt" (English names on the primary MedDRA path; '
    "use null for levels narrower than the grounded node). "
    "List each distinct medically relevant term once; omit terms you cannot anchor."
)

# Backward-compatible aliases.
ONTOLOGY_SFT_ALPACA_INSTRUCTION = ONTOLOGY_SFT_INSTRUCTION
ONTOLOGY_SFT_ALPACA_INSTRUCTION_HIERARCHICAL = ONTOLOGY_SFT_INSTRUCTION_HIERARCHICAL

_TIER_ORDER = ["SOC", "HLGT", "HLT", "PT", "LLT"]
_TIER_RANK  = {t: i for i, t in enumerate(_TIER_ORDER)}


# ── Format converters ──────────────────────────────────────────────────────


def to_alpaca(fr_body: str, response_json: str) -> str:
    """Wrap a training example in Alpaca (### Instruction / Input / Response) format."""
    return (
        "### Instruction:\n"
        f"{ONTOLOGY_SFT_INSTRUCTION}\n\n"
        "### Input:\n"
        f"{fr_body}\n\n"
        "### Response:\n"
        f"{response_json}"
    )


def to_alpaca_hierarchical(fr_body: str, response_json: str) -> str:
    """Alpaca format using the hierarchical instruction (full SOC→LLT path)."""
    return (
        "### Instruction:\n"
        f"{ONTOLOGY_SFT_INSTRUCTION_HIERARCHICAL}\n\n"
        "### Input:\n"
        f"{fr_body}\n\n"
        "### Response:\n"
        f"{response_json}"
    )


def to_mistral_instruct(fr_body: str, response_json: str) -> str:
    """Mistral-7B-Instruct format: ``<s>[INST] user [/INST] assistant</s>``."""
    user      = ONTOLOGY_SFT_INSTRUCTION + "\n\n### Input:\n" + fr_body.strip()
    assistant = response_json.strip()
    return f"<s>[INST] {user} [/INST] {assistant}</s>"


def to_mistral_instruct_hierarchical(fr_body: str, response_json: str) -> str:
    """Mistral-Instruct format using the hierarchical instruction."""
    user      = ONTOLOGY_SFT_INSTRUCTION_HIERARCHICAL + "\n\n### Input:\n" + fr_body.strip()
    assistant = response_json.strip()
    return f"<s>[INST] {user} [/INST] {assistant}</s>"


# ── Row helpers ────────────────────────────────────────────────────────────


def concept_to_row(word_fr: str, concept: dict[str, Any]) -> dict[str, Any]:
    """Build a flat SFT row from a French surface form and a graph concept payload."""
    return {
        "fr":    word_fr.strip(),
        "en":    str(concept.get("name") or "").strip(),
        "level": concept.get("level"),
        "tier":  canonical_meddra_tier(concept),
        "id":    str(concept.get("id") or "").strip(),
    }


def row_payload_json(concept: dict[str, Any], *, fr_surface: str) -> str:
    """JSON string for a single flat SFT row."""
    return json.dumps([concept_to_row(fr_surface, concept)], ensure_ascii=False)


def hierarchy_flat_fields(
    grounded_tier: str,
    by_tier: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    """Return SOC/HLGT/HLT/PT/LLT name columns, with ``None`` below the grounded level."""
    gt     = str(grounded_tier or "").strip().upper()
    g_rank = _TIER_RANK.get(gt, len(_TIER_ORDER))
    out: dict[str, str | None] = {}
    for tk in _TIER_ORDER:
        rk = _TIER_RANK[tk]
        if rk > g_rank:
            out[tk.lower()] = None
            continue
        pl = by_tier.get(tk)
        nm = str(pl.get("name") or "").strip() if pl else ""
        out[tk.lower()] = nm if nm else None
    return out


def canonical_en(
    *,
    supervision_en: str,
    grounded: dict[str, Any],
    by_tier: dict[str, dict[str, Any]],
) -> str:
    """Primary loss target for the ``en`` field.

    When ``supervision_en == 'pt'``, the PT name is used (regulatory preferred term);
    otherwise the grounded node's own name is used.
    """
    if supervision_en.strip().lower() != "pt":
        return str(grounded.get("name") or "").strip()
    pt_pl = by_tier.get("PT")
    if pt_pl and str(pt_pl.get("name") or "").strip():
        return str(pt_pl.get("name")).strip()
    return str(grounded.get("name") or "").strip()


def concept_to_row_hierarchical(
    word_fr: str,
    grounded: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    supervision_en: str = "pt",
) -> dict[str, Any]:
    """Rich SFT row: PT-canonical ``en``, full SOC→LLT path, grounded tier/id/level."""
    by_tier    = hierarchy.get("by_tier") or {}
    gtier      = canonical_meddra_tier(grounded)
    flat       = hierarchy_flat_fields(gtier, by_tier if isinstance(by_tier, dict) else {})
    en_primary = canonical_en(supervision_en=supervision_en, grounded=grounded, by_tier=by_tier)
    row: dict[str, Any] = {
        "fr":          word_fr.strip(),
        "en":          en_primary,
        "en_resolved": str(grounded.get("name") or "").strip(),
        "level":       grounded.get("level"),
        "tier":        gtier,
        "id":          str(grounded.get("id") or "").strip(),
    }
    row.update(flat)
    return row


def row_payload_json_hierarchical(
    grounded: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    fr_surface: str,
    supervision_en: str = "pt",
) -> str:
    """JSON string for a single hierarchical SFT row."""
    return json.dumps(
        [concept_to_row_hierarchical(fr_surface, grounded, hierarchy, supervision_en=supervision_en)],
        ensure_ascii=False,
    )
