"""Shared Alpaca formatting for MedDRA ontology NER SFT (segment or full-graph export)."""

from __future__ import annotations

import json
from typing import Any

from pipeline.meddra_io import canonical_meddra_tier

ONTOLOGY_SFT_ALPACA_INSTRUCTION = (
    "Extract medical terms from the French medical text below. "
    "Return only a JSON list of objects. Each object must have keys: "
    '"fr" (surface form as in the text), '
    '"en" (English MedDRA preferred term), '
    '"level" (integer MedDRA hierarchy level), '
    '"tier" (string, e.g. PT or LLT), '
    '"id" (MedDRA concept id string). '
    "List each distinct medically relevant term once; omit terms you cannot anchor."
)

# PT-level `en` by default, full SOC→LLT path, plus grounded-node string in `en_resolved`.
ONTOLOGY_SFT_ALPACA_INSTRUCTION_HIERARCHICAL = (
    "Extract medical terms from the French medical text below. "
    "Return only a JSON list of objects. Each object must have keys: "
    '"fr" (surface form as in the text), '
    '"en" (canonical English at MedDRA PT level when a PT exists on the branch — regulatory preferred term), '
    '"en_resolved" (English string for the grounded node actually matched: PT or LLT typically), '
    '"tier" (grounded node tier: SOC, HLGT, HLT, PT, or LLT), '
    '"id" (MedDRA concept id for the grounded node), '
    '"level" (integer hierarchy level 1–5 for the grounded node), '
    '"soc", "hlgt", "hlt", "pt", "llt" (English names on the primary MedDRA path from SOC downward; '
    "use null for levels narrower than the grounded node). "
    "List each distinct medically relevant term once; omit terms you cannot anchor."
)

_TIER_ORDER = ["SOC", "HLGT", "HLT", "PT", "LLT"]
_TIER_RANK = {t: i for i, t in enumerate(_TIER_ORDER)}


def to_alpaca(fr_body: str, response_json: str) -> str:
    return (
        "### Instruction:\n"
        f"{ONTOLOGY_SFT_ALPACA_INSTRUCTION}\n\n"
        "### Input:\n"
        f"{fr_body}\n\n"
        "### Response:\n"
        f"{response_json}"
    )


def to_alpaca_hierarchical(fr_body: str, response_json: str) -> str:
    return (
        "### Instruction:\n"
        f"{ONTOLOGY_SFT_ALPACA_INSTRUCTION_HIERARCHICAL}\n\n"
        "### Input:\n"
        f"{fr_body}\n\n"
        "### Response:\n"
        f"{response_json}"
    )


def to_mistral_instruct(fr_body: str, response_json: str) -> str:
    """Mistral-7B-Instruct style: user turn + assistant JSON (no bare Alpaca section headers)."""
    user = ONTOLOGY_SFT_ALPACA_INSTRUCTION + "\n\n### Input:\n" + fr_body.strip()
    assistant = response_json.strip()
    return f"<s>[INST] {user} [/INST] {assistant}</s>"


def to_mistral_instruct_hierarchical(fr_body: str, response_json: str) -> str:
    user = ONTOLOGY_SFT_ALPACA_INSTRUCTION_HIERARCHICAL + "\n\n### Input:\n" + fr_body.strip()
    assistant = response_json.strip()
    return f"<s>[INST] {user} [/INST] {assistant}</s>"


def concept_to_row(word_fr: str, concept: dict[str, Any]) -> dict[str, Any]:
    return {
        "fr": word_fr.strip(),
        "en": str(concept.get("name") or "").strip(),
        "level": concept.get("level"),
        "tier": canonical_meddra_tier(concept),
        "id": str(concept.get("id") or "").strip(),
    }


def row_payload_json(concept: dict[str, Any], *, fr_surface: str) -> str:
    return json.dumps([concept_to_row(fr_surface, concept)], ensure_ascii=False)


def hierarchy_flat_fields(
    grounded_tier: str,
    by_tier: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    """Map SOC..LLT English names; null for tiers strictly narrower than the grounded node."""
    gt = str(grounded_tier or "").strip().upper()
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
    """Primary loss target for `en`: PT name when supervision_en=='pt', else grounded name."""
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
    """Rich row: PT-canonical `en`, path columns, grounded tier/id/level, `en_resolved`."""
    by_tier = hierarchy.get("by_tier") or {}
    gtier = canonical_meddra_tier(grounded)
    flat = hierarchy_flat_fields(gtier, by_tier if isinstance(by_tier, dict) else {})
    en_primary = canonical_en(supervision_en=supervision_en, grounded=grounded, by_tier=by_tier)
    row: dict[str, Any] = {
        "fr": word_fr.strip(),
        "en": en_primary,
        "en_resolved": str(grounded.get("name") or "").strip(),
        "level": grounded.get("level"),
        "tier": gtier,
        "id": str(grounded.get("id") or "").strip(),
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
    return json.dumps(
        [concept_to_row_hierarchical(fr_surface, grounded, hierarchy, supervision_en=supervision_en)],
        ensure_ascii=False,
    )
