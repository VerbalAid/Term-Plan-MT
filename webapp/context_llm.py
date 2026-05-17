"""OpenRouter-hosted routing over MedDRA graph candidates (OpenAI-compatible API)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3-8b-instruct:free"

_SYSTEM_PROMPT = """You are a medical translation and pharmacovigilance coding assistant \
(TermPlanMT). Use British English in all prose fields.

Tasks:
1. Pick the best MedDRA concept from the candidates using the full context sentence.
2. Briefly note register differences between official MedDRA wording and natural clinical prose.

Prefer the candidate whose hierarchy fits treatment context, timing, and pathology in the sentence.

Respond ONLY with valid JSON. selected_concept_id MUST be one of the candidate IDs listed."""

_JSON_ID_RE = re.compile(r"^\d+$")
_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def llm_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def llm_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/VerbalAid/Term-Plan-MT"),
            "X-Title": "MedDRA Lookup",
        },
    )


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if match:
            return json.loads(match.group(0))
        raise


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for c in candidates:
        lines.append(
            f"- ID: {c['id']} | EN: {c.get('name', '')} | FR: {c.get('fr_label') or '—'} "
            f"| Tier: {c.get('tier', '')} | Level: {c.get('level', '—')} "
            f"| Source: {c.get('match_source', '')} | Score: {c.get('score', '—')}"
        )
        parents = c.get("parent_names") or []
        if parents:
            lines.append(f"  Parents: {', '.join(parents[:6])}")
        anc = c.get("ancestor_summary")
        if anc:
            lines.append(f"  Lineage: {anc}")
    return "\n".join(lines)


def _valid_ids(candidates: list[dict[str, Any]]) -> set[str]:
    return {str(c["id"]) for c in candidates if _JSON_ID_RE.match(str(c.get("id", "")))}


def resolve_context(
    context_sentence: str,
    target_term: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call OpenRouter to pick a concept and explain register nuances."""
    if not candidates:
        return {
            "ok": False,
            "error": "no_candidates",
            "message": "No graph candidates to disambiguate.",
        }
    if not llm_configured():
        return {
            "ok": False,
            "error": "llm_not_configured",
            "message": "Context routing is unavailable (OpenRouter not configured).",
        }

    valid = _valid_ids(candidates)
    user_prompt = f"""Context sentence:
"{context_sentence.strip()}"

Target term to ground in MedDRA:
"{target_term.strip()}"

Candidate MedDRA nodes:
{_format_candidates(candidates)}

Return JSON with exactly these keys:
- "selected_concept_id": string, one of: {", ".join(sorted(valid))}
- "clinical_justification": string, 2–4 sentences, British English
- "stylistic_analysis": string, 2–4 sentences on register (MedDRA vs translator prose)
- "abstain": boolean, true only if no candidate is defensible
- "confidence": "high", "medium", or "low"
"""

    model = llm_model()
    try:
        client = _client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content or "{}"
        parsed = _parse_json_payload(raw)
    except json.JSONDecodeError as exc:
        log.warning("Model returned invalid JSON: %s", exc)
        return {"ok": False, "error": "invalid_json", "message": str(exc)}
    except Exception as exc:
        log.exception("OpenRouter context resolve failed")
        return {"ok": False, "error": "llm_request_failed", "message": str(exc)}

    selected = str(parsed.get("selected_concept_id", "")).strip()
    abstain = bool(parsed.get("abstain", False))
    if abstain:
        return {
            "ok": True,
            "abstain": True,
            "selected_concept_id": None,
            "clinical_justification": parsed.get("clinical_justification", ""),
            "stylistic_analysis": parsed.get("stylistic_analysis", ""),
            "confidence": parsed.get("confidence", "low"),
            "model": model,
        }
    if selected not in valid:
        log.warning("Model selected unknown id %r; valid=%s", selected, valid)
        return {
            "ok": False,
            "error": "invalid_selection",
            "message": f"Model returned id {selected!r} outside the candidate set.",
            "raw": parsed,
        }

    return {
        "ok": True,
        "abstain": False,
        "selected_concept_id": selected,
        "clinical_justification": parsed.get("clinical_justification", ""),
        "stylistic_analysis": parsed.get("stylistic_analysis", ""),
        "confidence": parsed.get("confidence", "medium"),
        "model": model,
    }
