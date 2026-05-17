"""OpenRouter context routing (OpenAI-compatible gateway, rolling model id)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# Rolling production id — avoids pinned variants deprecated on OpenRouter (e.g. v0.1).
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = """You are a medical translation and pharmacovigilance coding specialist \
(TermPlanMT). Use British English in all prose fields.

Tasks:
1. Pick the best MedDRA concept from the candidates using the full context sentence.
2. Briefly note register differences between official MedDRA wording and natural clinical prose.

Prefer the candidate whose hierarchy fits treatment context, timing, and pathology in the sentence.

Respond ONLY with valid JSON. selected_concept_id MUST be one of the candidate IDs listed."""

_JSON_ID_RE = re.compile(r"^\d+$")
_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _env(name: str, *legacy_names: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    for leg in legacy_names:
        val = os.environ.get(leg, "").strip()
        if val:
            return val
    return ""


def llm_configured() -> bool:
    return bool(_env("LLM_API_KEY", "OPENROUTER_API_KEY"))


def llm_model() -> str:
    return _env("LLM_MODEL_NAME", "OPENROUTER_MODEL") or DEFAULT_MODEL


def llm_base_url() -> str:
    return _env("LLM_API_BASE_URL") or DEFAULT_BASE_URL


def _client():
    from openai import OpenAI

    referer = _env("LLM_HTTP_REFERER", "OPENROUTER_REFERER") or (
        "https://github.com/VerbalAid/Term-Plan-MT"
    )
    title = os.environ.get("LLM_X_TITLE", "TermPlanMT-WebUI")

    return OpenAI(
        api_key=_env("LLM_API_KEY", "OPENROUTER_API_KEY"),
        base_url=llm_base_url(),
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": title,
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
    for idx, c in enumerate(candidates, start=1):
        lines.append(f"Candidate [{idx}]:")
        lines.append(f"  - MedDRA ID: {c.get('id')}")
        lines.append(f"  - English Preferred Term: {c.get('name', '')}")
        lines.append(f"  - French Grounded Label: {c.get('fr_label') or '—'}")
        lines.append(f"  - Tier: {c.get('tier', '')} | Level: {c.get('level', '—')}")
        path = c.get("ancestor_summary")
        if path:
            lines.append(f"  - Lineage to SOC: {path}")
        parents = c.get("parent_names") or []
        if parents:
            lines.append(f"  - Immediate parents: {', '.join(parents[:6])}")
        lines.append("")
    return "\n".join(lines)


def _valid_ids(candidates: list[dict[str, Any]]) -> set[str]:
    return {str(c["id"]) for c in candidates if _JSON_ID_RE.match(str(c.get("id", "")))}


def _fallback_result(
    candidates: list[dict[str, Any]],
    *,
    model: str,
    reason: str,
) -> dict[str, Any]:
    valid = _valid_ids(candidates)
    cid = next(iter(valid), None)
    return {
        "ok": True,
        "abstain": False,
        "fallback": True,
        "selected_concept_id": cid,
        "clinical_justification": reason,
        "stylistic_analysis": "Stylistic register analysis unavailable.",
        "confidence": "low",
        "model": model,
    }


def resolve_context(
    context_sentence: str,
    target_term: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call OpenRouter (Mistral 7B rolling id) to pick a concept and explain register."""
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
            "message": "Context routing is unavailable (LLM_API_KEY not set).",
        }

    valid = _valid_ids(candidates)
    model = llm_model()
    user_prompt = f"""Context sentence:
"{context_sentence.strip()}"

Target source term:
"{target_term.strip()}"

Grounded MedDRA graph candidates:
{_format_candidates(candidates)}

Return JSON with exactly these keys:
- "selected_concept_id": string, one of: {", ".join(sorted(valid))}
- "clinical_justification": string, 2–4 sentences, British English
- "stylistic_analysis": string, 2–4 sentences on register (MedDRA vs translator prose)
- "abstain": boolean, true only if no candidate is defensible
- "confidence": "high", "medium", or "low"
"""

    try:
        client = _client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
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
        return _fallback_result(
            candidates,
            model=model,
            reason=f"Invalid JSON from model; defaulted to top graph candidate. ({exc})",
        )
    except Exception as exc:
        log.exception("OpenRouter context resolve failed")
        return _fallback_result(
            candidates,
            model=model,
            reason=f"OpenRouter routing unavailable; defaulted to top graph candidate. ({exc})",
        )

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
        if valid:
            return _fallback_result(
                candidates,
                model=model,
                reason=(
                    f"Model returned id {selected!r} outside the candidate set; "
                    "defaulted to top graph candidate."
                ),
            )
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
