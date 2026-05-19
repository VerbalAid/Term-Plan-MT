"""OpenRouter context routing (OpenAI-compatible gateway, rolling model id)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# Free-first on OpenRouter; set LLM_MODEL_NAME=openai/gpt-4o-mini in .env for paid routing.
DEFAULT_MODEL = "openrouter/free"
DEFAULT_FALLBACK_MODELS = (
    "openrouter/free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "openai/gpt-4o-mini",
)
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


def llm_model_chain() -> list[str]:
    """Primary model first, then env and built-in fallbacks (deduped)."""
    chain: list[str] = []
    extra = _env("LLM_MODEL_FALLBACKS")
    for mid in (llm_model(), *extra.split(","), *DEFAULT_FALLBACK_MODELS):
        m = mid.strip()
        if m and m not in chain:
            chain.append(m)
    return chain or [DEFAULT_MODEL]


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
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
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


def _is_model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "404" in text
        or "no endpoints found" in text
        or "model not found" in text
        or "does not exist" in text
    )


def _chat_completion(
    client: Any,
    base_kwargs: dict[str, Any],
    model: str,
) -> Any:
    """Call one model; retry without response_format if the gateway rejects it."""
    kwargs = {**base_kwargs, "model": model}
    try:
        kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        if _is_model_unavailable(exc):
            raise
        kwargs.pop("response_format", None)
        return client.chat.completions.create(**kwargs)


def _complete_json(
    client: Any,
    base_kwargs: dict[str, Any],
    models: list[str],
) -> tuple[dict[str, Any], str]:
    """Try each model until JSON parses; skip models that 404 or return garbage."""
    last_json_err: json.JSONDecodeError | None = None
    last_exc: BaseException | None = None
    for model in models:
        try:
            response = _chat_completion(client, base_kwargs, model)
            raw = response.choices[0].message.content or "{}"
            return _parse_json_payload(raw), model
        except json.JSONDecodeError as exc:
            log.warning("Model %s returned invalid JSON: %s", model, exc)
            last_json_err = exc
            continue
        except Exception as exc:
            if _is_model_unavailable(exc):
                log.warning("OpenRouter model unavailable (%s): %s", model, exc)
                last_exc = exc
                continue
            raise
    if last_json_err is not None:
        raise last_json_err
    assert last_exc is not None
    raise last_exc


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
    models = llm_model_chain()
    user_prompt = f"""Context sentence:
"{context_sentence.strip()}"

Target source term:
"{target_term.strip()}"

Grounded MedDRA graph candidates:
{_format_candidates(candidates)}

Return JSON with exactly these keys:
- "selected_concept_id": string, one of: {", ".join(sorted(valid))}
- "clinical_justification": string, 1–2 sentences, British English
- "stylistic_analysis": string, 1–2 sentences on register (MedDRA vs translator prose)
- "abstain": boolean, true only if no candidate is defensible
- "confidence": "high", "medium", or "low"
"""

    model = models[0]
    try:
        client = _client()
        base_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        parsed, model = _complete_json(client, base_kwargs, models)
    except json.JSONDecodeError:
        return _fallback_result(
            candidates,
            model=model,
            reason="Context routing returned invalid JSON; the top graph match was used instead.",
        )
    except Exception:
        log.exception("OpenRouter context resolve failed")
        return _fallback_result(
            candidates,
            model=model,
            reason=(
                "Context routing could not reach any configured OpenRouter model "
                f"({', '.join(models[:2])}). The top graph match was used instead."
            ),
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
