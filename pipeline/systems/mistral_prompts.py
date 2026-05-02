"""Instruction prompts for Mistral (GraphRAG and full-document modes)."""

from __future__ import annotations

import os
from typing import Any

from pipeline.graph import TermGraph


def medra_context_lines(graph: TermGraph, terms: list[dict], *, source_sentence: str = "") -> list[str]:
    lines: list[str] = []
    ctx = source_sentence.strip() or None
    for term in terms:
        concept = graph.ground(term.get("word", ""), context=ctx)
        if not concept:
            continue
        w = term.get("word", "")
        lines.append(
            f"  '{w}' → '{concept['name']}' (MedDRA {concept['tier']} L{concept['level']})"
        )
    return lines


def medra_lines_with_locks(
    graph: TermGraph,
    seg: dict[str, Any],
    locks: dict[str, str] | None,
) -> list[str]:
    """One line per grounded NER span: French → English (Stage-3 lock or graph PT name)."""
    lines: list[str] = []
    fr_ctx = (seg.get("fr") or "").strip() or None
    for term in seg.get("terms") or []:
        w = (term.get("word") or "").strip()
        if not w:
            continue
        concept = graph.ground(w, context=fr_ctx)
        if not concept:
            continue
        en = (locks or {}).get(w) or concept["name"]
        lines.append(
            f"  '{w}' → '{en}' (MedDRA {concept['tier']} L{concept['level']})"
        )
    return lines


def build_mistral_prompt(fr_text: str, context_lines: list[str]) -> str:
    system = (
        "You are a medical translator for EMA regulatory documents. "
        "The MedDRA context below gives the exact preferred English rendering "
        "for each French term. Use these renderings exactly. Do not generalise. "
        "Return only the translation."
    )
    ctx = "\n".join(context_lines) if context_lines else "(no grounded MedDRA terms)"
    user = f"MedDRA context:\n{ctx}\n\nTranslate:\n{fr_text}"
    return "<s>[INST] " + system + "\n\n" + user + " [/INST]"


def build_mistral_full_document_prompt(
    full_section_fr: str,
    target_fr: str,
    context_lines: list[str],
) -> str:
    system = (
        "You are a medical translator for EMA regulatory documents (SmPC). "
        "You are given the full French text of Section 4.8 for document-level context, "
        "plus optional MedDRA preferred English renderings. "
        "Translate ONLY the TARGET sentence into English. "
        "Return only that English sentence, nothing else."
    )
    ctx = "\n".join(context_lines) if context_lines else "(no grounded MedDRA terms for this sentence)"
    user = (
        f"FULL SECTION (French, for context only — do not translate this block as output):\n{full_section_fr}\n\n"
        f"MedDRA hints for the target sentence:\n{ctx}\n\n"
        f"TARGET sentence to translate:\n{target_fr}"
    )
    return "<s>[INST] " + system + "\n\n" + user + " [/INST]"


def truncate_full_section_for_token_budget(
    full_section_fr: str,
    target_fr: str,
    context_lines: list[str],
    tokenizer,
    max_input_tokens: int | None = None,
) -> str:
    """Trim the document block from the left until the S2 prompt fits in max_input_tokens.

    Long Section 4.8 texts can exceed GPU memory during Mistral prefill on 8GB cards.
    Override with env MISTRAL_MAX_INPUT_TOKENS (default 6144).
    """
    if max_input_tokens is None:
        raw = os.environ.get("MISTRAL_MAX_INPUT_TOKENS", "6144").strip()
        try:
            max_input_tokens = max(512, int(raw))
        except ValueError:
            max_input_tokens = 6144

    section = full_section_fr
    for _ in range(1000):
        prompt = build_mistral_full_document_prompt(section, target_fr, context_lines)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n <= max_input_tokens:
            return section
        drop = max(200, len(section) // 12)
        section = section[drop:].lstrip()
        if len(section) < 50:
            break
    return section
