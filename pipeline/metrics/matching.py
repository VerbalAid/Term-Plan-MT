"""Normalized substring checks for terminology metrics (hyphens, spacing, unicode)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2212":
        s = s.replace(ch, "-")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def phrase_in_text(text: str, phrase: str) -> bool:
    """True if normalized ``phrase`` is a substring of normalized ``text``."""
    if not phrase:
        return False
    return normalize_text(phrase) in normalize_text(text)


def phrase_in_hyp(hyp: str, phrase: str) -> bool:
    """Backward-compatible name for :func:`phrase_in_text` (hypothesis string)."""
    return phrase_in_text(hyp, phrase)


def all_renderings(spec: dict[str, Any]) -> list[str]:
    """English surface strings from a term spec dict (``en_label`` + ``en_aliases``)."""
    out: list[str] = []
    if spec.get("en_label"):
        out.append(str(spec["en_label"]))
    for a in spec.get("en_aliases") or []:
        if a:
            out.append(str(a))
    return out


def canonical_rendering_if_match(hyp: str, spec: dict[str, Any]) -> str | None:
    """If ``hyp`` contains a rendering from ``spec``, return ``en_label``; else ``None``."""
    for r in all_renderings(spec):
        if phrase_in_hyp(hyp, r):
            return str(spec["en_label"])
    return None
