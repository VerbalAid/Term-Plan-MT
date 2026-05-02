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


def phrase_in_hyp(hyp: str, phrase: str) -> bool:
    if not phrase:
        return False
    return normalize_text(phrase) in normalize_text(hyp)


def all_renderings(gold: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if gold.get("en_label"):
        out.append(str(gold["en_label"]))
    for a in gold.get("en_aliases") or []:
        if a:
            out.append(str(a))
    return out


def canonical_rendering_if_match(hyp: str, gold: dict[str, Any]) -> str | None:
    """If ``hyp`` contains a gold rendering, return ``en_label``; else ``None``."""
    for r in all_renderings(gold):
        if phrase_in_hyp(hyp, r):
            return str(gold["en_label"])
    return None
