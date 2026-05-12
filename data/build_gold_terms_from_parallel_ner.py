#!/usr/bin/env python3
"""Draft FR→EN glossary from segment JSONL (reference n-gram counts).

For each French NER ``terms[].word``, we first pick **one** draft English n-gram
per segment from ``en_ref`` (longest non-trivial span), then count how often
each distinct FR key in that segment co-occurs with that span across the corpus.
The most frequent span per FR key becomes the glossary entry.

**Not clinical-grade:** use as a starting list, then hand-verify before S6
oracle experiments.

Example::

    PYTHONPATH=. python data/build_gold_terms_from_parallel_ner.py \\
        --segments data/section48/segments_ner_biollm.jsonl \\
        --out data/section48/gold_glossary.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def normalise_fr(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip()).casefold()
    return " ".join(s.split())


def _best_phrase_for_segment_reference(en_ref: str) -> str | None:
    """One English n-gram per segment (longest non-stopword-heavy span).

    Not aligned to individual French terms — callers attribute the same span
    to each distinct FR term in the segment once (weak draft heuristic).
    """
    en_tokens = re.findall(r"[a-zA-Z\-]+", en_ref.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "and",
        "or",
        "for",
        "with",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "from",
        "on",
        "as",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
        "than",
        "then",
        "when",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "how",
        "if",
        "each",
        "all",
        "any",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "patient",
        "patients",
        "treatment",
        "study",
    }
    for n in (5, 4, 3, 2):
        for i in range(len(en_tokens) - n + 1):
            chunk_tokens = en_tokens[i : i + n]
            if all(w in stopwords for w in chunk_tokens):
                continue
            chunk = " ".join(chunk_tokens)
            if len(chunk) >= 6:
                return chunk
    return None


def extract_en_renderings_from_segments(
    segments: list[dict],
) -> dict[str, Counter]:
    """For each distinct FR term key, count how often a segment-level ref phrase was chosen.

    Each segment contributes **one** draft English phrase (the longest qualifying
    n-gram in ``en_ref``).  That phrase is then counted once per **distinct**
    ``fr_key`` appearing among ``terms`` in that segment (avoids every term
    inheriting all n-grams of the sentence).
    """
    fr_to_en_counts: dict[str, Counter] = defaultdict(Counter)

    for seg in segments:
        en_ref = (seg.get("en_ref") or "").strip()
        if not en_ref:
            continue
        cand = _best_phrase_for_segment_reference(en_ref)
        if not cand:
            continue
        seen: set[str] = set()
        for t in seg.get("terms") or []:
            fr_raw = (t.get("word") or "").strip()
            if not fr_raw:
                continue
            fr_key = normalise_fr(fr_raw)
            if not fr_key or fr_key in seen:
                continue
            seen.add(fr_key)
            fr_to_en_counts[fr_key][cand] += 1

    return fr_to_en_counts


def build_glossary_from_parallel(
    segments: list[dict],
    *,
    min_count: int = 1,
    top_n: int = 1,
) -> list[dict]:
    fr_to_en_counts = extract_en_renderings_from_segments(segments)
    out: list[dict] = []
    for fr_key, counter in sorted(fr_to_en_counts.items(), key=lambda x: -sum(x[1].values())):
        for en, count in counter.most_common(top_n):
            if count >= min_count:
                out.append({"fr": fr_key, "en": en, "count": count})
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--segments",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner_biollm.jsonl",
        help="Segment JSONL with terms[] and en_ref.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "section48" / "gold_glossary.json",
        help="Output glossary JSON (list of {fr, en, count}).",
    )
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=1)
    args = ap.parse_args()

    seg_path = args.segments if args.segments.is_absolute() else ROOT / args.segments
    if not seg_path.is_file():
        raise SystemExit(f"Segments file not found: {seg_path}")

    segments: list[dict] = []
    with seg_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                segments.append(json.loads(s))

    print(f"Loaded {len(segments)} segments from {seg_path}")
    glossary = build_glossary_from_parallel(segments, min_count=args.min_count, top_n=args.top_n)

    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(glossary)} glossary entries → {out_path}")
    print("\nTop 10 entries:")
    for e in glossary[:10]:
        print(f"  {e['fr']!r:40s} → {e['en']!r:40s}  (count={e['count']})")
    print(
        "\nNOTE: Heuristic extraction from the reference. "
        "Manual review is required before trusting S6 as an oracle."
    )


if __name__ == "__main__":
    main()
