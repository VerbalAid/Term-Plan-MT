#!/usr/bin/env python3
"""Extract SmPC Section 4.8 from Keytruda PDFs, align FR/EN by sentence, save JSONL.

Default PDFs under ``data/test_data/`` yield **127** sentence-paired segments (``48_001`` … ``48_127``):
regex slice §4.8, NLTK Punkt tokenisation per language, then row *i* = FR sentence *i* + EN
sentence *i* (``n = min(len(fr_sents), len(en_sents))``). Optional CamemBERT NER fills ``terms``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber
import nltk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FR_START = re.compile(r"4\.8\s+Effets ind", re.IGNORECASE)
EN_START = re.compile(r"4\.8\s+Undesirable", re.IGNORECASE)
SECTION_END = re.compile(r"4\.9")


def _ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def extract_section_48(text: str, lang: str) -> str:
    pat = FR_START if lang == "fr" else EN_START
    m = None
    for m in pat.finditer(text):
        pass
    if not m:
        raise ValueError(f"Section 4.8 heading not found for lang={lang}")
    tail = text[m.start() :]
    m_end = SECTION_END.search(tail[1:])
    if m_end:
        return tail[: m_end.start() + 1].strip()
    return tail.strip()


def pdf_to_text(pdf_path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fr-pdf",
        type=Path,
        default=ROOT / "data" / "test_data" / "keytruda_fr.pdf",
    )
    parser.add_argument(
        "--en-pdf",
        type=Path,
        default=ROOT / "data" / "test_data" / "keytruda_en.pdf",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "section48" / "segments_ner.jsonl",
    )
    parser.add_argument(
        "--ner-model",
        type=str,
        default="Jean-Baptiste/camembert-ner",
        help=(
            "Hugging Face token-classification checkpoint for pipeline('ner'). "
            "Do not use a base LM (e.g. almanach/camembert-bio-base): it has no trained NER head."
        ),
    )
    # Optional local checkpoint after external fine-tune, e.g. --ner-model models/camembert-quaero-ner
    parser.add_argument(
        "--min-ner-score",
        type=float,
        default=0.80,
        help="Discard NER spans at or below this confidence (default 0.80).",
    )
    parser.add_argument(
        "--reuse-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Skip PDFs: read existing segment rows (id, fr, en_ref), re-run NER on fr only, "
            "write --out. Use when PDFs are unavailable but segments_ner.jsonl already exists."
        ),
    )
    args = parser.parse_args()

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "ner",
        model=args.ner_model,
        aggregation_strategy="max",
        device=device,
    )

    def ner_terms(fr: str) -> list[dict]:
        raw = ner(fr)
        terms: list[dict] = []
        for ent in raw:
            sc = float(ent.get("score", 0.0))
            if sc <= args.min_ner_score:
                continue
            word = ent.get("word", "").strip()
            if not word:
                continue
            terms.append(
                {
                    "word": word,
                    "entity": ent.get("entity_group", ent.get("entity", "")),
                    "score": sc,
                }
            )
        return terms

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_jsonl is not None:
        inp = args.reuse_jsonl
        if not inp.is_file():
            raise SystemExit(f"--reuse-jsonl file not found: {inp}")
        lines = [ln.strip() for ln in inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = [json.loads(line) for line in lines]
        n = 0
        with args.out.open("w", encoding="utf-8") as out_f:
            for row in rows:
                fr = row.get("fr") or ""
                row["terms"] = ner_terms(fr)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"Re-ran NER on {n} rows → {args.out}")
        return

    _ensure_nltk_punkt()

    fr_text = extract_section_48(pdf_to_text(args.fr_pdf), "fr")
    en_text = extract_section_48(pdf_to_text(args.en_pdf), "en")

    fr_sents = nltk.sent_tokenize(fr_text, language="french")
    en_sents = nltk.sent_tokenize(en_text, language="english")
    n = min(len(fr_sents), len(en_sents))

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(n):
            fr = fr_sents[i].strip()
            en_ref = en_sents[i].strip()
            row = {
                "id": f"48_{i + 1:03d}",
                "fr": fr,
                "en_ref": en_ref,
                "terms": ner_terms(fr),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {n} aligned segments to {args.out}")


if __name__ == "__main__":
    main()
