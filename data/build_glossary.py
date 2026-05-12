#!/usr/bin/env python3
"""
build_glossary.py — Glossary Constructor for TermPlanMT S6 Oracle Ablation

Extracts bilingual FR→EN terminology pairs from aligned sentence pairs
by passing them through a local Ollama model. Produces gold_glossary.json
in the format expected by the S6 logit-boost system.

Usage:
    python data/build_glossary.py --input data/section48/segments_ner_biollm.jsonl
    python data/build_glossary.py --input aligned.jsonl --model mistral --src-field fr --ref-field en
    python data/build_glossary.py --input aligned.jsonl --dry-run --batch-size 3

Output:
    gold_glossary.json: [{"fr": "pneumopathie inflammatoire", "en": "pneumonitis"}, ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(prompt: str, model: str, timeout: int = 120) -> Optional[str]:
    """Send a prompt to Ollama and return the response text, or None on error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print(
            "[ERROR] Could not connect to Ollama. Is it running? (ollama serve)",
            file=sys.stderr,
        )
        return None
    except requests.exceptions.Timeout:
        print("[WARN] Ollama request timed out — skipping batch", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[WARN] Ollama error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a pharmaceutical terminology extractor.
You will be given one or more pairs of aligned sentences: a French source and its English translation.
Your task is to extract medical/pharmaceutical term pairs where the French and English are translations of each other.

Rules:
- Focus on adverse events, symptoms, clinical findings, body systems, drug names, and medical procedures.
- Do NOT extract general words (e.g. "patients", "study", "reported", "the").
- Do NOT extract numbers, percentages, or dates.
- Keep French terms exactly as they appear in the source (including accents).
- Keep English terms exactly as they appear in the translation.
- If a French term has multiple valid English renderings in the text, include both.
- Return ONLY a JSON array. No explanation, no markdown, no preamble.

Format: [{"fr": "terme français", "en": "english term"}, ...]

If you find no medical terms, return an empty array: []"""


def build_prompt(pairs: list[tuple[str, str]]) -> str:
    """Build the extraction prompt for a batch of (fr, en) sentence pairs."""
    lines = [SYSTEM_INSTRUCTIONS, "\n---\nSentence pairs:\n"]
    for i, (fr, en) in enumerate(pairs, 1):
        lines.append(f"Pair {i}:")
        lines.append(f"  FR: {fr}")
        lines.append(f"  EN: {en}")
        lines.append("")
    lines.append("JSON array of term pairs:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

# Common field name variants seen in NER-annotated segment files.
SRC_CANDIDATES = ["src", "source", "fr", "french", "source_text", "fr_text"]
REF_CANDIDATES = ["ref", "reference", "en_ref", "en", "english", "target", "target_text", "en_text"]


def detect_field(record: dict, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in record:
            return c
    return None


def load_aligned_pairs(
    path: Path,
    src_field: Optional[str],
    ref_field: Optional[str],
) -> list[tuple[str, str]]:
    """Load aligned (fr, en) pairs from a JSONL file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: JSON parse error — {e}", file=sys.stderr)
                continue

            # Auto-detect fields on the first record if not specified.
            if src_field is None:
                src_field = detect_field(rec, SRC_CANDIDATES)
                if src_field is None:
                    sys.exit(
                        f"[ERROR] Could not auto-detect source field. "
                        f"Keys found: {list(rec.keys())}. "
                        f"Use --src-field to specify."
                    )
                print(f"[INFO] Auto-detected source field: '{src_field}'")

            if ref_field is None:
                ref_field = detect_field(rec, REF_CANDIDATES)
                if ref_field is None:
                    sys.exit(
                        f"[ERROR] Could not auto-detect reference field. "
                        f"Keys found: {list(rec.keys())}. "
                        f"Use --ref-field to specify."
                    )
                print(f"[INFO] Auto-detected reference field: '{ref_field}'")

            fr = rec.get(src_field, "").strip()
            en = rec.get(ref_field, "").strip()

            if fr and en:
                pairs.append((fr, en))
            else:
                print(f"[WARN] Line {line_no}: empty src or ref — skipping", file=sys.stderr)

    print(f"[INFO] Loaded {len(pairs)} aligned sentence pairs from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Extraction and deduplication
# ---------------------------------------------------------------------------

def extract_terms_from_batch(
    pairs: list[tuple[str, str]],
    model: str,
    dry_run: bool,
) -> list[dict]:
    """Call Ollama on a batch of pairs and return extracted term pairs."""
    prompt = build_prompt(pairs)

    if dry_run:
        print(f"\n[DRY RUN] Prompt for {len(pairs)} pair(s):\n{prompt}\n")
        return []

    response = call_ollama(prompt, model)
    if response is None:
        return []

    # Strip markdown fences if the model adds them.
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    try:
        terms = json.loads(response)
        if not isinstance(terms, list):
            print("[WARN] Model returned non-list JSON — skipping", file=sys.stderr)
            return []
        valid = []
        for t in terms:
            if not isinstance(t, dict) or "fr" not in t or "en" not in t:
                continue
            fr = str(t["fr"]).strip()
            en_val = t["en"]
            # Model sometimes returns a list of English renderings — expand them.
            if isinstance(en_val, list):
                for en in en_val:
                    en = str(en).strip()
                    if fr and en:
                        valid.append({"fr": fr, "en": en})
            else:
                en = str(en_val).strip()
                if fr and en:
                    valid.append({"fr": fr, "en": en})
        return valid
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse model response as JSON:\n{response[:200]}", file=sys.stderr)
        return []


def deduplicate(entries: list[dict]) -> list[dict]:
    """Keep all distinct (fr, en) pairs; preserve multiple English renderings per French term."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for e in entries:
        key = (e["fr"].lower(), e["en"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return sorted(unique, key=lambda x: x["fr"].lower())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FR→EN pharmaceutical glossary from aligned sentence pairs using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="JSONL file with aligned FR-EN sentence pairs")
    p.add_argument("--output", "-o", type=Path, default=Path("data/section48/gold_glossary.json"),
                   help="Output JSON file (default: data/section48/gold_glossary.json)")
    p.add_argument("--model", "-m", default="mistral",
                   help="Ollama model name (default: mistral). Try: llama3, gemma3, qwen2.5")
    p.add_argument("--src-field", default=None,
                   help="JSONL field for French source text (auto-detected if omitted)")
    p.add_argument("--ref-field", default=None,
                   help="JSONL field for English reference text (auto-detected if omitted)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Sentence pairs per Ollama call (default: 4)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between batches (default: 0.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the first batch prompt and exit without calling Ollama")
    p.add_argument("--verbose", action="store_true",
                   help="Print extracted terms as they are found")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"[ERROR] Input file not found: {args.input}")

    pairs = load_aligned_pairs(args.input, args.src_field, args.ref_field)
    if not pairs:
        sys.exit("[ERROR] No aligned pairs loaded — check your input file and field names")

    batches = [pairs[i: i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
    print(f"[INFO] {len(batches)} batches of up to {args.batch_size} pairs each")
    print(f"[INFO] Model: {args.model}")

    all_terms: list[dict] = []
    for batch_idx, batch in enumerate(batches, 1):
        print(f"[INFO] Batch {batch_idx}/{len(batches)} ({len(batch)} pairs)...", end=" ", flush=True)
        terms = extract_terms_from_batch(batch, args.model, args.dry_run)
        if args.dry_run:
            break
        print(f"→ {len(terms)} terms extracted")
        if args.verbose:
            for t in terms:
                print(f"    {t['fr']}  →  {t['en']}")
        all_terms.extend(terms)
        if args.sleep > 0:
            time.sleep(args.sleep)

    if args.dry_run:
        print("[INFO] Dry run complete — no output written")
        return

    glossary = deduplicate(all_terms)
    print(f"\n[INFO] {len(all_terms)} raw extractions → {len(glossary)} unique pairs after deduplication")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Glossary written to {args.output}")

    print("\n--- Sample (first 10 entries) ---")
    for entry in glossary[:10]:
        print(f"  {entry['fr']!r:40s} → {entry['en']!r}")


if __name__ == "__main__":
    main()
