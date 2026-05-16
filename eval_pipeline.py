"""
TermPlanMT General-Purpose Evaluation Pipeline
Samples segments, evaluates them via GPT-4o acting as an EMA medical translator,
and writes per-segment CSV + a synthesis report.

Usage:
  python3 eval_pipeline.py [options]

Run --help for full option list.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── System paths (same as unified_eval.py) ───────────────────────────────────
SEGS_JSONL = "data/section48/segments_ner_biollm.jsonl"

SYSTEM_PATHS = {
    "s1":   "results/ner_biollm/s1.jsonl",
    "s2":   "results/ner_biollm/s2.jsonl",
    "s3":   "results/ner_biollm/s3.jsonl",
    "s3ft": "results/ner_biollm_finetuned/s3_clean.jsonl",
    "s5b":  "results/ner_biollm/s5_mistral.jsonl",
    "s5ft": "results/ner_biollm_finetuned/s5_mistral_clean.jsonl",
    "s6":   "results/ner_biollm/s6.jsonl",
}

# Labels shown in the prompt and report
SYSTEM_LABELS = {
    "s1":   "S1",
    "s2":   "S2",
    "s3":   "S3-base",
    "s3ft": "S3-FT",
    "s5b":  "S5-base",
    "s5ft": "S5-FT",
    "s6":   "S6",
}

ALL_SYSTEM_CHOICES = list(SYSTEM_PATHS.keys())

# ── EMA medical translator system prompt (same as unified_eval.py) ────────────
SYSTEM_PROMPT = """You are a senior medical translator with 20 years of experience
working on EMA regulatory submissions, specifically Summaries of Product
Characteristics (SmPCs). You have deep expertise in MedDRA terminology, EMA style
guidelines, and the register required for adverse event sections (Section 4.8).

You know that EMA SmPCs require consistent, precise adverse event terminology;
MedDRA Preferred Terms are the pharmacovigilance coding standard; however,
professional SmPC prose often uses slightly more descriptive clinical language
rather than bare MedDRA labels; the gold standard is the certified human reference.
Your evaluations will be used in an academic paper. Be specific and direct."""


def _build_seg_prompt(seg: dict, systems: list[str], maps: dict[str, dict]) -> str:
    """Build per-segment evaluation prompt showing only the selected systems."""
    sid = seg["id"]
    lines = [
        "Evaluate these machine translations of a French SmPC adverse event segment."
        " All systems translate the same French source.",
        "",
        "FRENCH SOURCE:",
        seg["fr"][:600],
        "",
        "CERTIFIED HUMAN REFERENCE (EMA submission):",
        seg["en_ref"][:500],
        "",
    ]
    for key in systems:
        label = SYSTEM_LABELS[key]
        hyp = (maps[key].get(sid) or "N/A")[:380]
        lines.append(f"[{label}]")
        lines.append(hyp)
        lines.append("")

    # Choices string for BEST_SYSTEM field
    choices = "|".join(SYSTEM_LABELS[k] for k in systems) + "|HUMAN|TIE"
    all_labels = "|".join(SYSTEM_LABELS[k] for k in systems) + "|HUMAN"

    lines += [
        "Answer with exactly these field names, one per line, colon-separated:",
        "",
        f"BEST_SYSTEM [{choices}]:",
        "",
        f"RANKING [comma-separated best to worst, all options including HUMAN]:",
        "",
    ]

    if any(k in systems for k in ("s3", "s3ft", "s5b", "s5ft")):
        lines += [
            "NER_WINNER [baseline|finetuned|tied]:",
            "Which NER condition (baseline vs fine-tuned) produces better graph system output?",
            "",
            "NER_REASON [1 sentence]:",
            "",
        ]

    for key in systems:
        label = SYSTEM_LABELS[key].replace("-", "_").upper()
        lines.append(f"{label}_VERDICT [1 sentence]:")

    lines += [
        "",
        "KEY_OBSERVATION [2 sentences max]: Most important terminology quality finding.",
    ]
    return "\n".join(lines)


def _parse(text: str) -> dict:
    """Parse field:value response lines into a dict."""
    text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
    fields: dict = {}
    cur_key: str | None = None
    cur_val: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().upper().replace(" ", "_")
            if re.match(r"^[A-Z][A-Z0-9_\-]*$", k):
                if cur_key:
                    fields[cur_key] = " ".join(cur_val).strip()
                cur_key, cur_val = k, [v.strip()] if v.strip() else []
                continue
        if cur_key:
            cur_val.append(line)
    if cur_key:
        fields[cur_key] = " ".join(cur_val).strip()
    return fields


def _build_synthesis_prompt(
    n: int,
    systems: list[str],
    best_votes_str: str,
    avg_ranks_str: str,
    ner_tally_str: str,
    observations: list[str],
) -> str:
    sys_labels = ", ".join(SYSTEM_LABELS[k] for k in systems)
    return f"""You evaluated {n} French-to-English EMA SmPC segments across
{len(systems) + 1} systems ({sys_labels}, Human reference).

BEST SYSTEM VOTES ({n} segments):
{best_votes_str}

AVERAGE RANK (1=best, {len(systems) + 1}=worst):
{avg_ranks_str}

NER CONDITION — FINE-TUNED vs BASELINE for graph systems:
{ner_tally_str}

KEY OBSERVATIONS (sample):
{chr(10).join(observations[:15])}

Write a structured expert assessment with these exact sections:

OVERALL_VERDICT:
Pick one recommended system for EMA regulatory MT. Defend it in 3-4 sentences.

SYSTEM_BY_SYSTEM:
For each system give 1-2 sentences: strengths, weaknesses, verdict.

NER_MODEL_VERDICT:
Baseline BioMistral (zero-shot) vs fine-tuned BioMistral for the graph-augmented
pipeline. Which is better and why? 2-3 sentences.

REGISTER_MISMATCH_VERDICT:
Does MedDRA ontology grounding help or hurt from a practicing EMA translator's
perspective? 2-3 sentences. Be honest.

PRACTICAL_RECOMMENDATION:
What would you tell an EMA translation team about deploying these systems?
3-4 sentences."""


def load_map(path: str) -> dict:
    """Load a JSONL file as {id: hyp} map."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return {}
    return {
        r["id"]: (r.get("hyp") or "").strip()
        for r in [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    }


def run(
    segments: int,
    seed: int,
    systems: list[str],
    use_ids_path: str | None,
    out_csv: str,
    out_report: str,
    model: str,
) -> None:
    import random

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    # Validate systems
    for s in systems:
        if s not in ALL_SYSTEM_CHOICES:
            print(f"ERROR: unknown system '{s}'. Choices: {ALL_SYSTEM_CHOICES}", file=sys.stderr)
            sys.exit(1)

    # Load segments index
    segs_all = [
        json.loads(l)
        for l in Path(SEGS_JSONL).read_text().splitlines()
        if l.strip()
    ]
    segs_index = {r["id"]: r for r in segs_all}

    # Load system maps
    maps: dict[str, dict] = {s: load_map(SYSTEM_PATHS[s]) for s in systems}

    # Determine sample
    if use_ids_path:
        id_rows = list(csv.DictReader(open(use_ids_path, encoding="utf-8")))
        sample_ids = [r["id"] for r in id_rows]
        sample = [segs_index[sid] for sid in sample_ids if sid in segs_index]
        print(f"Loaded {len(sample)} segment IDs from {use_ids_path}")
    else:
        # Filter to segments where all selected systems have output
        valid = [
            s for s in segs_all
            if all((maps[k].get(s["id"]) or "").strip() for k in systems)
        ]
        random.seed(seed)
        sample = random.sample(valid, min(segments, len(valid)))
        print(f"Sampled {len(sample)} segments (seed={seed})")

    # Ensure output dirs exist
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)

    # ── Per-segment evaluation ────────────────────────────────────────────────
    rows = []
    best_votes: dict = defaultdict(int)
    rank_scores: dict = defaultdict(list)
    ner_tally: dict = defaultdict(int)
    observations: list = []

    for i, seg in enumerate(sample):
        sid = seg["id"]
        print(f"  [{i+1}/{len(sample)}] {sid}...", end=" ", flush=True)

        prompt = _build_seg_prompt(seg, systems, maps)

        try:
            msg = client.chat.completions.create(
                model=model,
                max_tokens=900,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            parsed = _parse(raw)
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")
            parsed = {}
            raw = f"ERROR: {e}"

        time.sleep(0.5)

        best = parsed.get("BEST_SYSTEM", "UNKNOWN").strip().upper()
        best_votes[best] += 1

        for pos, sys_label in enumerate(
            [r.strip().upper() for r in parsed.get("RANKING", "").split(",") if r.strip()]
        ):
            rank_scores[sys_label].append(pos + 1)

        ner_tally[parsed.get("NER_WINNER", "").strip().lower()] += 1
        obs = parsed.get("KEY_OBSERVATION", "")
        if obs:
            observations.append(f"[{sid}] {obs[:200]}")

        row: dict = {
            "id":          sid,
            "fr":          seg["fr"][:200],
            "human_ref":   seg["en_ref"][:200],
            "best_system": best,
            "ranking":     parsed.get("RANKING", ""),
            "ner_winner":  parsed.get("NER_WINNER", ""),
            "ner_reason":  parsed.get("NER_REASON", ""),
        }
        for key in systems:
            label_field = SYSTEM_LABELS[key].replace("-", "_").upper()
            row[f"{label_field.lower()}_verdict"] = parsed.get(f"{label_field}_VERDICT", "")
        row["key_observation"] = obs
        row["raw_response"] = raw
        rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {out_csv}")

    # ── Aggregates ────────────────────────────────────────────────────────────
    n = len(sample)

    def pct(c: int) -> int:
        return round(100 * c / n) if n else 0

    best_votes_str = "\n".join(
        f"  {k:<14} {v}/{n} = {pct(v)}%"
        for k, v in sorted(best_votes.items(), key=lambda x: -x[1])
    )
    avg_ranks_str = "\n".join(
        f"  {k:<14} avg rank {sum(v)/len(v):.2f}  (n={len(v)})"
        for k, v in sorted(
            rank_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 99,
        )
        if v
    )
    ner_tally_str = "\n".join(
        f"  {k:<12} {v}/{n} = {pct(v)}%"
        for k, v in sorted(ner_tally.items(), key=lambda x: -x[1])
    )

    # ── Synthesis ─────────────────────────────────────────────────────────────
    print("\nRunning synthesis...", flush=True)
    synth_prompt = _build_synthesis_prompt(
        n, systems, best_votes_str, avg_ranks_str, ner_tally_str, observations
    )
    try:
        synth = client.chat.completions.create(
            model=model,
            max_tokens=1400,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": synth_prompt},
            ],
        ).choices[0].message.content.strip()
    except Exception as e:
        synth = f"Synthesis ERROR: {e}"

    # ── Report ────────────────────────────────────────────────────────────────
    sys_label_str = ", ".join(SYSTEM_LABELS[k] for k in systems)
    report = f"""TERMPLANMT EXPERT EVALUATION
{n} segments · model={model} · seed={seed} · systems={sys_label_str}
{'='*70}

BEST SYSTEM VOTES
{best_votes_str}

AVERAGE RANK (1=best)
{avg_ranks_str}

NER CONDITION: FINE-TUNED vs BASELINE
{ner_tally_str}

{'='*70}
EXPERT SYNTHESIS
{'='*70}

{synth}

{'='*70}
KEY OBSERVATIONS (per segment)
{'='*70}

{chr(10).join(observations)}
"""
    Path(out_report).write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport : {out_report}")
    print(f"CSV    : {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TermPlanMT general-purpose evaluation pipeline"
    )
    parser.add_argument("--segments", type=int, default=30,
                        help="Number of segments to sample (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--systems",
        type=str,
        default="s1,s2,s5b,s6",
        help=(
            "Comma-separated list of systems to evaluate. "
            f"Choices: {', '.join(ALL_SYSTEM_CHOICES)}. Default: s1,s2,s5b,s6"
        ),
    )
    parser.add_argument(
        "--use-ids",
        metavar="PATH",
        default=None,
        help="CSV file with an 'id' column; use those IDs instead of random sampling",
    )
    parser.add_argument(
        "--out-csv",
        default="error_analysis/eval_output.csv",
        help="Output CSV path (default: error_analysis/eval_output.csv)",
    )
    parser.add_argument(
        "--out-report",
        default="error_analysis/eval_report.txt",
        help="Output report path (default: error_analysis/eval_report.txt)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    args = parser.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in ALL_SYSTEM_CHOICES:
            parser.error(f"Unknown system '{s}'. Choices: {ALL_SYSTEM_CHOICES}")

    run(
        segments=args.segments,
        seed=args.seed,
        systems=systems,
        use_ids_path=args.use_ids,
        out_csv=args.out_csv,
        out_report=args.out_report,
        model=args.model,
    )


if __name__ == "__main__":
    main()
