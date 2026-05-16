"""
TermPlanMT Unified Expert Evaluation
Same 30 segments as audit_pipeline.py (seed=42), all 8 systems,
both NER conditions. EMA medical translator persona.
Results -> error_analysis/unified_eval.csv + unified_eval_report.txt
"""

import json, csv, random, time, os, sys, re
from pathlib import Path
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from openai import OpenAI

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set.")
    sys.exit(1)

client = OpenAI()
Path("error_analysis").mkdir(exist_ok=True)

OUT_CSV  = "error_analysis/unified_eval.csv"
REPORT   = "error_analysis/unified_eval_report.txt"

# ── Load the exact 30 IDs from the audit (seed=42) ───────────────────────────
audit_rows = list(csv.DictReader(open("error_analysis/audit_annotated.csv")))
SAMPLE_IDS = [r["id"] for r in audit_rows]
print(f"Loaded {len(SAMPLE_IDS)} segment IDs from audit_annotated.csv")

# ── Load all system outputs ───────────────────────────────────────────────────
def load_map(path):
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found")
        return {}
    return {r["id"]: (r.get("hyp") or "").strip()
            for r in [json.loads(l) for l in p.read_text().splitlines() if l.strip()]}

segs_index = {r["id"]: r for r in
              [json.loads(l) for l in
               open("data/section48/segments_ner_biollm.jsonl").read().splitlines() if l.strip()]}

s1    = load_map("results/ner_biollm/s1.jsonl")
s2    = load_map("results/ner_biollm/s2.jsonl")
s3b   = load_map("results/ner_biollm/s3.jsonl")
s3ft  = load_map("results/ner_biollm_finetuned/s3_clean.jsonl")
s5b   = load_map("results/ner_biollm/s5_mistral.jsonl")
s5ft  = load_map("results/ner_biollm_finetuned/s5_mistral_clean.jsonl")
s6    = load_map("results/ner_biollm/s6.jsonl")

sample = [segs_index[sid] for sid in SAMPLE_IDS if sid in segs_index]
print(f"Matched {len(sample)} segments in JSONL")

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior medical translator with 20 years of experience
working on EMA regulatory submissions, specifically Summaries of Product
Characteristics (SmPCs). You have deep expertise in MedDRA terminology, EMA style
guidelines, and the register required for adverse event sections (Section 4.8).

You know that EMA SmPCs require consistent, precise adverse event terminology;
MedDRA Preferred Terms are the pharmacovigilance coding standard; however,
professional SmPC prose often uses slightly more descriptive clinical language
rather than bare MedDRA labels; the gold standard is the certified human reference.
Your evaluations will be used in an academic paper. Be specific and direct."""

SEG_PROMPT = """Evaluate these machine translations of a French SmPC adverse event
segment. All systems translate the same French source.

FRENCH SOURCE:
{fr}

CERTIFIED HUMAN REFERENCE (EMA submission):
{ref}

[S1]     NLLB-200, no ontology, baseline NER:
{s1}

[S2]     Mistral-7B full document context, no ontology:
{s2}

[S3-base] Mistral + MedDRA GraphRAG, baseline NER (39% CCR):
{s3b}

[S3-FT]  Mistral + MedDRA GraphRAG, fine-tuned NER (36% CCR, cleaner spans):
{s3ft}

[S5-base] Mistral + logit-boosted MedDRA terms, baseline NER:
{s5b}

[S5-FT]  Mistral + logit-boosted MedDRA terms, fine-tuned NER:
{s5ft}

[S6]     NLLB + oracle glossary (terms from reference):
{s6}

Answer with exactly these field names, one per line, colon-separated:

BEST_SYSTEM [S1|S2|S3-base|S3-FT|S5-base|S5-FT|S6|HUMAN|TIE]:

RANKING [comma-separated best to worst, all 8 options including HUMAN]:

NER_WINNER [baseline|finetuned|tied]:
Which NER condition (baseline vs fine-tuned) produces better graph system output?

NER_REASON [1 sentence]:

S1_VERDICT [1 sentence]:
S2_VERDICT [1 sentence]:
S3_BASE_VERDICT [1 sentence]:
S3_FT_VERDICT [1 sentence]:
S5_BASE_VERDICT [1 sentence]:
S5_FT_VERDICT [1 sentence]:
S6_VERDICT [1 sentence]:

KEY_OBSERVATION [2 sentences max]: Most important terminology quality finding."""

SYNTHESIS_PROMPT = """You evaluated {n} French-to-English EMA SmPC segments across
8 systems (S1, S2, S3-base, S3-FT, S5-base, S5-FT, S6, Human reference).

BEST SYSTEM VOTES ({n} segments):
{best_votes}

AVERAGE RANK (1=best, 8=worst):
{avg_ranks}

NER CONDITION — FINE-TUNED vs BASELINE for graph systems:
{ner_tally}

KEY OBSERVATIONS (sample):
{observations}

Write a structured expert assessment with these exact sections:

OVERALL_VERDICT:
Pick one recommended system for EMA regulatory MT. Defend it in 3-4 sentences.

SYSTEM_BY_SYSTEM:
For each system (S1, S2, S3-base, S3-FT, S5-base, S5-FT, S6) give 1-2 sentences:
strengths, weaknesses, verdict.

NER_MODEL_VERDICT:
Baseline BioMistral (zero-shot) vs fine-tuned BioMistral for the graph-augmented
pipeline. Which is better and why? 2-3 sentences.

REGISTER_MISMATCH_VERDICT:
Does MedDRA ontology grounding help or hurt from a practicing EMA translator's
perspective? 2-3 sentences. Be honest.

PRACTICAL_RECOMMENDATION:
What would you tell an EMA translation team about deploying these systems?
3-4 sentences."""

# ── Parse ─────────────────────────────────────────────────────────────────────
def parse(text):
    text = re.sub(r'```[a-z]*\n?', '', text).strip('`').strip()
    fields, cur_key, cur_val = {}, None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+\.\s*', '', line)
        if ':' in line:
            k, _, v = line.partition(':')
            k = k.strip().upper().replace(' ', '_')
            if re.match(r'^[A-Z][A-Z0-9_\-]*$', k):
                if cur_key:
                    fields[cur_key] = ' '.join(cur_val).strip()
                cur_key, cur_val = k, [v.strip()] if v.strip() else []
                continue
        if cur_key:
            cur_val.append(line)
    if cur_key:
        fields[cur_key] = ' '.join(cur_val).strip()
    return fields

# ── Run per-segment ───────────────────────────────────────────────────────────
rows = []
best_votes  = defaultdict(int)
rank_scores = defaultdict(list)
ner_tally   = defaultdict(int)
observations = []

for i, seg in enumerate(sample):
    sid = seg["id"]
    print(f"  [{i+1}/{len(sample)}] {sid}...", end=" ", flush=True)

    prompt = SEG_PROMPT.format(
        fr   = seg["fr"][:600],
        ref  = seg["en_ref"][:500],
        s1   = (s1.get(sid)   or "N/A")[:380],
        s2   = (s2.get(sid)   or "N/A")[:380],
        s3b  = (s3b.get(sid)  or "N/A")[:380],
        s3ft = (s3ft.get(sid) or "N/A")[:380],
        s5b  = (s5b.get(sid)  or "N/A")[:380],
        s5ft = (s5ft.get(sid) or "N/A")[:380],
        s6   = (s6.get(sid)   or "N/A")[:380],
    )

    try:
        msg = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=900,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
        )
        raw    = msg.choices[0].message.content.strip()
        parsed = parse(raw)
        print("ok")
    except Exception as e:
        print(f"ERROR: {e}")
        parsed = {}
        raw = f"ERROR: {e}"

    time.sleep(0.5)

    best = parsed.get("BEST_SYSTEM", "UNKNOWN").strip().upper()
    best_votes[best] += 1

    for pos, sys in enumerate(
            [r.strip().upper() for r in parsed.get("RANKING","").split(",") if r.strip()]):
        rank_scores[sys].append(pos + 1)

    ner_tally[parsed.get("NER_WINNER","").strip().lower()] += 1
    obs = parsed.get("KEY_OBSERVATION","")
    if obs:
        observations.append(f"[{sid}] {obs[:200]}")

    rows.append({
        "id":             sid,
        "fr":             seg["fr"][:200],
        "human_ref":      seg["en_ref"][:200],
        "best_system":    best,
        "ranking":        parsed.get("RANKING",""),
        "ner_winner":     parsed.get("NER_WINNER",""),
        "ner_reason":     parsed.get("NER_REASON",""),
        "s1_verdict":     parsed.get("S1_VERDICT",""),
        "s2_verdict":     parsed.get("S2_VERDICT",""),
        "s3base_verdict": parsed.get("S3_BASE_VERDICT",""),
        "s3ft_verdict":   parsed.get("S3_FT_VERDICT",""),
        "s5base_verdict": parsed.get("S5_BASE_VERDICT",""),
        "s5ft_verdict":   parsed.get("S5_FT_VERDICT",""),
        "s6_verdict":     parsed.get("S6_VERDICT",""),
        "key_observation":obs,
        "raw_response":   raw,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV: {OUT_CSV}")

# ── Aggregates ────────────────────────────────────────────────────────────────
n = len(sample)
def pct(c): return round(100 * c / n)

best_votes_str = "\n".join(
    f"  {k:<14} {v}/{n} = {pct(v)}%"
    for k, v in sorted(best_votes.items(), key=lambda x: -x[1]))

avg_ranks_str = "\n".join(
    f"  {k:<14} avg rank {sum(v)/len(v):.2f}  (n={len(v)})"
    for k, v in sorted(rank_scores.items(),
                       key=lambda x: sum(x[1])/len(x[1]) if x[1] else 99)
    if v)

ner_tally_str = "\n".join(
    f"  {k:<12} {v}/{n} = {pct(v)}%"
    for k, v in sorted(ner_tally.items(), key=lambda x: -x[1]))

# ── Synthesis ─────────────────────────────────────────────────────────────────
print("\nRunning synthesis...", flush=True)
try:
    synth = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1400,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SYNTHESIS_PROMPT.format(
                n=n,
                best_votes=best_votes_str,
                avg_ranks=avg_ranks_str,
                ner_tally=ner_tally_str,
                observations="\n".join(observations[:15]),
            )},
        ]
    ).choices[0].message.content.strip()
except Exception as e:
    synth = f"Synthesis ERROR: {e}"

# ── Report ────────────────────────────────────────────────────────────────────
report = f"""TERMPLANMT UNIFIED EXPERT EVALUATION
{n} segments (seed=42, same as audit_pipeline.py) · 8 systems · GPT-4o
{'='*70}

BEST SYSTEM VOTES
{best_votes_str}

AVERAGE RANK (1=best, 8=worst)
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

Path(REPORT).write_text(report, encoding="utf-8")
print(report)
print(f"\nReport : {REPORT}")
print(f"CSV    : {OUT_CSV}")
