"""
TermPlanMT Expert Medical Translation Evaluation
GPT-4o acts as a senior EMA medical translator and evaluates all systems.
Produces per-segment rankings + system-level verdict + NER model comparison.
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
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or pass inline.")
    sys.exit(1)

client = OpenAI()
Path("error_analysis").mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLE  = 25
SEED      = 99
OUT_CSV   = "error_analysis/expert_eval.csv"
REPORT    = "error_analysis/expert_report.txt"

# ── Load all system outputs ───────────────────────────────────────────────────
def load_map(path, key="hyp"):
    p = Path(path)
    if not p.exists():
        return {}
    return {r["id"]: r.get(key, "") for r in
            [json.loads(l) for l in p.read_text().splitlines() if l.strip()]}

segs = [json.loads(l) for l in
        open("data/section48/segments_ner_biollm.jsonl").read().splitlines() if l.strip()]

# Baseline NER condition
s1   = load_map("results/ner_biollm/s1.jsonl")
s2   = load_map("results/ner_biollm/s2.jsonl")
s3b  = load_map("results/ner_biollm/s3.jsonl")
s4b  = load_map("results/ner_biollm/s4.jsonl")
s5b  = load_map("results/ner_biollm/s5_mistral.jsonl")
s6   = load_map("results/ner_biollm/s6.jsonl")

# Fine-tuned NER condition (graph systems only — S1/S2 unchanged)
s3ft = load_map("results/ner_biollm_finetuned/s3_clean.jsonl")
s5ft = load_map("results/ner_biollm_finetuned/s5_mistral_clean.jsonl")

# Only keep segments where core systems have output
valid = [s for s in segs
         if s["id"] in s2 and s["id"] in s5b
         and (s2[s["id"]] or "").strip()
         and (s5b[s["id"]] or "").strip()]

random.seed(SEED)
sample = random.sample(valid, min(N_SAMPLE, len(valid)))
print(f"Sampled {len(sample)} segments for expert evaluation")

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior medical translator with 20 years of experience
working on EMA regulatory submissions, specifically Summaries of Product
Characteristics (SmPCs). You have deep expertise in MedDRA terminology, EMA style
guidelines, and the specific register required for adverse event sections (Section 4.8).

You know that:
- EMA SmPCs require consistent, precise adverse event terminology
- MedDRA Preferred Terms (PTs) are the standard coding vocabulary for pharmacovigilance
- However, professional SmPC prose often uses slightly more descriptive clinical language
  rather than bare MedDRA PT labels
- The gold standard is the certified human reference translation submitted to EMA
- You will evaluate machine translation systems honestly and critically

Your evaluations will be used in an academic paper comparing MT approaches for
regulatory medical document translation. Be specific, technical, and direct."""

SEG_PROMPT = """Evaluate these machine translations of a French SmPC adverse event
segment. Rate each system and explain your reasoning as an expert EMA medical translator.

FRENCH SOURCE:
{fr}

CERTIFIED HUMAN REFERENCE (EMA submission):
{ref}

SYSTEM OUTPUTS:
[S1] NLLB-200 baseline (no ontology, NMT):
{s1}

[S2] Mistral-7B full document context (no ontology, LLM):
{s2}

[S3-base] Mistral + MedDRA GraphRAG, baseline NER:
{s3b}

[S3-ft] Mistral + MedDRA GraphRAG, fine-tuned NER:
{s3ft}

[S5-base] Mistral + logit-boosted MedDRA terms, baseline NER:
{s5b}

[S5-ft] Mistral + logit-boosted MedDRA terms, fine-tuned NER:
{s5ft}

[S6] NLLB + oracle glossary (reference terms injected):
{s6}

EVALUATION TASKS — answer each with the exact field name:

BEST_SYSTEM [S1|S2|S3-base|S3-ft|S5-base|S5-ft|S6|HUMAN|TIE]:
The single output closest to what a certified EMA translator would produce.

RANKING [comma-separated from best to worst, e.g. S2,HUMAN,S1,...]:
Full ranking of all outputs including HUMAN reference.

S1_VERDICT [1-2 sentences]:
S2_VERDICT [1-2 sentences]:
S3_BASE_VERDICT [1-2 sentences]:
S3_FT_VERDICT [1-2 sentences]:
S5_BASE_VERDICT [1-2 sentences]:
S5_FT_VERDICT [1-2 sentences]:
S6_VERDICT [1-2 sentences]:

NER_WINNER [baseline|finetuned|tied]:
Which NER condition produces better translations for the graph-augmented systems?

NER_REASON [1 sentence]:
Why?

KEY_OBSERVATION [2-3 sentences]:
The single most important observation about terminology quality in this segment.
Be specific — reference actual words used.

Respond with exact field names, one per line, colon-separated."""

SYNTHESIS_PROMPT = """You have now evaluated {n} segments from a French-to-English
EMA SmPC translation study comparing 6 MT systems.

Here is the aggregated data from all segment evaluations:

BEST SYSTEM VOTES:
{best_votes}

AVERAGE RANKINGS (1=best):
{avg_ranks}

NER CONDITION WINNER TALLY:
{ner_tally}

SAMPLE OF KEY OBSERVATIONS FROM INDIVIDUAL SEGMENTS:
{observations}

Now write a comprehensive expert assessment as a senior EMA medical translator.
Structure your response with these exact sections:

OVERALL_VERDICT:
Which system would you actually recommend for regulatory MT work and why?
Be direct — pick one winner and defend it. (3-4 sentences)

SYSTEM_BY_SYSTEM:
S1 (NLLB baseline): strengths, weaknesses, when to use (2-3 sentences)
S2 (Mistral full-doc): strengths, weaknesses, when to use (2-3 sentences)
S3 (GraphRAG): strengths, weaknesses, when to use (2-3 sentences)
S5 (logit boosting): strengths, weaknesses, when to use (2-3 sentences)
S6 (oracle glossary): strengths, weaknesses, when to use (2-3 sentences)

NER_MODEL_VERDICT:
Which NER model (baseline BioMistral zero-shot vs fine-tuned) produces better
results for the graph-augmented pipeline, and why? (2-3 sentences)

ONTOLOGY_GROUNDING_VERDICT:
Does MedDRA ontology grounding actually help or hurt translation quality from
the perspective of a practicing EMA translator? Be honest. (2-3 sentences)

PRACTICAL_RECOMMENDATION:
If you were advising an EMA translation team today, what would you tell them
about using these systems in production? (3-4 sentences)"""

# ── Parse response ────────────────────────────────────────────────────────────
def parse(text):
    text = re.sub(r'```[a-z]*\n?', '', text).strip('`').strip()
    fields = {}
    current_key = None
    current_val = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbers
        line = re.sub(r'^\d+\.\s*', '', line)
        if ':' in line:
            # Check if this looks like a field name (all caps / underscores before colon)
            candidate_key, _, val = line.partition(':')
            candidate_key = candidate_key.strip().upper().replace(' ', '_')
            if re.match(r'^[A-Z][A-Z0-9_]*$', candidate_key):
                if current_key:
                    fields[current_key] = ' '.join(current_val).strip()
                current_key = candidate_key
                current_val = [val.strip()] if val.strip() else []
                continue
        if current_key:
            current_val.append(line)
    if current_key:
        fields[current_key] = ' '.join(current_val).strip()
    return fields

# ── Run per-segment evaluation ────────────────────────────────────────────────
rows = []
best_votes   = defaultdict(int)
rank_scores  = defaultdict(list)   # lower = better rank
ner_tally    = defaultdict(int)
observations = []

SYSTEMS_ORDER = ["HUMAN", "S2", "S3-FT", "S3-BASE", "S5-FT",
                 "S5-BASE", "S6", "S1"]

for i, seg in enumerate(sample):
    sid = seg["id"]
    print(f"  [{i+1}/{N_SAMPLE}] {sid}...", end=" ", flush=True)

    prompt = SEG_PROMPT.format(
        fr   =seg["fr"][:600],
        ref  =seg["en_ref"][:500],
        s1   =(s1.get(sid)  or "N/A")[:400],
        s2   =(s2.get(sid)  or "N/A")[:400],
        s3b  =(s3b.get(sid) or "N/A")[:400],
        s3ft =(s3ft.get(sid)or "N/A")[:400],
        s5b  =(s5b.get(sid) or "N/A")[:400],
        s5ft =(s5ft.get(sid)or "N/A")[:400],
        s6   =(s6.get(sid)  or "N/A")[:400],
    )

    try:
        msg = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=900,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
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

    ranking_raw = parsed.get("RANKING", "")
    rank_list = [r.strip().upper() for r in ranking_raw.split(",") if r.strip()]
    for pos, sys in enumerate(rank_list):
        rank_scores[sys].append(pos + 1)

    ner_winner = parsed.get("NER_WINNER", "").strip().lower()
    ner_tally[ner_winner] += 1

    obs = parsed.get("KEY_OBSERVATION", "")
    if obs:
        observations.append(f"[{sid}] {obs[:200]}")

    rows.append({
        "id":            sid,
        "fr":            seg["fr"][:200],
        "human_ref":     seg["en_ref"][:200],
        "best_system":   best,
        "ranking":       ranking_raw,
        "ner_winner":    ner_winner,
        "s1_verdict":    parsed.get("S1_VERDICT", ""),
        "s2_verdict":    parsed.get("S2_VERDICT", ""),
        "s3base_verdict":parsed.get("S3_BASE_VERDICT", ""),
        "s3ft_verdict":  parsed.get("S3_FT_VERDICT", ""),
        "s5base_verdict":parsed.get("S5_BASE_VERDICT", ""),
        "s5ft_verdict":  parsed.get("S5_FT_VERDICT", ""),
        "s6_verdict":    parsed.get("S6_VERDICT", ""),
        "key_observation":obs,
        "raw_response":  raw,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV: {OUT_CSV}")

# ── Build aggregates for synthesis ────────────────────────────────────────────
n = len(sample)
def pct(c): return round(100 * c / n)

best_votes_str = "\n".join(
    f"  {k:<14} {v}/{n} = {pct(v)}%"
    for k, v in sorted(best_votes.items(), key=lambda x: -x[1])
)

avg_ranks_str = "\n".join(
    f"  {k:<14} avg rank {sum(v)/len(v):.2f}  (n={len(v)})"
    for k, v in sorted(rank_scores.items(),
                       key=lambda x: sum(x[1])/len(x[1]) if x[1] else 99)
    if v
)

ner_tally_str = "\n".join(
    f"  {k:<12} {v}/{n} = {pct(v)}%"
    for k, v in sorted(ner_tally.items(), key=lambda x: -x[1])
)

obs_sample = "\n".join(observations[:12])

# ── Synthesis call ────────────────────────────────────────────────────────────
print("\nRunning synthesis...", flush=True)
synth_prompt = SYNTHESIS_PROMPT.format(
    n=n,
    best_votes=best_votes_str,
    avg_ranks=avg_ranks_str,
    ner_tally=ner_tally_str,
    observations=obs_sample,
)

try:
    synth_msg = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1400,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": synth_prompt}
        ]
    )
    synthesis = synth_msg.choices[0].message.content.strip()
except Exception as e:
    synthesis = f"Synthesis ERROR: {e}"

# ── Write report ──────────────────────────────────────────────────────────────
report = f"""TERMPLANMT EXPERT EVALUATION REPORT
{n} segments · GPT-4o expert (EMA medical translator persona) · seed={SEED}
{'='*70}

BEST SYSTEM VOTES
{best_votes_str}

AVERAGE RANK (1 = best, lower is better)
{avg_ranks_str}

NER CONDITION: BASELINE vs FINE-TUNED
{ner_tally_str}

{'='*70}
EXPERT SYNTHESIS
{'='*70}

{synthesis}

{'='*70}
SAMPLE KEY OBSERVATIONS (individual segments)
{'='*70}

{chr(10).join(observations)}
"""

Path(REPORT).write_text(report, encoding="utf-8")
print(report)
print(f"\nReport: {REPORT}")
print(f"CSV:    {OUT_CSV}")
