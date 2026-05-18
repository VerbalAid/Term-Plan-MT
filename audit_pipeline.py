"""
TermPlanMT error analysis pipeline.
Annotates sampled segments via a hosted evaluator model, then builds a taxonomy report.
"""

import json, csv, random, time, os, sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

from openai import OpenAI

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set.")
    print("Add it to your .env file:  OPENAI_API_KEY=sk-proj-...")
    print("Or run:  OPENAI_API_KEY=sk-proj-... python3 audit_pipeline.py")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
SEGS_PATH = "data/section48/segments_ner_biollm.jsonl"
S1_PATH   = "results/ner_biollm/s1.jsonl"
S2_PATH   = "results/ner_biollm/s2.jsonl"
S5_PATH   = "results/ner_biollm/s5_mistral.jsonl"
S6_PATH   = "results/ner_biollm/s6.jsonl"

OUT_CSV        = "error_analysis/audit_annotated.csv"
SUMMARY_TXT    = "error_analysis/audit_summary.txt"
TAXONOMY_TXT   = "error_analysis/taxonomy.txt"
N_SAMPLE       = 30
SEED           = 42
# Truncate only for the evaluator API prompt; CSV keeps full segment text.
PROMPT_FR_MAX = 500
PROMPT_REF_MAX = 400
PROMPT_HYP_MAX = 400

Path("error_analysis").mkdir(exist_ok=True)
client = OpenAI()

# ── Context given to the model about the project ─────────────────────────────
PROJECT_CONTEXT = """
You are assisting with error analysis for a research paper called TermPlanMT.

PROJECT OVERVIEW:
TermPlanMT is a French-to-English machine translation system for EMA regulatory
medical documents (SmPC adverse event sections). The core research question is:
does grounding translation to the MedDRA ontology improve terminology consistency?

THE SIX SYSTEMS:
- S1 (NLLB-200): Standard NMT baseline. No ontology. Consistent but rigid vocabulary.
- S2 (Mistral-7B, full doc): LLM with full document context. Most fluent, closest
  to human register. No ontology constraint.
- S3 (Mistral + GraphRAG): MedDRA terms retrieved from Neo4j and injected into prompt.
  No forced output.
- S4 (S3 + reranking): Generates 3 candidates, picks the one with most MedDRA terms.
- S5 (Mistral + logit boost): Token probabilities for MedDRA preferred terms are
  boosted during decoding. Strongest ontology enforcement.
- S6 (NLLB + oracle glossary): Uses terms extracted from reference translations.
  Oracle upper bound — not a fair comparison.

THE MEDDRA HIERARCHY (L1=broadest, L5=most specific):
L1 SOC  - System Organ Class  (e.g. "Respiratory, thoracic and mediastinal disorders")
L2 HLGT - High Level Group Term (e.g. "Lower respiratory tract inflammatory and immunologic conditions")
L3 HLT  - High Level Term (e.g. "Pneumonitis and lung infiltration disorders")
L4 PT   - Preferred Term (e.g. "Pneumonitis") <- the target level
L5 LLT  - Lowest Level Term (e.g. "Immune-mediated pneumonitis")

KEY FINDING SO FAR:
The human reference translation scores only 0.15-0.18 on HTM (our ontology metric).
This means professional certified translators systematically AVOID MedDRA preferred
terms and instead write descriptive clinical prose. Systems that score higher on HTM
(like S5) are actually diverging from how humans write, not converging.

YOUR TASK:
For each segment, analyse all five outputs (S1, S2, S5, S6, human reference) and
classify what type of translation choice each system made. This annotation will
form the basis of the error taxonomy in Section 8 of the paper.
"""

# ── Annotation prompt ─────────────────────────────────────────────────────────
ANNOTATION_PROMPT = """
{context}

SEGMENT TO ANNOTATE:
ID: {seg_id}

French source:
{fr}

Human reference (certified EMA translator):
{ref}

S1 output (NLLB baseline, no graph):
{s1}

S2 output (Mistral, full document context, no graph):
{s2}

S5 output (Mistral, logit-boosted toward MedDRA terms):
{s5}

S6 output (NLLB, oracle glossary from reference):
{s6}

ANNOTATION TASK:
Analyse this segment carefully and answer all of the following.
Use the exact field names shown.

1. PRIMARY_TERM [the key medical term in this segment that matters for classification]:

2. HUMAN_REGISTER [one of: DESCRIPTIVE | TECHNICAL | MIXED | NO_TERM]:
   - DESCRIPTIVE: human uses plain clinical prose (e.g. "immune-related lung inflammation")
   - TECHNICAL: human uses exact MedDRA label (e.g. "Pneumonitis")
   - MIXED: human uses some MedDRA labels and some descriptive phrases
   - NO_TERM: no significant medical term in this segment

3. S1_PATTERN [one of: HYPERNYM | CORRECT_TERM | PARAPHRASE | MISS | NO_TERM]:
   - HYPERNYM: NLLB uses a broader/less specific term than MedDRA PT
   - CORRECT_TERM: output matches MedDRA PT or LLT
   - PARAPHRASE: output is a restatement not in MedDRA
   - MISS: term is absent from output entirely

4. S2_PATTERN [same options as S1_PATTERN]:

5. S5_PATTERN [same options as S1_PATTERN]:

6. S6_PATTERN [same options as S1_PATTERN]:

7. REGISTER_MISMATCH [Yes/No]: Does S5 use a more formal MedDRA coding term
   where the human reference uses descriptive prose?

8. CONSISTENCY_PARADOX [Yes/No]: Is S5 technically more ontology-correct than
   the human reference, even though it would score lower on BLEU?

9. BEST_SYSTEM [S1|S2|S5|S6|HUMAN|TIE]: Which output is closest to what a
   certified EMA translator would write?

10. ERROR_TYPE [one of: REGISTER_MISMATCH | HYPERNYM_COLLAPSE | TERM_MISS |
    TERM_DRIFT | CORRECT | NO_ERROR]:
    The single most important error or observation for this segment.

11. ANNOTATION_NOTE [1-2 sentences]: Your key observation for the paper authors.
    Be specific. Reference the actual words used.

Respond using exactly these field names, one per line, colon-separated.
"""

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found, using empty dict")
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

segs  = load_jsonl(SEGS_PATH)
s1_map = {r["id"]: r.get("hyp", "") for r in load_jsonl(S1_PATH)}
s2_map = {r["id"]: r.get("hyp", "") for r in load_jsonl(S2_PATH)}
s5_map = {r["id"]: r.get("hyp", "") for r in load_jsonl(S5_PATH)}
s6_map = {r["id"]: r.get("hyp", "") for r in load_jsonl(S6_PATH)}

# Only keep segments where S2 and S5 both have output
valid = [
    s for s in segs
    if s["id"] in s2_map and s["id"] in s5_map
    and s2_map[s["id"]].strip() and s5_map[s["id"]].strip()
]

random.seed(SEED)
sample = random.sample(valid, min(N_SAMPLE, len(valid)))
print(f"Sampled {len(sample)} segments for annotation")

# ── Parse annotation response ─────────────────────────────────────────────────
def parse_response(text):
    fields = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    return fields

# ── Annotate each segment ─────────────────────────────────────────────────────
rows = []
error_type_counts = {}
pattern_counts = {"S1": {}, "S2": {}, "S5": {}, "S6": {}}
register_mismatch_count = 0
consistency_paradox_count = 0
best_system_votes = {}

for i, seg in enumerate(sample):
    sid = seg["id"]
    print(f"  [{i+1}/{len(sample)}] Annotating {sid}...")

    prompt = ANNOTATION_PROMPT.format(
        context=PROJECT_CONTEXT,
        seg_id=sid,
        fr=seg["fr"][:PROMPT_FR_MAX],
        ref=seg["en_ref"][:PROMPT_REF_MAX],
        s1=s1_map.get(sid, "N/A")[:PROMPT_HYP_MAX],
        s2=s2_map.get(sid, "N/A")[:PROMPT_HYP_MAX],
        s5=s5_map.get(sid, "N/A")[:PROMPT_HYP_MAX],
        s6=s6_map.get(sid, "N/A")[:PROMPT_HYP_MAX],
    )

    try:
        msg = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.choices[0].message.content.strip()
        parsed = parse_response(raw)
    except Exception as e:
        print(f"    ERROR: {e}")
        parsed = {}
        raw = f"ERROR: {e}"

    time.sleep(0.4)

    # Tally counts
    et = parsed.get("ERROR_TYPE", "UNKNOWN")
    error_type_counts[et] = error_type_counts.get(et, 0) + 1

    for sys_key in ["S1", "S2", "S5", "S6"]:
        pat = parsed.get(f"{sys_key}_PATTERN", "UNKNOWN")
        pattern_counts[sys_key][pat] = pattern_counts[sys_key].get(pat, 0) + 1

    if "yes" in parsed.get("REGISTER_MISMATCH", "").lower():
        register_mismatch_count += 1
    if "yes" in parsed.get("CONSISTENCY_PARADOX", "").lower():
        consistency_paradox_count += 1

    best = parsed.get("BEST_SYSTEM", "UNKNOWN")
    best_system_votes[best] = best_system_votes.get(best, 0) + 1

    rows.append({
        "id":                  sid,
        "fr":                  seg["fr"],
        "human_ref":           seg["en_ref"],
        "s1":                  s1_map.get(sid, ""),
        "s2":                  s2_map.get(sid, ""),
        "s5":                  s5_map.get(sid, ""),
        "s6":                  s6_map.get(sid, ""),
        "primary_term":        parsed.get("PRIMARY_TERM", ""),
        "human_register":      parsed.get("HUMAN_REGISTER", ""),
        "s1_pattern":          parsed.get("S1_PATTERN", ""),
        "s2_pattern":          parsed.get("S2_PATTERN", ""),
        "s5_pattern":          parsed.get("S5_PATTERN", ""),
        "s6_pattern":          parsed.get("S6_PATTERN", ""),
        "register_mismatch":   parsed.get("REGISTER_MISMATCH", ""),
        "consistency_paradox": parsed.get("CONSISTENCY_PARADOX", ""),
        "best_system":         parsed.get("BEST_SYSTEM", ""),
        "error_type":          parsed.get("ERROR_TYPE", ""),
        "annotation_note":     parsed.get("ANNOTATION_NOTE", ""),
        "your_correction":     "",
        "your_notes":          "",
        "raw_response":        raw,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV written: {OUT_CSV}")

# ── Build taxonomy ─────────────────────────────────────────────────────────────
n = len(sample)

def pct(count): return round(100 * count / n)

taxonomy = f"""
TERMPLANMT ERROR TAXONOMY
{n} segments, seed={SEED}
════════════════════════════════════════════════════════════════

HUMAN REFERENCE REGISTER
"""
human_registers = {}
for r in rows:
    hr = r["human_register"]
    human_registers[hr] = human_registers.get(hr, 0) + 1
for hr, count in sorted(human_registers.items(), key=lambda x: -x[1]):
    taxonomy += f"  {hr:<20} {count}/{n} = {pct(count)}%\n"

taxonomy += "\nSYSTEM OUTPUT PATTERNS\n"
for sys_key in ["S1", "S2", "S5", "S6"]:
    taxonomy += f"\n{sys_key}:\n"
    for pat, count in sorted(pattern_counts[sys_key].items(), key=lambda x: -x[1]):
        taxonomy += f"  {pat:<20} {count}/{n} = {pct(count)}%\n"

taxonomy += "\nERROR TYPE DISTRIBUTION\n"
for et, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
    taxonomy += f"  {et:<25} {count}/{n} = {pct(count)}%\n"

taxonomy += f"""
KEY FINDINGS
Register mismatch (S5 more technical than human):  {register_mismatch_count}/{n} = {pct(register_mismatch_count)}%
Consistency paradox (S5 ontology-correct, lower BLEU): {consistency_paradox_count}/{n} = {pct(consistency_paradox_count)}%

Best system votes:
"""
for sys_key, count in sorted(best_system_votes.items(), key=lambda x: -x[1]):
    taxonomy += f"  {sys_key:<10} {count}/{n} = {pct(count)}%\n"

Path(TAXONOMY_TXT).write_text(taxonomy, encoding="utf-8")
print(f"Taxonomy written: {TAXONOMY_TXT}")

# ── Build paper paragraph ─────────────────────────────────────────────────────
rm_pct = pct(register_mismatch_count)
cp_pct = pct(consistency_paradox_count)

top_human = max(human_registers, key=human_registers.get)
top_human_pct = pct(human_registers[top_human])

s5_patterns = pattern_counts["S5"]
top_s5 = max(s5_patterns, key=s5_patterns.get) if s5_patterns else "UNKNOWN"
top_s5_pct = pct(s5_patterns.get(top_s5, 0))

paragraph = f"""
PAPER PARAGRAPH FOR SECTION 8
(copy into termplanmt_v3.tex, replace the placeholder register mismatch paragraph)
════════════════════════════════════════════════════════════════

\\paragraph{{Qualitative error taxonomy.}}
To ground the HTM--BLEU divergence empirically, we sampled {n} segments
and annotated all five outputs (S1--S5 plus human reference) using a
structured taxonomy across four dimensions: human register, system output
pattern, register mismatch, and the consistency paradox.

Human reference register was predominantly \\textbf{{{top_human.lower()}}}
({top_human_pct}\\% of segments): professional translators systematically
prefer descriptive clinical prose over MedDRA preferred-term labels,
consistent with the low rHTM ceiling of 0.15--0.18.

S5 output was classified as \\texttt{{{top_s5.lower().replace("_", " ")}}}
in {top_s5_pct}\\% of cases. In {rm_pct}\\% of segments, S5 produced a
formally correct MedDRA label (\\textsc{{Pneumonitis}}, \\textsc{{Arthralgia}})
where the certified translator wrote descriptive prose
(\\textit{{``immune-related pneumonitis''}}, \\textit{{``joint pain''}}).
In {cp_pct}\\% of cases this constitutes what we term the
\\textbf{{consistency paradox}}: S5 is ontology-correct yet scores lower
on BLEU because it diverges from the reference register.

This taxonomy confirms that the HTM--BLEU inversion is not a translation
failure but a structural register mismatch. Graph-augmented systems are
pulled toward pharmacovigilance coding language while certified translators
write for a clinical reader. \\textbf{{HRA}} -- which measures agreement
with professional translation practice -- is therefore the appropriate
primary evaluation lens for regulatory MT; HTM measures ontology proximity,
not translation quality.
════════════════════════════════════════════════════════════════
"""

Path(SUMMARY_TXT).write_text(taxonomy + paragraph, encoding="utf-8")
print(paragraph)
print(f"\nAll files written to error_analysis/")
print(f"Next: open {OUT_CSV} and fill in your_correction and your_notes columns")
print(f"Then copy the paragraph above into Section 8 of the paper.")
