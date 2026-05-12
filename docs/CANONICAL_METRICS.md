## Canonical metrics decision (paper source of truth)

This document is the single source of truth for which metrics/numbers appear in the paper.

### Segment exclusion policy (canonical)

- **Canonical policy**: **Include** segment `48_028` (table segment).
- **Evidence**: Standalone rHTM recomputation using `compute_htm_a(..., results_jsonl=None)` gives the same rHTM values with and without `48_028`.

Standalone rHTM (T1) checks:

| Run | ner_biollm rHTM | ner_biollm_ft rHTM |
|-----|-----------------|---------------------|
| A (exclude 48_028) | 0.151 | 0.178 |
| B (include 48_028) | 0.151 | 0.178 |

Counts used (standalone runs):
- **A/B ner_biollm**: 126/127 segment rows; 168 unique terms; 69 grounded terms
- **A/B ner_biollm_ft**: 126/127 segment rows; 242 unique terms; 112 grounded terms

### Canonical rHTM values for HTM-A (alias tiers)

Using `pipeline/metrics/htm_alias.py` (alias expansion T1/T2/T3) and treating `en_ref` as the hypothesis:

| Tier | ner_biollm rHTM | ner_biollm_ft rHTM | delta_from_T1 |
|------|-----------------|---------------------|---------------|
| T1 | 0.151 | 0.178 | --- |
| T2 | 0.151 | 0.178 | +0.000 / +0.000 |
| T3 | 0.151 | 0.178 | +0.000 / +0.000 |

Decision:
- **CONFIRMED**: alias expansion does not close the metric-practice gap. The mismatch is structural, not ontological.

### Prompt contamination impact (S3/S4/S5-Mistral, finetuned condition)

Cleaned versions were generated under `results/ner_biollm_finetuned/`:
- `s3_clean.jsonl`
- `s4_clean.jsonl`
- `s5_mistral_clean.jsonl`

Cleaning rule:
- If `hyp` contains any of: `Translation:`, `MedDRA LLT`, `### Task`, `### Instruction`, replace `hyp` with `__CONTAMINATED__` (kept row; scored as empty hypothesis).

Raw vs cleaned corpus scores (sacrebleu against `en_ref`):

| System | BLEU raw | BLEU clean | chrF raw | chrF clean | rows_contaminated |
|--------|----------|------------|----------|------------|-------------------|
| S3 | 15.36 | 17.49 | 34.60 | 32.01 | 18 |
| S4 | 15.66 | 17.52 | 34.92 | 32.04 | 18 |
| S5-M | 15.98 | 17.50 | 34.85 | 32.35 | 18 |

Materiality definition (paper):
- **Material** if \(\\Delta\\) BLEU > 1.0 **or** \(\\Delta\\) chrF > 0.5.

Decision:
- **Contamination materially changes** these systems’ scores (BLEU changes by > 1.0 and chrF changes by > 0.5).
- For paper claims that rely on S3/S4/S5-Mistral textual quality, **use the cleaned scores** (or explicitly state that raw outputs include prompt leakage and provide both).

### Which numbers go in the paper (and why)

- **HTM ceilings / rHTM**: use the **standalone HTM-A** values above for alias-tier discussion, and explicitly note they differ from the committed `scores_summary.csv` `htm_en_ref_dataset` values (different evaluation path).
- **S3/S4/S5-Mistral text metrics (BLEU/chrF)**: use **cleaned** numbers or present both raw vs clean; raw scores are not reliable due to prompt leakage.

### S6 (glossary oracle) — eval manifest

- **Result files**: `s6.jsonl` (NLLB + glossary PhraseLogitBoost), `s6_mistral.jsonl` (Mistral + glossary prompt + boost).
- **`mistral_clean` set**: S6 uses the same filenames as `standard` (no MedDRA prompt contamination from graph metadata; still use sensible qualitative checks on outputs).
- **Glossary**: build a draft with `tools/data/build_gold_terms_from_parallel_ner.py`; **hand-review** before interpreting S6 as an oracle vs S5.
