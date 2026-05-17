# Upstream NER fine-tuning (Condition B)

This document is the **canonical reference** for the fine-tuned NER condition used in the paper and poster (`results/ner_biollm_finetuned/`, segment file `data/section48/segments_ner_unsloth_full.jsonl`).

## Two NER conditions

| | **Condition A (baseline)** | **Condition B (fine-tuned)** |
|---|---------------------------|------------------------------|
| **Directory** | `results/ner_biollm/` | `results/ner_biollm_finetuned/` |
| **Segments** | `segments_ner_biollm.jsonl` | `segments_ner_unsloth_full.jsonl` |
| **Model** | BioMistral-7B, zero-shot French JSON-list prompt | BioMistral-7B + Unsloth 4-bit LoRA (QUAERO SFT) |
| **CCR (126 eval segs)** | 39.0% (184 / 472 spans grounded) | 36.4% (211 / 580 spans grounded) |
| **Known artefact** | *hypothyroïdie* in 92/127 full-corpus segments | Artefact removed |

**Important:** CCR *rate* can fall while **grounded span count rises** (580 vs 472 extracted spans). Do not report fine-tuning as “higher CCR %” without this context.

## Training data (QUAERO BRAT)

- **Corpus:** [QUAERO French Medical](https://huggingface.co/datasets/Dr-BERT/QUAERO) in **BRAT** format (not MedDRA hierarchy labels).
- **Splits used:** `corpus/train/EMEA` + `corpus/train/MEDLINE` (combined before shuffling).
- **Entity types kept:** `DISO`, `CHEM`, `PROC` (disorders, chemicals/drugs, procedures); all other types → `O`.
- **Granularity:** one training example per **line** in each `.txt` file (sentence- or title-level), with BRAT spans clipped to that line.

### Sentence counts

| Set | Sentences |
|-----|-----------|
| EMEA train only | 706 |
| **EMEA + MEDLINE train (combined)** | **1,540** |
| Typical 90/10 train/val split | ~1,386 train · ~154 val |

Obtain QUAERO under its licence and place it at `data/QUAERO_FrenchMed/` (gitignored). Re-count after download:

```bash
PYTHONPATH=. python tools/count_quaero_brat_sentences.py \
  --emea data/QUAERO_FrenchMed/corpus/train/EMEA \
  --medline data/QUAERO_FrenchMed/corpus/train/MEDLINE
```

## Training hyperparameters (paper / poster)

- **Base model:** `BioMistral/BioMistral-7B`
- **Method:** Unsloth 4-bit QLoRA, Alpaca instruction SFT (French sentence → JSON list of entity strings)
- **LoRA:** rank 16, α 32, dropout 0.05 (all linear projections)
- **Schedule:** 3 epochs, batch 4, gradient accumulation 4, lr 2×10⁻⁴, cosine, 10 warmup steps, response-only loss (`SFTTrainer`)

Fine-tuning scripts and merged weights are **not** stored in this repository (see [`models/README.md`](../models/README.md)). Regenerate locally, then run NER inference to produce `segments_ner_unsloth_full.jsonl`.

## Downstream MT effect (126 segments, segment `48_028` excluded)

Paired bootstrap ΔBLEU (baseline NER → fine-tuned NER, Mistral graph outputs contamination-filtered):

| System | ΔBLEU (approx.) |
|--------|-----------------|
| S3 GraphRAG | +7.07 |
| S4 rerank | +7.42 |
| S5 Mistral + boost | +7.16 |
| S1, S2, S5 NLLB, S6 | ≈ 0 |

Full table: `results/ner_biollm_finetuned/figures/bleu_delta_bootstrap_95ci.csv`.

## Decoding boost surface rates (S5 / S6)

Post-hoc check: does the **target phrase** appear verbatim in the hypothesis? (Lock map was not logged at inference.)

| System | What is checked | Verbatim surface rate |
|--------|-----------------|------------------------|
| **S5** | French NER span in S5 output (weak proxy) | **34%** (159/472) |
| **S5** | English MedDRA phrase passed to logit boost | **~38%** (recompute with graph) |
| **S6** | Oracle English glossary gloss in S6 NLLB output | **36%** (142/397) in `error_analysis/boost_success_rate.txt` |

**3+ token** MedDRA / glossary strings: **0%** verbatim surface in both S5 and S6.

Details: [`error_analysis/boost_success_rate.txt`](../error_analysis/boost_success_rate.txt).

## What not to claim without new experiments

The following poster figures are **not** supported by committed evaluation artifacts:

- Token-level F₁ 0.81 → 0.93 on SmPC §4.8
- “15K” QUAERO sentences or “MedDRA-level” training labels
- +14.8% entity precision, 71% L4→L5 promotion, “3× faster than full fine-tuning”

Use the metrics in this file and [`docs/RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) instead.
