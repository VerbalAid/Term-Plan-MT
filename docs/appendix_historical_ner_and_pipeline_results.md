# Appendix — Historical term extraction conditions and full MT pipeline results

This note is for **thesis appendix / supplementary material**: it records **which French term extractors** were used, how they map to **segment JSONL** files, and **corpus-level metrics** for the **full evaluation stack** (same six MT outputs under each NER condition where available).

**Provenance.** Rows marked **archived** come from the repository **immediately before** the main driver was narrowed to two NER conditions (`git` tree at parent of `11783a0`, i.e. `11783a0^`). Rows marked **current** are from the working tree as of the last regeneration of `results/*/figures/scores_summary.csv`. **Vector / string CCR** numbers are copied from `data/section48/vector_ccr_all_models.json` (full six-way snapshot at `11783a0^`; the committed file was later trimmed to the two primary segment sources).

**Protocol caveats.**

- Each NER condition uses its **own** segment JSONL: **number of NER spans** (`total` in the CCR table) differs by extractor, so **`ccr_dataset` in `scores_summary.csv` is not comparable across rows** as a standalone “quality” score—it is grounded-span coverage **for that extractor’s span set**.
- Metrics in the CSVs follow whatever **`scripts/evaluate.py`** / **`scripts/plot_results.py`** settings were used when the CSV was written (typically **string** grounding for HTM unless you re-ran with `--grounding-mode vector`).
- The main reproduce path **excludes segment `48_028` by default** (same id as in `rerun_all.sh`). Older CSVs in git history may pre-date that convention—re-run evaluation for strict parity.

---

## 1. MT pipeline outputs evaluated (same under every NER condition)

These are the **hypothesis JSONL** labels from `pipeline/metrics/eval_manifest.py` (`EVAL_FILES`):

| Label | Role |
|--------|------|
| **s1** | S1 — NLLB translation |
| **s2** | S2 — Mistral document-level |
| **s3** | S3 — GraphRAG (grounded planning context) |
| **s4** | S4 — Reranking |
| **s5** | S5 — NLLB + logit boost |
| **s5_mistral** | S5 — Mistral + logit boost (alternate decoder) |

So each “full pipeline” row in the tables below is **one of these six systems**, evaluated on the **same reference** and **same segment ids** as the other systems **within that NER condition** (but not necessarily the same segment ids as another NER condition, if the JSONL segment count differs).

---

## 2. Term extraction (NER) conditions — segment files and archive location

| Condition (results folder) | Segment JSONL (typical) | What it is | Where it lives now |
|-----------------------------|-------------------------|------------|---------------------|
| **ner_baseline** | `segments_ner_baseline.jsonl` | Hugging Face **CamemBERT** token NER on Section 4.8 FR text (`Jean-Baptiste/camembert-ner` style workflow; see `scripts/prepare_data.py`). | **[`archive/data/section48/segments_ner_baseline.jsonl`](../archive/data/section48/segments_ner_baseline.jsonl)** |
| **ner_finetuned** | `segments_ner_finetuned.jsonl` (historically symlinked to generic `segments_ner.jsonl`) | **Fine-tuned CamemBERT** on QUAERO-style labels (same sentence alignment as generic NER export in older runs). | Symlink name archived as **[`archive/data/section48/segments_ner_finetuned.jsonl`](../archive/data/section48/segments_ner_finetuned.jsonl)**; generic **[`data/section48/segments_ner.jsonl`](../data/section48/segments_ner.jsonl)** still used by `prepare_data.py` |
| **ner_biollm** | `segments_ner_biollm.jsonl` | **BioMistral-7B** JSON-list prompting (`experiments/french_medical_ner/biomistral_prompt_ner.py`). | **`data/section48/segments_ner_biollm.jsonl`** |
| **ner_biollm_finetuned** | `segments_ner_unsloth.jsonl` or `segments_ner_unsloth_full.jsonl` | **Fine-tuned BioMistral** NER (Unsloth LoRA merge). | **`data/section48/`** (see `rerun_all.sh`) |
| **Mistral Instruct tagging** | `segments_ner_mistral_instruct.jsonl` | Experimental FR term list using **Mistral Instruct** (not on main `rerun_all.sh`). | **[`archive/data/section48/segments_ner_mistral_instruct.jsonl`](../archive/data/section48/segments_ner_mistral_instruct.jsonl)** |
| **Llama 3 tagging** | `segments_ner_llama3.jsonl` | Experimental FR term list using **Llama 3** (not on main `rerun_all.sh`). | **[`archive/data/section48/segments_ner_llama3.jsonl`](../archive/data/section48/segments_ner_llama3.jsonl)** |

**Full five-stage MT stack** in this project is **S1 → S2 → S3 → S4 → S5** (with **S5** available as **NLLB** or **Mistral** logit-boost variant above). All conditions that shipped `results/<ner_*>/s*.jsonl` were evaluated through that same **stage graph**; only the **input `terms[]`** changed with the segment JSONL.

---

## 3. Corpus metrics — `scores_summary.csv` (archived CamemBERT runs)

Values are **exactly** as in `results/<condition>/figures/scores_summary.csv` at `11783a0^`. Columns: **BLEU**, **chrF++**, **HTM**, **`ccr_dataset`**, mean seconds / segment, p95 seconds.

### 3.1 `ner_baseline` (CamemBERT HF NER)

| label | display | bleu | chrf | htm | ccr_dataset | mean_s | p95_s |
|--------|---------|------|------|-----|----------------|--------|-------|
| s1 | S1 NLLB | 18.893 | 31.897 | 0.304 | 0.266 | 0.741 | 3.392 |
| s2 | S2 Mistral (doc) | 19.656 | 35.503 | 0.304 | 0.266 | 7.088 | 16.179 |
| s3 | S3 GraphRAG | 15.390 | 34.127 | 0.321 | 0.266 | 3.140 | 11.930 |
| s4 | S4 rerank | 16.970 | 34.714 | 0.339 | 0.266 | 9.326 | 34.137 |
| s5 | S5 NLLB + boost | 18.893 | 31.898 | 0.304 | 0.266 | 0.754 | 3.544 |
| s5_mistral | S5 Mistral + boost | 16.513 | 34.868 | 0.321 | 0.266 | 3.178 | 11.474 |

### 3.2 `ner_finetuned` (QUAERO-style fine-tuned CamemBERT NER)

| label | display | bleu | chrf | htm | ccr_dataset | mean_s | p95_s |
|--------|---------|------|------|-----|----------------|--------|-------|
| s1 | S1 NLLB | 18.893 | 31.897 | 0.304 | 0.266 | 0.741 | 3.392 |
| s2 | S2 Mistral (doc) | 19.656 | 35.503 | 0.304 | 0.266 | 7.088 | 16.179 |
| s3 | S3 GraphRAG | 15.945 | 35.222 | 0.339 | 0.266 | 4.040 | 17.450 |
| s4 | S4 rerank | 16.373 | 35.480 | 0.339 | 0.266 | 15.702 | 65.263 |
| s5 | S5 NLLB + boost | 17.267 | 30.484 | 0.304 | 0.266 | 0.929 | 3.649 |
| s5_mistral | S5 Mistral + boost | 15.838 | 35.047 | 0.339 | 0.266 | 3.051 | 13.883 |

---

## 4. Corpus metrics — current primary conditions (BioMistral)

### 4.1 `ner_biollm` (BioMistral JSON-list NER)

| label | display | bleu | chrf | htm | ccr_dataset | mean_s | p95_s |
|--------|---------|------|------|-----|----------------|--------|-------|
| s1 | S1 NLLB | 18.893 | 31.897 | 0.304 | 0.354 | 0.741 | 3.392 |
| s2 | S2 Mistral (doc) | 19.656 | 35.503 | 0.304 | 0.354 | 7.088 | 16.179 |
| s3 | S3 GraphRAG | 10.788 | 32.646 | 0.375 | 0.354 | 4.741 | 15.019 |
| s4 | S4 rerank | 10.958 | 32.341 | 0.375 | 0.354 | 26.291 | 69.496 |
| s5 | S5 NLLB + boost | 17.341 | 30.553 | 0.304 | 0.354 | 0.811 | 3.659 |
| s5_mistral | S5 Mistral + boost | 10.824 | 32.637 | 0.375 | 0.354 | 4.949 | 15.195 |

### 4.2 `ner_biollm_finetuned` (Unsloth fine-tuned BioMistral NER)

| label | display | bleu | chrf | htm | ccr_dataset | mean_s | p95_s |
|--------|---------|------|------|-----|----------------|--------|-------|
| s1 | S1 NLLB | 18.893 | 31.897 | 0.304 | 0.266 | 0.741 | 3.392 |
| s2 | S2 Mistral (doc) | 19.656 | 35.503 | 0.304 | 0.266 | 7.088 | 16.179 |
| s3 | S3 GraphRAG | 14.206 | 34.419 | 0.375 | 0.266 | 3.569 | 14.202 |
| s4 | S4 rerank | 13.744 | 34.165 | 0.375 | 0.266 | 11.422 | 38.576 |
| s5 | S5 NLLB + boost | 18.131 | 31.418 | 0.304 | 0.266 | 0.815 | 3.491 |
| s5_mistral | S5 Mistral + boost | 14.714 | 34.629 | 0.375 | 0.266 | 3.530 | 14.353 |

---

## 5. NER span grounding (CCR) — string vs vector index (historical snapshot)

From **`vector_ccr_all_models.json`** at **`11783a0^`**. **CCR** = fraction of extracted FR spans that received a non-null graph grounding in the reported mode. **Totals** differ because each JSONL contains a different number of NER spans.

| Segment file | String CCR | Vector CCR | Spans (total) | String grounded | Vector grounded |
|--------------|------------|------------|---------------|-----------------|-----------------|
| `segments_ner_baseline.jsonl` | 0.052 | 0.862 | 58 | 3 | 50 |
| `segments_ner_finetuned.jsonl` | 0.266 | 0.998 | 433 | 115 | 432 |
| `segments_ner_biollm.jsonl` | 0.354 | 0.987 | 472 | 167 | 466 |
| `segments_ner_mistral_instruct.jsonl` | 0.222 | 0.999 | 1012 | 225 | 1011 |
| `segments_ner_llama3.jsonl` | 0.326 | 0.992 | 485 | 158 | 481 |
| `segments_ner_unsloth_full.jsonl` | 0.324 | 0.998 | 580 | 188 | 579 |

**Note.** In this repository snapshot, **Mistral-Instruct** and **Llama-3** segment lists were used for **grounding / CCR analysis**; a full **`scores_summary.csv`** bundle for **`results/ner_mistral_*`** / **`ner_llama*`** was **not** kept under `results/` in the archived tree checked here. If you regenerate full pipelines for those JSONLs, drop outputs under a new `results/<name>/` and re-run `scripts/evaluate.py` + `scripts/plot_results.py` to extend this appendix.

---

## 6. How to regenerate or verify

| Goal | Command / path |
|------|----------------|
| Re-run two-condition driver | `./rerun_all.sh` (see root `README.md`) |
| Re-evaluate one condition | `PYTHONPATH=. python scripts/evaluate.py --results-dir results/<condition> --segments data/section48/<segments>.jsonl` |
| Rebuild figures / `scores_summary.csv` | `PYTHONPATH=. python scripts/plot_results.py --results-dir … --segments …` |
| Ambiguous FR keys (two JSONLs + locks) | `PYTHONPATH=. python scripts/report_ambiguous_grounding.py` → `exports/ambiguous_grounding_report.csv` |
| Recover deleted `results/ner_*` CSV from git | e.g. `git show '11783a0^:results/ner_baseline/figures/scores_summary.csv'` |

---

## 7. Revision pointer (for your PDF)

Suggested short citation in the appendix:

> Supplementary tables reproduce `scores_summary.csv` and `vector_ccr_all_models.json` from the TermPlan-MT repository; CamemBERT rows correspond to commit **`11783a0^`** (parent of “strip main pipeline to ner_biollm + ner_biollm_finetuned”).

Adjust the commit hash if you cherry-pick or squash history after publication.
