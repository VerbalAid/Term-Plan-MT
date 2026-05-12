# CLI scripts (`tools/`)

Entrypoints are grouped **by workflow** (not a flat `tools/*.py` list). Run from the repo root with `PYTHONPATH=.` (or your venv’s `python`).

## `tools/data/` — corpora, MedDRA graph, ontology exports

| Script | Role |
|--------|------|
| `extract_meddra.py`, `build_graph.py` | MedDRA ASCII → Neo4j `:Concept` graph. |
| `build_gold_terms_from_parallel_ner.py` | **Draft** FR→EN `gold_glossary.json` for **S6** (review before oracle runs). |
| `export_full_ontology_ner_sft_jsonl.py` | Full ontology → Alpaca / Mistral JSONL (hierarchical or legacy). |
| `split_ontology_sft_jsonl.py` | Train/val/test split for ontology JSONL. |
| `patch_ontology_sft_hierarchy_jsonl.py` | Fix `soc`…`llt` in hierarchical JSONL from English `mdhier.asc`. |
| `mistral_hierarchical_jsonl_to_alpaca.py` | Reframe Mistral `[INST]` ontology lines as Alpaca `text`. |

## `tools/pipeline/` — run translation ladder (S1–S6)

| Script | Role |
|--------|------|
| `run_pipeline.py` | Run systems → `results/<condition>/*.jsonl`. **S6**: `--system s6`, `--glossary`, `--s6-backend` (`nllb` / `mistral` / `both` → `s6.jsonl` / `s6_mistral.jsonl`). |

## `tools/eval/` — metrics & paper figures

| Script | Role |
|--------|------|
| `evaluate.py` | BLEU / chrF / COMET + CCR / HTM / rHTM (+ optional vector HTM columns). |
| `plot_figures.py` | Figures + `scores_summary.csv` for one `results/` tree. |
| `run_eval_plot_matrix.py` | Batch `evaluate` + `plot` over `EVAL_RERUN_PROFILES` (used by `rerun_all.sh`). |
| `bootstrap_bleu_delta.py` | Paired segment bootstrap 95% CIs for corpus BLEU deltas between two `results/*/s*.jsonl` trees. |
| `plot_cross_ner_dashboard.py` | Cross-condition comparison plots. |
| `compare_htm_vector_thresholds.py` | String vs vector HTM threshold analysis. |

### Bootstrap CIs for delta BLEU (paper Table 5)

`bootstrap_bleu_delta.py` resamples **segment ids** (paired across conditions), recomputes corpus BLEU on each bootstrap draw, and forms **delta = BLEU (finetuned) − BLEU (baseline)**. It intersects ids present in both JSONL trees and uses the same `fluency_hypothesis_text` rule as `evaluate.py` (contaminated rows score as empty hypotheses).

Example (after `results/ner_biollm/s*.jsonl` exist and match the segment / exclusion policy used for the paper):

```bash
PYTHONPATH=. python tools/eval/bootstrap_bleu_delta.py \
  --baseline-dir results/ner_biollm \
  --finetuned-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl \
  --exclude-segment-ids "" \
  --baseline-eval-file-set standard \
  --finetuned-eval-file-set mistral_clean \
  --n-bootstrap 2000 \
  --out-csv results/ner_biollm_finetuned/figures/bleu_delta_bootstrap_95ci.csv
```

Use the **same** `--segments` and `--exclude-segment-ids` as `plot_figures.py` / `evaluate.py` for that condition so references and segment sets align.

## `tools/error_analysis/` — audits & annotation helpers

| Script | Role |
|--------|------|
| `list_ambiguous_spans.py` | Top ambiguous FR grounding examples → `error_analysis/`. |
| `report_ambiguous_grounding.py` | Key-level ambiguous grounding CSV. |
| `sample_errors_for_annotation.py` | Sample rows for human / LLM-assisted error review. |
| `analyse_terminology.py` | Drift, level distribution, missing terms, consistency CSVs. |

## `tools/admin/` — housekeeping

| Script | Role |
|--------|------|
| `archive_results_snapshot.sh` | Optional `tar.gz` of `results/`, `error_analysis/`, planning locks. |
