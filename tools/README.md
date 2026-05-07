# CLI scripts (`tools/`)

Entrypoints are grouped **by workflow** (not a flat `tools/*.py` list). Run from the repo root with `PYTHONPATH=.` (or your venv’s `python`).

## `tools/data/` — corpora, MedDRA graph, ontology exports

| Script | Role |
|--------|------|
| `prepare_data.py` | PDF → aligned segment JSONL. |
| `extract_meddra.py`, `build_graph.py` | MedDRA ASCII → Neo4j `:Concept` graph. |
| `build_gold_terms_from_parallel_ner.py` | Optional `gold_terms` JSON for graph seeding. |
| `export_full_ontology_ner_sft_jsonl.py` | Full ontology → Alpaca JSONL (NER / hierarchy formats). |
| `split_ontology_sft_jsonl.py` | Train/val split for ontology JSONL. |
| `build_ontology_sft.py` | MedDRA-only instruction pairs (no sentences) → `data/sft/ontology_{train,val}.jsonl`. |

## `tools/pipeline/` — run translation ladder (S1–S5)

| Script | Role |
|--------|------|
| `run_pipeline.py` | Run systems → `results/<condition>/*.jsonl`. |

## `tools/eval/` — metrics & paper figures

| Script | Role |
|--------|------|
| `evaluate.py` | BLEU / chrF / COMET + CCR / HTM / rHTM (+ optional vector HTM columns). |
| `plot_figures.py` | Figures + `scores_summary.csv` for one `results/` tree. |
| `run_eval_plot_matrix.py` | Batch `evaluate` + `plot` over `EVAL_RERUN_PROFILES` (used by `rerun_all.sh`). |
| `plot_cross_ner_dashboard.py` | Cross-condition comparison plots. |
| `compare_htm_vector_thresholds.py` | String vs vector HTM threshold analysis. |

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
