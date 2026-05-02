# Error analysis (generated CSVs)

CSV files here are generated (not source of truth). Regenerate after pipeline runs:

```bash
PYTHONPATH=. python scripts/sample_errors_for_annotation.py \
  --out-csv error_analysis/error_review_50.csv --n 50 --annotate-backend ollama
```

BioMistral NER only (prompt + fine-tuned), worst **20** segments by sentence chrF, same CSV schema / ontology flags (optional `--ollama-model`, default `llama3.2`):

```bash
PYTHONPATH=. python scripts/sample_errors_for_annotation.py \
  --out-csv error_analysis/error_review_biollm_ft_top20.csv \
  --n 20 \
  --ner-glob ner_biollm \
  --ner-glob ner_biollm_finetuned \
  --annotate-backend ollama
```

Older rows tied to removed conditions live under [error_analysis/legacy/](legacy/).

Schema: [docs/error_analysis/schema.md](../docs/error_analysis/schema.md).

Snapshot everything (results + this folder + planning locks): [scripts/archive_results_snapshot.sh](../scripts/archive_results_snapshot.sh).

Optional ambiguous-key audit (two NER JSONLs vs `planning_locks.json`): [scripts/report_ambiguous_grounding.py](../scripts/report_ambiguous_grounding.py) → `error_analysis/ambiguous_grounding_report.csv`. See [docs/error_analysis/ambiguous_grounding.md](../docs/error_analysis/ambiguous_grounding.md).
