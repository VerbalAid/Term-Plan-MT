# Error analysis (generated artefacts)

Files here are regenerated from the pipeline JSONLs and are not the primary source of truth.

## Contents

| Path | Description |
|------|-------------|
| `annotations/audit_annotated.csv` | LLM-assisted audit labels for sampled segments (S1/S2/S5/S6). |
| `annotations/system_pattern_annotations.csv` | Human register/pattern labels per system. |
| `annotations/error_review_50.csv` | 50 worst-chrF segment rows sampled for manual review. |
| `ambiguous_spans_top10.md` / `.csv` | NER spans where one French string matches multiple MedDRA concepts. |
| `ambiguous_grounding_report.csv` | Per-key ambiguity counts across both NER conditions. |
| `audit_summary.txt`, `taxonomy.txt` | Summaries from `audit_pipeline.py`. |

## Regeneration

Requires Neo4j running and `PYTHONPATH=.`:

```bash
# Ambiguous grounding examples
python archive/tools/error_analysis/list_ambiguous_spans.py --n 10

# Cross-condition ambiguity report
python archive/tools/error_analysis/report_ambiguous_grounding.py \
  --segments-a data/section48/segments_ner_biollm.jsonl \
  --segments-b data/section48/segments_ner_unsloth_full.jsonl

# Error sampling (no LLM)
python archive/tools/error_analysis/sample_errors_for_annotation.py \
  --out-csv error_analysis/annotations/error_review_50.csv --n 50 --annotate-backend none
```

> These scripts have been archived to `archive/tools/error_analysis/` and are not part of the core paper reproducibility path.
