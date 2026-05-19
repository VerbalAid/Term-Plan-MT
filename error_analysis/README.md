# Error analysis (generated artefacts)

Files here are regenerated from the pipeline JSONLs and are not the primary source of truth.

## Annotations (`annotations/`)

| File | Description |
|------|-------------|
| `audit_annotated.csv` | LLM-assisted audit labels (register, patterns, best system). |
| `system_pattern_annotations_Darragh.csv` | Annotator D — register/pattern labels and notes. |
| `system_pattern_annotations_N.csv` | Annotator N (Excel workbook saved as `.csv`). |
| `error_review_50.csv` | 50 worst-chrF segments sampled for manual review. |

## Inter-annotator agreement (regenerate)

```bash
PYTHONPATH=. python tools/inter_annotator_kappa.py
```

After resolving tied `best_system` choices in D’s sheet:

```bash
python tools/resolve_annotation_ties.py
```

| Output | Description |
|--------|-------------|
| `kappa_overall.png` | **Main chart** — pooled Cohen’s κ (D vs N, D vs LLM, N vs LLM, Human vs LLM). |
| `kappa_by_field.png` | Per-field κ detail (all comparisons). |
| `kappa_overall_summary.csv` | All κ values: per-field, macro-mean, and overall pooled. |
| `best_system_overall.png` | Best-translation agreement (D, N, LLM). |
| `best_system_agreement.png` | Count of best-system picks per rater. |
| `best_system_agreement.csv` | Per-segment raw/adjudicated choices and agreement flags. |
| `adjudication_decision_table.csv` | Tie-breaking rules and final adjudicated labels (supplementary). |

## Other outputs

| Path | Description |
|------|-------------|
| `ambiguous_spans_top10.md` / `.csv` | NER spans where one French string matches multiple MedDRA concepts. |
| `ambiguous_grounding_report.csv` | Per-key ambiguity counts across both NER conditions. |
| `audit_summary.txt`, `taxonomy.txt` | Summaries from `audit_pipeline.py`. |

## Regeneration (archived tools)

Requires Neo4j running and `PYTHONPATH=.`:

```bash
python archive/tools/error_analysis/list_ambiguous_spans.py --n 10
python archive/tools/error_analysis/sample_errors_for_annotation.py \
  --out-csv error_analysis/annotations/error_review_50.csv --n 50 --annotate-backend none
```

> Ambiguity scripts live under `archive/tools/error_analysis/` and are not part of the core paper reproducibility path.
