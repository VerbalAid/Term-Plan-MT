# Error analysis (generated artefacts)

Files here are **regenerated** after pipeline or metric changes; they are not the source of truth.

Ephemeral run logs under `error_analysis/*.log` are gitignored (use your own filename if you redirect `rerun_all.sh` output here).

## Inventory (regenerate after pipeline / Neo4j / segment changes)

| File | Description |
|------|-------------|
| [`ambiguous_spans_top10.md`](ambiguous_spans_top10.md) / [`.csv`](ambiguous_spans_top10.csv) | Up to 10 `(segment_id, term)` examples where string grounding hits **multiple** MedDRA concepts. |
| [`ambiguous_grounding_report.csv`](ambiguous_grounding_report.csv) | Per ambiguous **normalized FR key**: NER counts for two segment JSONLs + planning-lock overlap (see script defaults). |
| [`error_review_50.csv`](error_review_50.csv) | 50 worst chrF segment rows (heuristic pre-labels); use `--annotate-backend ollama` for LLM fill. |
| `ner_biollm_*.csv` | Terminology reports for **`results/ner_biollm`** (copies of `figures/term_drift`, `level_distribution`, `missing_terms`, `consistency`). |
| `ner_biollm_finetuned_*.csv` | Same for **`results/ner_biollm_finetuned`** + `segments_ner_unsloth_full.jsonl`. |

**One-shot refresh (this folder + copies of terminology CSVs):**

```bash
export PYTHONPATH=.
python tools/error_analysis/list_ambiguous_spans.py --n 10
python tools/error_analysis/report_ambiguous_grounding.py \
  --segments-b data/section48/segments_ner_unsloth_full.jsonl --label-b ner_unsloth_full
python tools/error_analysis/sample_errors_for_annotation.py \
  --out-csv error_analysis/error_review_50.csv --n 50 --annotate-backend none \
  --ner-glob ner_biollm --ner-glob ner_biollm_finetuned --seed 42
python tools/error_analysis/analyse_terminology.py --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl
python tools/error_analysis/analyse_terminology.py --results-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl
for f in term_drift level_distribution missing_terms consistency; do
  cp -f results/ner_biollm/figures/${f}.csv error_analysis/ner_biollm_${f}.csv
  cp -f results/ner_biollm_finetuned/figures/${f}.csv error_analysis/ner_biollm_finetuned_${f}.csv
done
```

Canonical `analyse_terminology` outputs also remain under each `results/<condition>/figures/` (duplicated here for a single handoff folder).

## Quick refresh (Neo4j + `PYTHONPATH=.`)

```bash
cd "/path/to/MT_Project_Terminology "   # quote if directory name ends with a space
export PYTHONPATH=.

# Rescore one condition (skip COMET for speed)
python tools/eval/evaluate.py --no-comet \
  --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl

python tools/eval/evaluate.py --no-comet \
  --results-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl

# Regenerate figures + scores_summary.csv under each results/*/figures/
python tools/eval/plot_figures.py --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl
python tools/eval/plot_figures.py --results-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl
```

The **`htm_hyp_ref_agreement`** column (console **`HypRefAg`**) compares HTM-style scores on **`hyp`** vs.\ **`en_ref`** per grounded French NER span; see stderr legend under the `evaluate.py` table.

## Ambiguous MedDRA grounding (10 examples)

When a normalised French NER string matches **more than one** MedDRA concept in the string cache, grounding is ambiguous. Lists for manual ontology lookup:

- [`ambiguous_spans_top10.md`](ambiguous_spans_top10.md) / [`ambiguous_spans_top10.csv`](ambiguous_spans_top10.csv) — from [`tools/error_analysis/list_ambiguous_spans.py`](../tools/error_analysis/list_ambiguous_spans.py) (default: `segments_ner_biollm.jsonl`).
- Broader key-level report: [`tools/error_analysis/report_ambiguous_grounding.py`](../tools/error_analysis/report_ambiguous_grounding.py) → [`ambiguous_grounding_report.csv`](ambiguous_grounding_report.csv); see [`docs/error_analysis/ambiguous_grounding.md`](../docs/error_analysis/ambiguous_grounding.md).

## LLM-assisted error sampling (optional)

```bash
PYTHONPATH=. python tools/error_analysis/sample_errors_for_annotation.py \
  --out-csv error_analysis/error_review_50.csv --n 50 --annotate-backend ollama
```

Mistral-based **annotation sheet** (GPU): `tools/error_analysis/sample_errors_for_annotation.py --annotation-sheet --results-dir results/ner_biollm …` (see script docstring).

## ACL-style write-up of results

[`docs/termplan_acl_snapshot.tex`](../docs/termplan_acl_snapshot.tex) — compact tables and metric definitions (`pdflatex docs/termplan_acl_snapshot.tex`).

## Older material

Legacy rows: [`legacy/`](legacy/). Schema: [`docs/error_analysis/schema.md`](../docs/error_analysis/schema.md). Snapshot tarball: [`tools/admin/archive_results_snapshot.sh`](../tools/admin/archive_results_snapshot.sh).
