# Results interpretation (authoritative numbers)

**Language:** English · [Português](pt/RESULTS_INTERPRETATION.md) · [Deutsch](de/RESULTS_INTERPRETATION.md)

Use this file when writing prose for papers or reports: **numbers must match** the committed [`scores_summary.csv`](../results/ner_biollm/figures/scores_summary.csv) files produced by `evaluate.py`, not informal overview text written earlier.

## Where the authoritative scores live

| Condition | Path |
|-----------|------|
| **ner_biollm** | [`results/ner_biollm/figures/scores_summary.csv`](../results/ner_biollm/figures/scores_summary.csv) |
| **ner_biollm_finetuned** | [`results/ner_biollm_finetuned/figures/scores_summary.csv`](../results/ner_biollm_finetuned/figures/scores_summary.csv) |

Human-readable tables (same source): [`paper_summary_table.md`](../results/ner_biollm/figures/paper_summary_table.md) under each condition’s `figures/` folder.

Re-evaluate after changing segments, Neo4j graph, or metric code:

```bash
docker compose up -d   # Neo4j for CCR / HTM / graph metrics
PYTHONPATH=. python evaluate.py …   # see README.md
```

## Snapshot aligned with **committed** CSVs (do not round differently elsewhere)

LEXICAL **`htm`** column in `scores_summary.csv` (NER-anchored grounding; thin coverage drives low absolute values).

### ner_biollm (`results/ner_biollm/figures/scores_summary.csv`)

| Stage | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.58 | **0.196** | 0.130 | 0.354 |
| S3 | 32.65 | **0.253** | 0.130 | 0.354 |

So **S3 HTM > S2 HTM** under this snapshot (not “S2 dominating” on HTM). chrF++ still favors **S2** over S3–S5.

### ner_biollm_finetuned (`results/ner_biollm_finetuned/figures/scores_summary.csv`)

| Stage | chrF++ | HTM (lex) | `htm_en_ref_dataset` | `ccr_dataset` |
|-------|--------|-----------|------------------------|---------------|
| S2 | 35.85 | **0.250** | 0.158 | 0.324 |
| S3 | 34.60 | **0.257** | 0.158 | 0.324 |

Again **S3 slightly edges S2 on HTM**; chrF++ peak remains **S2**.

## Overview vs CSV discrepancy

If any overview or slide deck cites HTM around **0.45 / 0.43** for S2/S3, that **does not match** the committed tables above (≈0.20–0.26 lexical HTM). Treat those headline figures as either **an older evaluation**, a **different metric variant**, or an **error** until you reproduce them with `evaluate.py` and archive the run configuration.

## Why HTM looks low (context)

- Current lexical HTM uses **NER-anchored** grounding (`terms[].word` → Neo4j → compare hypothesis). Sparse grounding ⇒ thin HTM signal.
- **`htm_en_ref_dataset`** in the CSVs (~0.13–0.16 committed here) is an upper-bound style signal on how often reference English aligns with MedDRA-style renderings under the same machinery—not a replacement for clinical gold lists.

## Prioritized follow-ups (working checklist)

1. **Re-run evaluation** with Neo4j up and document git revision + segment paths; refresh `scores_summary.csv` if anything changed.
2. **Gold terms list** — building `gold_terms.json` (e.g. via `build_gold_terms_from_parallel_ner.py`) so HTM is measured against intended concepts, not only NER coverage.
3. **Qualitative sheet** — fill real labels in [`error_analysis/annotations/error_review_50.csv`](../error_analysis/annotations/error_review_50.csv); prioritize high-drift rows from [`error_analysis/ner_biollm_term_drift.csv`](../error_analysis/ner_biollm_term_drift.csv).
4. **Ambiguous grounding** — resolve concrete cases (e.g. `pneumopathie inflammatoire`) using MedDRA context + optional gold-term locks.
5. **Cross-NER dashboard** — run `plot_cross_ner_dashboard.py` (or the eval phase in [`rerun_all.sh`](../rerun_all.sh)) with Neo4j up; output dir is typically `results/cross_ner_comparison/` (created on demand, not always committed).
6. **Ontology LoRA track** — appendix / future work unless cluster time allows.

---

Update this document whenever you commit new `scores_summary.csv` files so prose and tables stay aligned. **Localized copies:** update [`pt/RESULTS_INTERPRETATION.md`](pt/RESULTS_INTERPRETATION.md) and [`de/RESULTS_INTERPRETATION.md`](de/RESULTS_INTERPRETATION.md) in the same pass when numbers or checklist items change.
