# Data directory

| Path | Purpose |
|------|--------|
| `section48/` | Section 4.8 segment JSONLs (`segments_ner*.jsonl`), `planning_locks.json` (generated), small JSON reports. **Committed** (small). Retired extractors: [archive/data/section48/](../archive/data/section48/README.md). |
| `meddra/` | MedDRA release files for graph build (`scripts/extract_meddra.py` / `build_graph.py`). **Not committed** (large; requires a MedDRA **licence**) — see root `README.md`. |
| `QUAERO_FrenchMed/` | Optional corpus for NER fine-tuning. **Not committed** by default. |
| `error_analysis/` | Optional error-analysis inputs. |

Regenerate `data/section48/segments_ner.jsonl` with `scripts/prepare_data.py` (default FR/EN PDF paths under `test_data/` when present).

**HTM:** supply your own reviewed FR→EN JSON (fields such as `fr`, `en_label`, `en_aliases`, `level`, `tier`) and pass **`--gold-terms`** to `scripts/evaluate.py` / `scripts/plot_results.py`. Optional draft rows: `scripts/build_gold_terms_from_parallel_ner.py --out …`.
