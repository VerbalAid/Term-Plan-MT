# Data directory

| Path | Purpose |
|------|--------|
| `section48/` | Section 4.8 segment JSONLs (`segments_ner*.jsonl`), `planning_locks.json` (generated), small JSON reports. **Committed** (small). Retired extractors: [archive/data/section48/](../archive/data/section48/README.md). |
| `meddra/` | MedDRA release files for graph build (`tools/data/extract_meddra.py` / `tools/data/build_graph.py`). **Not committed** (large; requires a MedDRA **licence**) — see root `README.md`. |
| `QUAERO_FrenchMed/` | Optional corpus for NER fine-tuning. **Not committed** by default. |
| `error_analysis/` | Optional error-analysis inputs. |

Regenerate `data/section48/segments_ner.jsonl` with `tools/data/prepare_data.py` (default FR/EN PDF paths under `data/test_data/` when present; see [`test_data/README.md`](test_data/README.md)).

**HTM:** uses French **`terms[].word`** on the same segment JSONL you pass to **`--segments`** in `tools/eval/evaluate.py` / `tools/eval/plot_figures.py` (with Neo4j). Optional **`data/gold_terms.json`** is only for **Neo4j graph seeding** via `tools/data/build_graph.py`, not for HTM. To draft such rows from parallel NER, see `tools/data/build_gold_terms_from_parallel_ner.py --out …`.
