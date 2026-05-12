<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.14+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers"></a>
  <a href="https://www.sbert.net/"><img src="https://img.shields.io/badge/sentence--transformers-embeddings-7C3AED?style=for-the-badge" alt="sentence-transformers"></a>
  <a href="https://neo4j.com/"><img src="https://img.shields.io/badge/Neo4j-graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" alt="Neo4j"></a>
</p>
<p align="center">
  <a href="https://docs.docker.com/compose/"><img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Compose"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-arrays-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-plots-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
  <a href="https://github.com/mjpost/sacrebleu"><img src="https://img.shields.io/badge/sacreBLEU-metric-222222?style=for-the-badge" alt="sacreBLEU"></a>
  <a href="https://pytest.org/"><img src="https://img.shields.io/badge/pytest-tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest"></a>
</p>

# TermPlan-MT

**Terminology-aware machine translation** — French → English for SmPC Section 4.8, with MedDRA-grounded terminology across six MT systems (S1–S6).

---

## Where is what

| Location | Purpose |
|----------|---------|
| [`pipeline.py`](pipeline.py) | MedDRA graph grounding, terminology planning, MedDRA flat-file I/O. |
| [`systems.py`](systems.py) | Translation systems S1–S6, segment loading, model management. |
| [`metrics.py`](metrics.py) | HTM, CCR, BLEU, chrF, COMET, eval helpers, figures infrastructure. |
| [`tools/pipeline/`](tools/pipeline/) | CLI runner — [`run_pipeline.py`](tools/pipeline/run_pipeline.py). |
| [`tools/eval/`](tools/eval/) | Scoring and plots — [`evaluate.py`](tools/eval/evaluate.py), [`plot_figures.py`](tools/eval/plot_figures.py), [`run_eval_plot_matrix.py`](tools/eval/run_eval_plot_matrix.py), [`bootstrap_bleu_delta.py`](tools/eval/bootstrap_bleu_delta.py). |
| [`tools/data/`](tools/data/) | MedDRA extract/load scripts (`extract_meddra.py`, `build_graph.py`). |
| [`tools/error_analysis/`](tools/error_analysis/) | Qualitative analysis scripts (outputs under [`error_analysis/`](error_analysis/)). |
| [`data/section48/`](data/section48/) | Segment JSONLs (`segments_ner_biollm.jsonl`, `segments_ner_unsloth.jsonl`). |
| [`results/`](results/) | Run outputs — `s*.jsonl` are gitignored; figures and CSV summaries are committed. |
| [`docs/`](docs/) | [`RESULTS_INTERPRETATION.md`](docs/RESULTS_INTERPRETATION.md), [`CANONICAL_METRICS.md`](docs/CANONICAL_METRICS.md), error-review schema. |
| [`tests/`](tests/) | [`pytest`](https://pytest.org/) unit tests. |
| [`rerun_all.sh`](rerun_all.sh) | Full reproducibility driver (NER conditions + eval matrix). |
| [`docker-compose.yml`](docker-compose.yml) | Neo4j for graph grounding and HTM/CCR metrics. |

---

## Reproducing the paper results

| Step | What to do |
|------|------------|
| 1 | `cd` into the repo root. |
| 2 | `python -m venv .venv` → `.venv/bin/pip install -r requirements.txt` |
| 3 | `docker compose up -d` — starts Neo4j ([`docker-compose.yml`](docker-compose.yml)). |
| 4 | Load MedDRA: obtain a licence, extract with [`tools/data/extract_meddra.py`](tools/data/extract_meddra.py), then load with `PYTHONPATH=. python tools/data/build_graph.py` (see [`data/README.md`](data/README.md)). |
| 5 | Segment JSONLs are already in [`data/section48/`](data/section48/) — no NER re-run needed. |
| 6 | **Run the pipeline:** `PYTHONPATH=. python tools/pipeline/run_pipeline.py --segments data/section48/segments_ner_biollm.jsonl --results-dir results/ner_biollm` |
| 7 | **Score and plot:** `PYTHONPATH=. python tools/eval/evaluate.py --results-dir results/ner_biollm --segments data/section48/segments_ner_biollm.jsonl` then `PYTHONPATH=. python tools/eval/plot_figures.py …` |
| 8 | **Full matrix:** `./rerun_all.sh` runs both NER conditions and all eval/plot steps. |

**S1 / S2 reuse:** `REUSE_S1_S2_FROM_BIOLLM=1` (default in `rerun_all.sh`) copies `results/ner_biollm/s1.jsonl` and `s2.jsonl` into the finetuned-NER tree and only reruns S3–S6. Set to `0` for a full S1–S6 run per condition.

---

## Read more

| File | Topic |
|------|--------|
| [`tools/README.md`](tools/README.md) | CLI reference for pipeline, eval, and data tools. |
| [`data/README.md`](data/README.md) | Data tree and MedDRA setup. |
| [`docs/RESULTS_INTERPRETATION.md`](docs/RESULTS_INTERPRETATION.md) | Authoritative metric snapshot, discrepancy notes, paper checklist. |
| [`docs/CANONICAL_METRICS.md`](docs/CANONICAL_METRICS.md) | Definition of each metric and how contamination is handled. |

---

## Error analysis

Scripts under [`tools/error_analysis/`](tools/error_analysis/) produce CSVs from the pipeline JSONLs. The annotation schema is in [`docs/error_analysis/schema.md`](docs/error_analysis/schema.md). Outputs are stored under [`error_analysis/`](error_analysis/).
