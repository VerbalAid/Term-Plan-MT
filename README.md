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

**Terminology-aware machine translation** — French → English for SmPC Section 4.8, with MedDRA-grounded terminology across several MT setups.

---

## Where is what

| Location | Purpose |
|----------|---------|
| [`pipeline/`](pipeline/) | Core code: graph grounding, planning, translation systems (S1–S5), metrics helpers. |
| [`tools/pipeline/`](tools/pipeline/) | CLI to run the ladder — mainly [`run_pipeline.py`](tools/pipeline/run_pipeline.py). |
| [`tools/eval/`](tools/eval/) | Scores, plots, batch eval — e.g. [`evaluate.py`](tools/eval/evaluate.py), [`plot_figures.py`](tools/eval/plot_figures.py), [`run_eval_plot_matrix.py`](tools/eval/run_eval_plot_matrix.py). |
| [`tools/data/`](tools/data/) | MedDRA extract/load, ontology JSONL export/split/patch. |
| [`tools/error_analysis/`](tools/error_analysis/) | Scripts that produce qualitative CSVs / reports (outputs usually under [`error_analysis/`](error_analysis/)). |
| [`data/`](data/) | Corpora, segment JSONLs — see [`data/README.md`](data/README.md). Segment files live under [`data/section48/`](data/section48/). |
| [`training_scripts/`](training_scripts/) | NER inference + Unsloth fine-tuning — see [`training_scripts/README.md`](training_scripts/README.md). |
| [`results/`](results/) | Run outputs (large `*.jsonl` gitignored; figures / summaries may be committed per profile). |
| [`error_analysis/`](error_analysis/) | Stored qualitative analysis artifacts (CSVs, notes). |
| [`docs/`](docs/) | Extra docs and figures — e.g. ontology tuning [`mistral_instruct_Ontology-Fine-tuning.md`](docs/mistral_instruct_Ontology-Fine-tuning.md), error-review schema [`docs/error_analysis/schema.md`](docs/error_analysis/schema.md). |
| [`tests/`](tests/) | [`pytest`](https://pytest.org/) tests. |
| [`rerun_all.sh`](rerun_all.sh) | Full reproducibility driver (NER conditions + eval matrix); see header for `SKIP_*`. |
| [`docker-compose.yml`](docker-compose.yml) | Neo4j for grounding / metrics. |

---

## Quick start

| Step | What to do |
|------|------------|
| 1 | `cd` into the repo. If the folder name ends with a space, quote the path, e.g. `cd "/…/MT_Project_Terminology "`. |
| 2 | `python -m venv .venv` → `.venv/bin/pip install -r` [`requirements.txt`](requirements.txt) |
| 3 | `docker compose up -d` — Neo4j for grounding / metrics ([`docker-compose.yml`](docker-compose.yml)). |
| 4 | Segment JSONL under [`data/section48/`](data/section48/) (`segments_ner*.jsonl`). To (re)generate NER, use [`training_scripts/ner/biomistral_prompt_ner.py`](training_scripts/ner/biomistral_prompt_ner.py) (see [`training_scripts/README.md`](training_scripts/README.md)). |
| 5 | **Full matrix:** [`./rerun_all.sh`](rerun_all.sh) (see script header for `SKIP_*`). **Ad hoc:** `PYTHONPATH=. python` [`tools/pipeline/run_pipeline.py`](tools/pipeline/run_pipeline.py) `--segments … --results-dir …` |
| 6 | **Scores / figures:** [`tools/eval/evaluate.py`](tools/eval/evaluate.py), [`tools/eval/plot_figures.py`](tools/eval/plot_figures.py), or the eval phase inside [`rerun_all.sh`](rerun_all.sh) / [`tools/eval/run_eval_plot_matrix.py`](tools/eval/run_eval_plot_matrix.py). |

**S1 / S2 reuse:** In [`rerun_all.sh`](rerun_all.sh), `REUSE_S1_S2_FROM_BIOLLM=1` (default) copies [`results/ner_biollm/s1.jsonl`](results/ner_biollm/) and `s2.jsonl` into other result trees and runs **S3–S5** only. Set to `0` for a full S1–S5 rerun per condition.

**Results folders:** [`results/ner_biollm/`](results/ner_biollm/) — prompted BioMistral NER on `segments_ner_biollm.jsonl`. [`results/ner_biollm_finetuned/`](results/ner_biollm_finetuned/) — Unsloth NER segments (see [`rerun_all.sh`](rerun_all.sh)). Git keeps figures (PNG, CSV, markdown); `s*.jsonl` are local / gitignored — regenerate with the pipeline + eval.

---

## Read more

| File | Topic |
|------|--------|
| [`tools/README.md`](tools/README.md) | CLI layout (`run_pipeline`, `eval`, `data` tools) |
| [`data/README.md`](data/README.md) | Data tree, MedDRA, ontology JSONL |
| [`training_scripts/README.md`](training_scripts/README.md) | NER inference + Unsloth fine-tuning (QUAERO / ontology SFT) |
| [`docs/mistral_instruct_Ontology-Fine-tuning.md`](docs/mistral_instruct_Ontology-Fine-tuning.md) | Mistral-7B-Instruct ontology fine-tuning |

**MedDRA** is not redistributed; obtain a licence, extract with [`tools/data/extract_meddra.py`](tools/data/extract_meddra.py), load with [`tools/data/build_graph.py`](tools/data/build_graph.py) (details in [`data/README.md`](data/README.md)).

---

## Error analysis

Run audits and build annotation sheets from [`results/`](results/) using the scripts under [`tools/error_analysis/`](tools/error_analysis/). The expected columns for manual review live in [`docs/error_analysis/schema.md`](docs/error_analysis/schema.md). Outputs for qualitative review are typically saved under [`error_analysis/`](error_analysis/).
