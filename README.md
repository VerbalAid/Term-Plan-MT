<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.14+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers"></a>
  <a href="https://www.sbert.net/"><img src="https://img.shields.io/badge/sentence--transformers-embeddings-7C3AED?style=for-the-badge" alt="sentence-transformers"></a>
  <a href="https://neo4j.com/"><img src="https://img.shields.io/badge/Neo4j-graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" alt="Neo4j"></a>
  <a href="https://github.com/mjpost/sacrebleu"><img src="https://img.shields.io/badge/sacreBLEU-metric-222222?style=for-the-badge" alt="sacreBLEU"></a>
  <a href="https://pytest.org/"><img src="https://img.shields.io/badge/pytest-tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest"></a>
</p>

# TermPlan-MT

**Terminology-aware machine translation** — French → English for pharmaceutical adverse-event text (SmPC Section 4.8), with MedDRA-grounded terminology planning across six MT systems (S1–S6).

**Repository:** [github.com/VerbalAid/Term-Plan-MT](https://github.com/VerbalAid/Term-Plan-MT)

| NER condition | Training / inference | Eval segments (126 scored) |
|---------------|----------------------|----------------------------|
| Baseline | BioMistral-7B zero-shot JSON prompt | `segments_ner_biollm.jsonl` → `results/ner_biollm/` |
| Fine-tuned | Unsloth QLoRA on **~1,540** QUAERO BRAT sentences (EMEA + MEDLINE) | `segments_ner_unsloth_full.jsonl` → `results/ner_biollm_finetuned/` |

Full NER training counts, hyperparameters, and poster-safe claims: **[`docs/NER_FINETUNING.md`](docs/NER_FINETUNING.md)**.

---

## Structure

```
pipeline.py          # MedDRA graph grounding and terminology planning
systems.py           # Translation systems S1–S6
metrics.py           # HTM, CCR, BLEU, chrF, COMET, evaluation helpers

run_pipeline.py      # Run a system:   python run_pipeline.py --system s3
evaluate.py          # Score results:  python evaluate.py --results-dir results/ner_biollm
plot_figures.py      # Figures + scores_summary.csv for one results/ tree
plot_cross_ner_dashboard.py  # Cross-condition comparison plots
bootstrap_bleu_delta.py      # Paired bootstrap CIs for ΔBLEU (paper Table 5)
run_eval_plot_matrix.py      # Batch evaluate + plot over all profiles (used by rerun_all.sh)
rerun_all.sh         # Full reproducibility driver

data/
  section48/         # Segment JSONLs (inputs to the pipeline)
  build_graph.py     # Load MedDRA into Neo4j
  extract_meddra.py  # Extract MedDRA ASCII from zip

results/
  ner_biollm/        # BioMistral-7B prompted NER condition
  ner_biollm_finetuned/  # Fine-tuned BioMistral NER condition

tests/               # pytest unit tests
docker-compose.yml   # Neo4j
requirements.txt
```

---

## Reproducing the paper

**1. Environment**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
docker compose up -d          # starts Neo4j
```

**2. MedDRA graph** (obtain a licence from the MSSO first)

```bash
PYTHONPATH=. python data/extract_meddra.py --meddra-dir data/meddra
PYTHONPATH=. python data/build_graph.py
```

**3. Run the pipeline** (segment JSONLs already in `data/section48/`)

```bash
# Ad hoc — one condition:
PYTHONPATH=. python run_pipeline.py \
  --segments data/section48/segments_ner_biollm.jsonl \
  --results-dir results/ner_biollm

# Full matrix (both NER conditions, S1–S6, eval, figures):
./rerun_all.sh
```

`rerun_all.sh` skips phases with env flags — see the header comments. Key flag: `REUSE_S1_S2_FROM_BIOLLM=1` (default) copies S1/S2 from `ner_biollm` into the finetuned tree and runs S3–S6 only.

**4. Evaluate and plot**

```bash
PYTHONPATH=. python evaluate.py \
  --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl

PYTHONPATH=. python plot_figures.py \
  --results-dir results/ner_biollm \
  --segments data/section48/segments_ner_biollm.jsonl
```

**5. Bootstrap ΔBLEU** (paper Table 5)

```bash
PYTHONPATH=. python bootstrap_bleu_delta.py \
  --baseline-dir results/ner_biollm \
  --finetuned-dir results/ner_biollm_finetuned \
  --segments data/section48/segments_ner_unsloth_full.jsonl \
  --exclude-segment-ids "" \
  --baseline-eval-file-set standard \
  --finetuned-eval-file-set mistral_clean \
  --out-csv results/ner_biollm_finetuned/figures/bleu_delta_bootstrap_95ci.csv
```

---

## Core modules

| File | Contents |
|------|----------|
| `pipeline.py` | `TermGraph` (Neo4j grounding), `load_or_compute_locks` (planning), MedDRA flat-file I/O. |
| `systems.py` | S1 NLLB baseline · S2 Mistral doc-context · S3 GraphRAG · S4 rerank · S5 logit-boost · S6 glossary oracle. |
| `metrics.py` | HTM (hierarchy-aware terminology match), CCR, BLEU, chrF, COMET, eval helpers, figures infrastructure. |

---

## Results directories

| Directory | NER condition | Segment file |
|-----------|--------------|-------------|
| `results/ner_biollm/` | BioMistral-7B JSON-list prompting | `segments_ner_biollm.jsonl` |
| `results/ner_biollm_finetuned/` | Fine-tuned BioMistral NER | `segments_ner_unsloth_full.jsonl` |

`s*.jsonl` outputs are gitignored (large); figures and `scores_summary.csv` are committed. Regenerate with `rerun_all.sh`.

---

## MedDRA Lookup Tool (standalone)

The **`webapp/`** directory is a **separate deployable product** — graph term search only, not part of the MT evaluation pipeline.

```bash
pip install -r webapp/requirements.txt
PYTHONPATH=. uvicorn webapp.main:app --reload --port 8000
```

Deploy on [Render](https://render.com) via [`render.yaml`](render.yaml) (`rootDir: webapp`). Full docs: [`webapp/README.md`](webapp/README.md).

---

## Further reading

| File | Topic |
|------|-------|
| [`webapp/README.md`](webapp/README.md) | Lookup app: local run, API, Render deploy. |
| [`docs/NER_FINETUNING.md`](docs/NER_FINETUNING.md) | QUAERO training size (**1,540 sentences**), LoRA settings, CCR/BLEU effects, S5/S6 boost surface rates. |
| [`docs/RESULTS_INTERPRETATION.md`](docs/RESULTS_INTERPRETATION.md) | Authoritative metric snapshot, paper table checklist, known discrepancies. |
| [`docs/CANONICAL_METRICS.md`](docs/CANONICAL_METRICS.md) | Metric definitions and contamination-handling rules. |
| [`data/README.md`](data/README.md) | Segment JSONL format and MedDRA setup. |
| [`models/README.md`](models/README.md) | Local checkpoint paths (not in git). |
| [`docs/paper/termplanmt_v3.pdf`](docs/paper/termplanmt_v3.pdf) | Paper PDF. |
