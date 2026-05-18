# TermPlan-MT

Terminology-aware French→English machine translation for SmPC Section 4.8 adverse-event text, with MedDRA-grounded planning (systems S1–S6).

**Repository:** [github.com/VerbalAid/Term-Plan-MT](https://github.com/VerbalAid/Term-Plan-MT)

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose up -d
```

Load MedDRA into Neo4j (licence required):

```bash
PYTHONPATH=. python data/extract_meddra.py --meddra-dir data/meddra
PYTHONPATH=. python data/build_graph.py
```

Run the pipeline on bundled segments:

```bash
PYTHONPATH=. python run_pipeline.py \
  --segments data/section48/segments_ner_biollm.jsonl \
  --results-dir results/ner_biollm
```

Full reproducibility: `./rerun_all.sh`

## Layout

| Path | Role |
|------|------|
| `pipeline.py` | MedDRA grounding and terminology planning |
| `systems.py` | Translation systems S1–S6 |
| `metrics.py` | HTM, CCR, BLEU, chrF, evaluation |
| `data/section48/` | Segment JSONLs |
| `results/` | Evaluation outputs (JSONL gitignored) |
| `webapp/` | Standalone MedDRA lookup UI |
| `docs/paper/termplanmt_v3.pdf` | Paper |

## MedDRA lookup (webapp)

```bash
pip install -r webapp/requirements.txt
cp webapp/.env.example webapp/.env   # Neo4j + LLM_API_KEY
PYTHONPATH=. uvicorn webapp.main:app --reload --port 8000
```

**Deploy on Render:** Docker runtime — see [`docs/DEPLOY_RENDER.md`](docs/DEPLOY_RENDER.md). Set Aura `NEO4J_*`, `WEBAPP_PASSWORD` (e.g. `term`), optional `LLM_API_KEY`.

## Docs

- [`docs/NER_FINETUNING.md`](docs/NER_FINETUNING.md) — NER training and metrics
- [`data/README.md`](data/README.md) — Data layout
- [`models/README.md`](models/README.md) — Local checkpoints
