# MedDRA Lookup Tool

Standalone web app for exploring the MedDRA v28 Neo4j graph (separate from the TermPlanMT MT pipeline).

## Features

- Bidirectional search: French or English → concept with both labels
- Cascade: exact → RapidFuzz fuzzy → sentence-transformer semantic (lazy-loaded)
- Hierarchy: parents, children, ancestor chain SOC → match
- **In context:** sentence + term → graph candidates → OpenRouter disambiguation and register notes

## Requirements

- Python 3.12+
- Neo4j with MedDRA graph loaded
- ~500 MB RAM without semantic; ≥ 2 GB with semantic search
- OpenRouter API key for in-context routing (optional)

## Local run

From the repository root:

```bash
pip install -r webapp/requirements.txt
cp webapp/.env.example webapp/.env   # Neo4j + LLM_API_KEY
PYTHONPATH=. uvicorn webapp.main:app --reload --port 8000
```

Open http://localhost:8000

## API

| Method | Path | Body |
|--------|------|------|
| `GET` | `/api/health` | — |
| `POST` | `/api/lookup` | `{ "term", "lang": "auto" \| "fr" \| "en" }` |
| `POST` | `/api/context-lookup` | `{ "context_sentence", "target_term", "lang" }` |

## Deploy on Render

1. Connect repo (`render.yaml`, `rootDir: webapp`).
2. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, `OPENROUTER_API_KEY`.
3. Keep `PREWARM_SEMANTIC=false` unless you need instant semantic at boot.

First semantic query may download ~471 MB embeddings once (`~/.cache/huggingface`).

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI |
| `NEO4J_USER` / `NEO4J_PASS` | — | Auth |
| `LLM_API_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter gateway |
| `LLM_API_KEY` | — | OpenRouter secret (`sk-or-…`) |
| `LLM_MODEL_NAME` | `mistralai/mistral-7b-instruct` | Rolling production model id |
| `GROUND_FUZZY_CUTOFF` | `90` | RapidFuzz minimum |
| `LOOKUP_SEMANTIC_MIN` | `0.55` | Cosine floor |
| `PREWARM_SEMANTIC` | `false` | Load embeddings at startup |

## Layout

```
webapp/
  main.py
  lookup.py
  context_llm.py   # OpenRouter client
  graph.py
  static/
```
