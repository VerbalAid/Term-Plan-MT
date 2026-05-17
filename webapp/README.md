# MedDRA Lookup Web App

Interactive lookup for the TermPlanMT Neo4j graph: **exact string match → fuzzy (RapidFuzz) → semantic (sentence-transformers)**.

## Local run

1. Start Neo4j with MedDRA loaded (`docker compose up -d`, then `PYTHONPATH=. python data/build_graph.py`).
2. Copy `.env.example` to `.env` and set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`.
3. From the **repository root**:

```bash
pip install -r webapp/requirements.txt
PYTHONPATH=. uvicorn webapp.main:app --reload --port 8000
```

Open http://localhost:8000

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Neo4j connectivity |
| `POST` | `/api/lookup` | JSON body `{ "term": "…", "lang": "fr" }` |
| `GET` | `/api/lookup?term=…&lang=fr` | Same as POST |

Response includes `match_type` (`exact` \| `fuzzy` \| `semantic` \| `none`), `concept`, `parents`, `children`, and `ancestors` (SOC → … → match).

## Deploy on Render

1. Push this repo to GitHub.
2. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint** → connect repo (`render.yaml` at root).
3. Set secrets: `NEO4J_URI` (e.g. Neo4j Aura `neo4j+s://…`), `NEO4J_USER`, `NEO4J_PASS`.
4. Use a Render plan with **≥ 2 GB RAM** if you enable `PREWARM_SEMANTIC=true` (loads embedding model at startup).

First semantic query may take 30–60s while the embedding index builds.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI |
| `NEO4J_USER` / `NEO4J_PASS` | `neo4j` / `password` | Auth |
| `GROUND_FUZZY_CUTOFF` | `90` | RapidFuzz minimum (0–100) |
| `LOOKUP_SEMANTIC_MIN` | `0.55` | Cosine similarity floor |
| `TERMPLAN_EMBED_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Semantic model |
| `PREWARM_SEMANTIC` | `false` | Build embedding index at startup |
