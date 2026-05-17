# MedDRA Lookup Tool

**Standalone web app** for exploring the MedDRA v28 Neo4j graph. This is separate from the TermPlanMT machine-translation pipeline (`run_pipeline.py`, `systems.py`, evaluation, etc.).

## Features

- **Bidirectional search:** French or English query → concept with both labels
- **Cascade:** exact string match → RapidFuzz fuzzy → sentence-transformer semantic (lazy-loaded)
- **Hierarchy:** parents (broader), children (narrower), ancestor chain SOC → match
- **UI:** click any parent, child, or ancestor to run a new lookup

## Requirements

- Python 3.12+
- Neo4j with MedDRA graph loaded (see repo `data/build_graph.py` if you use the full monorepo)
- ~500 MB RAM without semantic; **≥ 2 GB** recommended when semantic search is used

## Local run

From the **repository root**:

```bash
pip install -r webapp/requirements.txt
cp webapp/.env.example .env   # set NEO4J_URI, NEO4J_USER, NEO4J_PASS
PYTHONPATH=. uvicorn webapp.main:app --reload --port 8000
```

Open http://localhost:8000

## API

| Method | Path | Body / query |
|--------|------|----------------|
| `GET` | `/api/health` | Neo4j + semantic index status |
| `POST` | `/api/lookup` | `{ "term": "…", "lang": "auto" \| "fr" \| "en" }` |
| `GET` | `/api/lookup?term=…&lang=auto` | Same |

Response fields: `match_type`, `query_lang`, `concept`, `parents`, `children`, `ancestors`, `alternatives`, `semantic_ready`.

## Deploy on Render

1. Push to GitHub.
2. Render → **New Blueprint** → connect repo (`render.yaml` uses `rootDir: webapp`).
3. Set secrets: `NEO4J_URI` (e.g. Aura `neo4j+s://…`), `NEO4J_USER`, `NEO4J_PASS`.
4. Keep `PREWARM_SEMANTIC=false` on Starter/Standard unless you need instant semantic (uses more RAM at boot).

First **semantic** query may take 30–60s while the embedding model (~471 MB) downloads once into `~/.cache/huggingface`; later semantic queries are fast.

### Hierarchy shows “None at this level”

The graph uses `(broader)-[:BROADER_THAN]->(narrower)` from `data/build_graph.py`. The app re-anchors matches to **numeric MedDRA codes** and skips orphan seed nodes without edges. If hierarchy is still empty, open `/api/debug/neighborhood?term=toux&lang=fr` or check `/api/graph/schema`.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI |
| `NEO4J_USER` / `NEO4J_PASS` | — | Auth |
| `GROUND_FUZZY_CUTOFF` | `90` | RapidFuzz minimum |
| `LOOKUP_SEMANTIC_MIN` | `0.55` | Cosine similarity floor |
| `TERMPLAN_EMBED_MODEL` | `…MiniLM-L12-v2` | Embedding model |
| `PREWARM_SEMANTIC` | `false` | Load embeddings at startup |

## Layout

```
webapp/
  main.py       # FastAPI entry
  lookup.py     # Search cascade
  graph.py      # Neo4j (no MT pipeline import)
  static/       # UI
  requirements.txt
```
