# Render deployment (term-plan-mt)

## One-time service setup

1. [Render Dashboard](https://dashboard.render.com) → **term-plan-mt** (or create Web Service from this repo).
2. **Settings → General**
   - **Runtime:** `Docker` (not Python)
   - **Dockerfile path:** `./Dockerfile`
   - **Root directory:** leave empty
3. **Clear** the **Build command** and **Start command** fields (Docker uses the Dockerfile `CMD`).
4. **Health check path:** `/api/health`

Redeploy: **Manual Deploy → Clear build cache & deploy**.

Docker fixes `uvicorn: command not found` from an old Start command stuck in the dashboard.

## Environment variables (required)

Set in **Environment** (not in git):

| Key | Value |
|-----|--------|
| `NEO4J_URI` | Aura URI, e.g. `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASS` | Aura password (from download) |
| `WEBAPP_PASSWORD` | e.g. `term` (login form before UI) |
| `LLM_API_KEY` | OpenRouter key (optional; for **In context** tab) |

Already set in `render.yaml` if you use Blueprint sync: `LLM_MODEL_NAME`, `PREWARM_SEMANTIC=false`, etc.

## Load MedDRA into Aura (once, from your laptop)

```bash
export NEO4J_URI='neo4j+s://YOUR-ID.databases.neo4j.io'
export NEO4J_USER=neo4j
export NEO4J_PASS='your-aura-password'
PYTHONPATH=. python data/build_graph.py
```

Without this, the app runs but lookups return no matches.

## After deploy

- URL: `https://term-plan-mt.onrender.com`
- Login with `WEBAPP_PASSWORD` (e.g. `term`)
- Logs should show: `Uvicorn running on http://0.0.0.0:...`

## Free tier (512 MB)

Semantic search loads a large model on first use. If the service crashes, upgrade to **Starter** (2 GB) or use **Term** lookup only until upgraded.

## Licence

Do not share the public URL widely. Use `WEBAPP_PASSWORD` and keep Aura credentials secret.
