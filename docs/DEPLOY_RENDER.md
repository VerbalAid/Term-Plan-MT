# Deploy MedDRA lookup on Render

## Fix `uvicorn: command not found`

Your service is still using this **wrong** start command:

```text
uvicorn webapp.main:app --host 0.0.0.0 --port $PORT --workers 1
```

Replace it. Pick **one** option below.

### Option A — Python runtime (edit Start command)

**Settings → Build**

```bash
pip install --upgrade pip && pip install -r requirements-render.txt
```

**Root directory:** leave empty.

**Settings → Deploy → Start command**

```bash
bash start.sh
```

**Health check path:** `/api/health`

Save, then **Manual Deploy → Clear build cache & deploy**.

### Option B — Docker runtime (ignores a bad Start command)

1. **Settings → General** → change **Runtime** from *Python* to **Docker**.
2. Leave **Dockerfile path** as `./Dockerfile`.
3. Clear **Start command** (Docker uses the Dockerfile `CMD`).
4. **Build command** can be empty for Docker.
5. Redeploy with cache clear.

## Neo4j Aura environment variables

From the Aura “Connection details” panel (not localhost):

| Key | Example |
|-----|---------|
| `NEO4J_URI` | `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASS` | password shown once when the instance was created |

Also set `LLM_API_KEY` (OpenRouter) if you use **In context**.

Load the MedDRA graph into Aura from your machine:

```bash
export NEO4J_URI='neo4j+s://YOUR-ID.databases.neo4j.io'
export NEO4J_USER=neo4j
export NEO4J_PASS='your-aura-password'
PYTHONPATH=. python data/build_graph.py
```

## Success

Deploy logs should contain:

```text
Uvicorn running on http://0.0.0.0:...
```

Not `uvicorn: command not found`.
