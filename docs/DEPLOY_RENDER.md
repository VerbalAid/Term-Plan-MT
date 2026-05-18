# Render deployment (term-plan-mt)

## Access model (unlisted link + strong password)

1. **Secret link** — only people with the full URL can reach the login page.  
   Anyone visiting `https://term-plan-mt.onrender.com/` without the key sees **404 Not found**.
2. **Strong password** — min. 16 characters, letters + digits; weak values like `term` are rejected at startup.
3. **No search indexing** — `robots.txt` blocks crawlers; pages send `noindex`.

Generate secrets locally:

```bash
python tools/generate_webapp_secrets.py
```

Copy the two lines into Render **Environment** (never commit them).

**Share with supervisors:** the printed URL  
`https://term-plan-mt.onrender.com/?k=YOUR_LINK_KEY`  
They open it once, then sign in with `WEBAPP_PASSWORD`.

## Render service settings

1. **Runtime:** Docker  
2. **Dockerfile path:** `Dockerfile`  
3. **Root directory:** empty  
4. **Build / Start commands:** empty  
5. **Health check:** `/api/health`

**Manual Deploy → Clear build cache & deploy**

## Environment variables

| Key | Required | Notes |
|-----|----------|--------|
| `WEBAPP_LINK_KEY` | Yes | ≥24 chars; use `generate_webapp_secrets.py` |
| `WEBAPP_PASSWORD` | Yes | ≥16 chars, letters + digits |
| `NEO4J_URI` | **Yes** | Aura only: `neo4j+s://xxxx.databases.neo4j.io` — **never** `localhost` on Render |
| `NEO4J_USER` | Yes | `neo4j` |
| `NEO4J_PASS` | Yes | Aura password |
| `LLM_API_KEY` | No | OpenRouter, for **In context** |

## Load MedDRA into Aura (once)

```bash
export NEO4J_URI='neo4j+s://YOUR-ID.databases.neo4j.io'
export NEO4J_USER=neo4j
export NEO4J_PASS='your-aura-password'
PYTHONPATH=. python data/build_graph.py
```

## Licence

Share link + password only with people covered by your MedDRA academic licence. Do not post the URL on public pages.
