"""MedDRA Lookup Tool — standalone FastAPI app (deploy on Render or run locally)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
# Repo defaults first; webapp/.env wins (overrides shell exports for local dev).
load_dotenv(REPO_ROOT / ".env")
load_dotenv(WEBAPP_DIR / ".env", override=True)

from webapp.auth import (
    AccessGateMiddleware,
    access_gate_enabled,
    password_ok,
    session_valid,
    set_session_cookie,
    warn_if_public,
)
from webapp.lookup import get_lookup_service
from webapp.neo4j_config import validate_neo4j_config

log = logging.getLogger(__name__)
STATIC = WEBAPP_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warn_if_public()
    validate_neo4j_config()
    from webapp.context_llm import llm_configured, llm_model

    if llm_configured():
        log.info("Context LLM: %s", llm_model())
    else:
        log.warning("Context LLM disabled (set LLM_API_KEY in webapp/.env)")
    svc = get_lookup_service()
    svc.prewarm_graph()
    if os.environ.get("PREWARM_SEMANTIC", "").lower() in ("1", "true", "yes"):
        svc.prewarm_semantic()
    yield
    from webapp import lookup as lookup_mod

    if lookup_mod._service is not None:
        lookup_mod._service.close()
        lookup_mod._service = None


app = FastAPI(
    title="MedDRA Lookup Tool",
    description=(
        "Standalone browser + API for MedDRA v28 concept search "
        "(exact → fuzzy → semantic) with hierarchy navigation."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

_cors = os.environ.get("CORS_ORIGINS", "")
if access_gate_enabled() and (not _cors or _cors.strip() == "*"):
    _cors_origins: list[str] = []
else:
    _cors_origins = [o.strip() for o in (_cors or "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessGateMiddleware)


class LookupRequest(BaseModel):
    term: str = Field(..., min_length=1, max_length=500)
    lang: str = Field(default="auto", description="fr | en | auto")


class ContextLookupRequest(BaseModel):
    context_sentence: str = Field(..., min_length=1, max_length=2000)
    target_term: str = Field(..., min_length=1, max_length=500)
    lang: str = Field(default="auto", description="fr | en | auto")


@app.get("/api/health")
def api_health():
    data = get_lookup_service().health()
    if access_gate_enabled() and data.get("status") == "ok":
        return {
            "status": "ok",
            "access_gate": True,
            "neo4j": data.get("neo4j"),
            "labels_loaded": data.get("labels_loaded"),
            "cache_ready": data.get("cache_ready"),
            "semantic_disabled": data.get("semantic_disabled"),
            "semantic_ready": data.get("semantic_ready"),
            "llm_configured": data.get("llm_configured"),
            "llm_model": data.get("llm_model"),
        }
    return data


@app.get("/api/graph/schema")
def api_graph_schema():
    try:
        return get_lookup_service()._graph.schema_summary()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/debug/neighborhood")
def api_debug_neighborhood(
    term: str = Query(..., min_length=1),
    lang: str = Query("auto"),
):
    """Inspect hierarchy edges for a term (debugging empty parents/children)."""
    svc = get_lookup_service()
    result = svc.lookup(term, lang=lang)
    if not result.concept:
        return {"lookup": result.to_dict(), "neighborhood": None}
    return {
        "lookup": result.to_dict(),
        "neighborhood": svc._graph.neighborhood(
            {
                "id": result.concept.id,
                "name": result.concept.name,
                "fr_label": result.concept.fr_label,
            }
        ),
    }


@app.get("/api/concept/{concept_id}")
async def api_concept_by_id(
    concept_id: str,
    lang: str = Query("auto"),
):
    """Hierarchy navigation: fetch one concept with parents, children, and ancestor chain."""
    try:
        return await run_in_threadpool(
            lambda: get_lookup_service().lookup_by_id(concept_id, lang=lang).to_dict()
        )
    except Exception as exc:
        log.exception("concept lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/lookup")
async def api_lookup(body: LookupRequest):
    try:
        return await run_in_threadpool(
            lambda: get_lookup_service().lookup(body.term, lang=body.lang).to_dict()
        )
    except Exception as exc:
        log.exception("lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/lookup")
async def api_lookup_get(
    term: str = Query(..., min_length=1, max_length=500),
    lang: str = Query("auto"),
):
    try:
        return await run_in_threadpool(
            lambda: get_lookup_service().lookup(term, lang=lang).to_dict()
        )
    except Exception as exc:
        log.exception("lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/context-lookup")
async def api_context_lookup(body: ContextLookupRequest):
    """In-context lookup: graph candidates plus OpenRouter disambiguation."""
    try:
        return await run_in_threadpool(
            lambda: get_lookup_service()
            .context_lookup(
                body.context_sentence,
                body.target_term,
                lang=body.lang,
            )
            .to_dict()
        )
    except Exception as exc:
        log.exception("context lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _login_html(next_path: str, *, error: bool) -> str:
    template = (STATIC / "login.html").read_text(encoding="utf-8")
    err_block = '<div class="err">Incorrect password.</div>' if error else ""
    return template.replace("<!--ERROR-->", err_block).replace("__NEXT__", next_path)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", error: int = 0):
    if not access_gate_enabled():
        return RedirectResponse("/")
    if session_valid(request):
        dest = next if next.startswith("/") and not next.startswith("//") else "/"
        return RedirectResponse(dest)
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    return _login_html(safe_next, error=bool(error))


@app.post("/login")
def login_submit(
    request: Request,
    password: str = Form(...),
    next: str = Form("/"),
):
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    if not password_ok(password):
        return RedirectResponse(f"/login?next={safe_next}&error=1", status_code=303)
    resp = RedirectResponse(safe_next, status_code=303)
    set_session_cookie(resp, secure=request.url.scheme == "https")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("meddra_session", path="/")
    return resp


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
