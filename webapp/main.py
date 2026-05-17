"""MedDRA Lookup Tool — standalone FastAPI app (deploy on Render or run locally)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
load_dotenv(WEBAPP_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

from webapp.lookup import get_lookup_service

log = logging.getLogger(__name__)
STATIC = WEBAPP_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    svc = get_lookup_service()
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
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LookupRequest(BaseModel):
    term: str = Field(..., min_length=1, max_length=500)
    lang: str = Field(default="auto", description="fr | en | auto")


@app.get("/api/health")
def api_health():
    return get_lookup_service().health()


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


@app.post("/api/lookup")
def api_lookup(body: LookupRequest):
    try:
        return get_lookup_service().lookup(body.term, lang=body.lang).to_dict()
    except Exception as exc:
        log.exception("lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/lookup")
def api_lookup_get(
    term: str = Query(..., min_length=1, max_length=500),
    lang: str = Query("auto"),
):
    try:
        return get_lookup_service().lookup(term, lang=lang).to_dict()
    except Exception as exc:
        log.exception("lookup failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
