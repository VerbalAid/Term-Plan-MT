"""TermPlanMT MedDRA lookup — FastAPI app for Render / local dev."""

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

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from webapp.lookup import get_lookup_service

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("PREWARM_SEMANTIC", "").lower() in ("1", "true", "yes"):
        try:
            get_lookup_service()._ensure_semantic_index()
        except Exception as exc:
            log.warning("Semantic prewarm skipped: %s", exc)
    yield
    global _service  # noqa: PLW0603 — lifespan teardown
    from webapp import lookup as lookup_mod

    if lookup_mod._service is not None:
        lookup_mod._service.close()
        lookup_mod._service = None


app = FastAPI(
    title="TermPlanMT MedDRA Lookup",
    description="French/English term → MedDRA concept with hierarchy (string, fuzzy, semantic).",
    version="1.0.0",
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
    lang: str = Field(default="fr")


@app.get("/api/health")
def api_health():
    return get_lookup_service().health()


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
    lang: str = Query("fr"),
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
