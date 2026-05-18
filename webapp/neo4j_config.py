"""Neo4j connection settings with clear errors for Render vs local."""

from __future__ import annotations

import os
from urllib.parse import urlparse

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_render_host() -> bool:
    return bool(os.environ.get("RENDER"))


def neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687").strip()


def neo4j_user() -> str:
    return os.environ.get("NEO4J_USER", "neo4j").strip() or "neo4j"


def neo4j_password() -> str:
    return os.environ.get("NEO4J_PASS", "password")


def _host_from_uri(uri: str) -> str:
    try:
        return (urlparse(uri).hostname or "").lower()
    except Exception:
        return ""


def is_local_uri(uri: str | None = None) -> bool:
    host = _host_from_uri(uri or neo4j_uri())
    return host in _LOCAL_HOSTS


def validate_neo4j_config() -> None:
    """Fail fast on Render when Neo4j still points at localhost."""
    uri = neo4j_uri()
    if not uri:
        raise RuntimeError("NEO4J_URI is empty. Set your Aura URI on Render.")

    if is_render_host() and is_local_uri(uri):
        raise RuntimeError(
            "NEO4J_URI points at localhost but this app runs on Render. "
            "In Render → Environment set NEO4J_URI to your Neo4j Aura URL, e.g. "
            "neo4j+s://xxxx.databases.neo4j.io (not bolt://localhost:7687)."
        )

    if is_render_host() and not uri.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
        raise RuntimeError(
            f"NEO4J_URI looks invalid: {uri!r}. Use an Aura URI like neo4j+s://….databases.neo4j.io"
        )


def friendly_connection_error(exc: Exception) -> str:
    if is_local_uri() and is_render_host():
        return (
            "Neo4j is not configured for Render. Set NEO4J_URI to your Aura "
            "connection string (neo4j+s://….databases.neo4j.io), NEO4J_USER=neo4j, "
            "and NEO4J_PASS in the Render dashboard, then redeploy."
        )
    if is_local_uri():
        return (
            "Cannot reach Neo4j at localhost:7687. Start Docker: "
            "docker compose up -d — or set NEO4J_URI to your Aura URL."
        )
    host = _host_from_uri() or "database"
    return f"Cannot reach Neo4j at {host}. Check NEO4J_URI, NEO4J_USER, and NEO4J_PASS on Render."


def safe_uri_hint() -> str:
    uri = neo4j_uri()
    host = _host_from_uri(uri) or "unknown"
    scheme = urlparse(uri).scheme or "bolt"
    return f"{scheme}://{host}"
