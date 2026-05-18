"""Optional HTTP Basic gate — required for licensed MedDRA deployments on the public internet."""

from __future__ import annotations

import base64
import logging
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)

# Render health probes hit this path without credentials.
_HEALTH_PATH = "/api/health"


def access_gate_enabled() -> bool:
    return bool(os.environ.get("WEBAPP_PASSWORD", "").strip())


def access_username() -> str:
    return os.environ.get("WEBAPP_USERNAME", "termplan").strip() or "termplan"


def _expected_credentials() -> tuple[str, str] | None:
    password = os.environ.get("WEBAPP_PASSWORD", "").strip()
    if not password:
        return None
    return access_username(), password


def _parse_basic(authorization: str | None) -> tuple[str, str] | None:
    if not authorization or not authorization.lower().startswith("basic "):
        return None
    try:
        raw = base64.b64decode(authorization.split(" ", 1)[1].strip()).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if ":" not in raw:
        return None
    user, pwd = raw.split(":", 1)
    return user, pwd


def credentials_valid(authorization: str | None) -> bool:
    expected = _expected_credentials()
    if expected is None:
        return True
    got = _parse_basic(authorization)
    if got is None:
        return False
    exp_user, exp_pass = expected
    return secrets.compare_digest(got[0], exp_user) and secrets.compare_digest(got[1], exp_pass)


class AccessGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not access_gate_enabled():
            return await call_next(request)
        if request.url.path == _HEALTH_PATH:
            return await call_next(request)
        if credentials_valid(request.headers.get("authorization")):
            return await call_next(request)
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="MedDRA Lookup"'},
            content="Authentication required.",
            media_type="text/plain",
        )


def warn_if_public() -> None:
    if access_gate_enabled():
        log.info("Access gate enabled (HTTP Basic).")
        return
    log.warning(
        "WEBAPP_PASSWORD is not set — UI and API are public. "
        "Do not expose licensed MedDRA data on the internet without a password."
    )
