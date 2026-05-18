"""Password gate for licensed MedDRA deployments (form login + optional HTTP Basic)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

log = logging.getLogger(__name__)

_HEALTH_PATH = "/api/health"
_LOGIN_PATH = "/login"
_SESSION_COOKIE = "meddra_session"
_SESSION_SALT = b"meddra-lookup-v1"


def access_gate_enabled() -> bool:
    return bool(os.environ.get("WEBAPP_PASSWORD", "").strip())


def access_username() -> str:
    return os.environ.get("WEBAPP_USERNAME", "termplan").strip() or "termplan"


def webapp_password() -> str:
    return os.environ.get("WEBAPP_PASSWORD", "").strip()


def session_cookie_value() -> str:
    return hmac.new(webapp_password().encode(), _SESSION_SALT, hashlib.sha256).hexdigest()


def password_ok(candidate: str) -> bool:
    if not access_gate_enabled():
        return True
    return secrets.compare_digest(candidate, webapp_password())


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


def basic_auth_valid(authorization: str | None) -> bool:
    if not access_gate_enabled():
        return True
    got = _parse_basic(authorization)
    if got is None:
        return False
    exp_user, exp_pass = access_username(), webapp_password()
    return secrets.compare_digest(got[0], exp_user) and secrets.compare_digest(got[1], exp_pass)


def session_valid(request: Request) -> bool:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token:
        return False
    return secrets.compare_digest(token, session_cookie_value())


def request_authenticated(request: Request) -> bool:
    if not access_gate_enabled():
        return True
    if basic_auth_valid(request.headers.get("authorization")):
        return True
    return session_valid(request)


def login_redirect(next_path: str) -> RedirectResponse:
    safe = next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"
    return RedirectResponse(f"{_LOGIN_PATH}?next={quote(safe)}", status_code=303)


def set_session_cookie(response: Response, *, secure: bool) -> None:
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=session_cookie_value(),
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 14,
        path="/",
    )


class AccessGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not access_gate_enabled():
            return await call_next(request)

        path = request.url.path
        if path in (_HEALTH_PATH, _LOGIN_PATH):
            return await call_next(request)

        if request_authenticated(request):
            return await call_next(request)

        if path.startswith("/api/"):
            return Response(
                status_code=401,
                content='{"detail":"Login required"}',
                media_type="application/json",
            )

        return login_redirect(path)


def warn_if_public() -> None:
    if access_gate_enabled():
        log.info("Access gate enabled (password form).")
        return
    log.warning(
        "WEBAPP_PASSWORD is not set — UI and API are public. "
        "Do not expose licensed MedDRA data on the internet without a password."
    )
