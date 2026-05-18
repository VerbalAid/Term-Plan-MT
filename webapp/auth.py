"""Unlisted link key + password gate (licensed MedDRA deployments)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from urllib.parse import quote, urlencode

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

log = logging.getLogger(__name__)

_HEALTH_PATH = "/api/health"
_LOGIN_PATH = "/login"
_ROBOTS_PATH = "/robots.txt"
_SESSION_COOKIE = "meddra_session"
_LINK_COOKIE = "meddra_link"
_SESSION_SALT = b"meddra-lookup-v1"
_LINK_SALT = b"meddra-link-v1"
_MIN_PASSWORD_LEN = 5

# Block only trivial single-word secrets (class demos may use e.g. Term10).
_WEAK_PASSWORDS = frozenset(
    {
        "term",
        "password",
        "meddra",
        "12345",
        "123456",
    }
)


def access_gate_enabled() -> bool:
    return bool(webapp_password())


def _env_secret(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1].strip()
    return val


def webapp_password() -> str:
    return _env_secret("WEBAPP_PASSWORD")


def link_key() -> str:
    return _env_secret("WEBAPP_LINK_KEY")


def link_key_required() -> bool:
    if not access_gate_enabled():
        return False
    return os.environ.get("WEBAPP_SKIP_LINK_KEY", "").lower() not in ("1", "true", "yes")


def password_is_strong(password: str) -> bool:
    if len(password) < _MIN_PASSWORD_LEN:
        return False
    if password.lower() in _WEAK_PASSWORDS:
        return False
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_letter and has_digit


def session_cookie_value() -> str:
    return hmac.new(webapp_password().encode(), _SESSION_SALT, hashlib.sha256).hexdigest()


def link_cookie_value() -> str:
    return hmac.new(link_key().encode(), _LINK_SALT, hashlib.sha256).hexdigest()


def password_ok(candidate: str) -> bool:
    if not access_gate_enabled():
        return True
    expected = webapp_password()
    got = (candidate or "").strip()
    if not expected or len(got) != len(expected):
        return False
    return secrets.compare_digest(got, expected)


def link_key_ok(candidate: str | None) -> bool:
    if not link_key():
        return not link_key_required()
    if not candidate:
        return False
    return secrets.compare_digest(candidate, link_key())


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
    return secrets.compare_digest(got[1], webapp_password())


def session_valid(request: Request) -> bool:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token:
        return False
    return secrets.compare_digest(token, session_cookie_value())


def link_verified(request: Request) -> bool:
    if not link_key_required():
        return True
    token = request.cookies.get(_LINK_COOKIE)
    if token and secrets.compare_digest(token, link_cookie_value()):
        return True
    return link_key_ok(request.query_params.get("k"))


def request_authenticated(request: Request) -> bool:
    if not access_gate_enabled():
        return True
    if basic_auth_valid(request.headers.get("authorization")):
        return True
    return session_valid(request)


def _not_found() -> Response:
    return Response(status_code=404, content="Not found.", media_type="text/plain")


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


def set_link_cookie(response: Response, *, secure: bool) -> None:
    response.set_cookie(
        key=_LINK_COOKIE,
        value=link_cookie_value(),
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 90,
        path="/",
    )


def shareable_entry_url(base_url: str) -> str | None:
    if not link_key():
        return base_url.rstrip("/") + "/"
    q = urlencode({"k": link_key()})
    return f"{base_url.rstrip('/')}/?{q}"


def _dev_mode() -> bool:
    return os.environ.get("WEBAPP_DEV", "").lower() in ("1", "true", "yes")


def validate_deploy_config() -> None:
    if _dev_mode():
        log.info("WEBAPP_DEV=1 — relaxed password/link checks for local use.")
        return
    if not access_gate_enabled():
        log.warning(
            "WEBAPP_PASSWORD not set — site is public. Required for licensed MedDRA on the internet."
        )
        return
    if not password_is_strong(webapp_password()):
        raise RuntimeError(
            f"WEBAPP_PASSWORD must be at least {_MIN_PASSWORD_LEN} characters "
            "with letters and digits, and not a common weak value."
        )
    if link_key_required() and not link_key():
        raise RuntimeError(
            "WEBAPP_LINK_KEY is required (set a long random secret). "
            "Share: https://your-app.onrender.com/?k=YOUR_LINK_KEY"
        )
    if link_key_required() and len(link_key()) < 24:
        raise RuntimeError("WEBAPP_LINK_KEY must be at least 24 characters.")
    log.info("Access gate: unlisted link key + password.")


class AccessGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == _ROBOTS_PATH:
            return PlainTextResponse("User-agent: *\nDisallow: /\n")

        response = await self._dispatch_gated(request, call_next)
        if response is not None:
            response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    async def _dispatch_gated(self, request: Request, call_next):
        if not access_gate_enabled():
            return await call_next(request)

        path = request.url.path
        if path == _HEALTH_PATH:
            return await call_next(request)

        key_param = request.query_params.get("k")
        if link_key_ok(key_param):
            dest = path if path != "/" else "/"
            if request.query_params.get("next"):
                dest = request.query_params.get("next", "/")
            safe_dest = dest if dest.startswith("/") and not dest.startswith("//") else "/"
            resp = RedirectResponse(safe_dest, status_code=303)
            set_link_cookie(resp, secure=request.url.scheme == "https")
            return resp

        if not link_verified(request):
            return _not_found()

        if path == _LOGIN_PATH:
            return await call_next(request)

        if request_authenticated(request):
            return await call_next(request)

        if path.startswith("/api/"):
            return Response(
                status_code=401,
                content='{"detail":"Authentication required"}',
                media_type="application/json",
            )

        return login_redirect(path)


def warn_if_public() -> None:
    validate_deploy_config()
