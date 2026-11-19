"""CSRF protection, rate limiting and HTTP security headers.

CSRF policy
-----------
* A random token is stored in the (signed, HttpOnly) session cookie and
  exposed to pages through ``<meta name="csrf-token">`` and hidden form
  fields.
* Every POST/PUT/PATCH/DELETE must send it (form field ``csrf_token`` or
  header ``X-CSRF-Token``), compared with :func:`hmac.compare_digest`.
* Two documented exceptions:
  - views decorated with :func:`csrf_exempt` that authenticate with a bearer
    token instead of a cookie (the purge endpoint);
  - views decorated with :func:`csrf_anonymous_ok` (stateless screening)
    *when the caller is not signed in*: they store nothing and act with no
    cookie authority, so there is nothing to forge. Once the owner is
    signed in the token is required, because screening may then save.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import deque
from functools import wraps

from flask import current_app, request, session

from .errors import ApiError

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def csrf_exempt(view):
    view._csrf_exempt = True
    return view


def csrf_anonymous_ok(view):
    view._csrf_anonymous_ok = True
    return view


def get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def rotate_csrf_token() -> str:
    session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def csrf_is_valid() -> bool:
    expected = session.get("csrf_token")
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
    if not expected or not supplied:
        return False
    return hmac.compare_digest(str(expected).encode(), str(supplied).encode())


def check_csrf():
    """``before_request`` hook enforcing the policy above."""
    from .auth import is_owner  # local import to avoid a cycle

    if request.method not in UNSAFE_METHODS:
        return None
    view = current_app.view_functions.get(request.endpoint or "")
    if view is None or getattr(view, "_csrf_exempt", False):
        return None
    if getattr(view, "_csrf_anonymous_ok", False) and not is_owner():
        return None
    if not csrf_is_valid():
        raise ApiError(400, "csrf_failed", "Your session expired or the security token is missing. Reload the page and try again.")
    return None


class RateLimiter:
    """In-memory sliding-window limiter, keyed by client IP.

    Suitable for a single gunicorn worker (the Render free setup). State is
    per process and resets on restart; that is acceptable for brute-force
    damping but it is not a distributed limiter.
    """

    def __init__(self, limit: int, window_seconds: int, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits

    def blocked_for(self, key: str) -> int:
        """Seconds until ``key`` may try again (0 if not blocked)."""
        with self._lock:
            now = self.clock()
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return 0
            return max(1, int(self.window - (now - hits[0])) + 1)

    def hit(self, key: str) -> None:
        with self._lock:
            now = self.clock()
            self._prune(key, now).append(now)
            if len(self._hits) > 10_000:  # bound memory under abuse
                for stale in [k for k, v in self._hits.items() if not v][:5000]:
                    self._hits.pop(stale, None)

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


def client_key() -> str:
    return request.remote_addr or "unknown"


def rate_limited(limiter_name: str):
    """Decorator: count every call and reject with 429 once the limit is reached."""

    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            limiter: RateLimiter = current_app.extensions["talentlens"].limiters[limiter_name]
            wait = limiter.blocked_for(client_key())
            if wait:
                raise ApiError(429, "rate_limited", f"Too many requests. Try again in {wait} seconds.",
                               headers={"Retry-After": str(wait)})
            limiter.hit(client_key())
            return view(*args, **kwargs)

        return wrapper

    return decorator


CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; "
    "form-action 'self'; frame-ancestors 'none'"
)


def apply_security_headers(response):
    response.headers.setdefault("Content-Security-Policy", CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if current_app.extensions["talentlens"].settings.production:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if request.path.startswith("/api/") and "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response
