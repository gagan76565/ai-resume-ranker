"""Single-owner authentication.

The owner account comes from ``ADMIN_USERNAME`` and ``ADMIN_PASSWORD_HASH``
(a Werkzeug hash produced by ``flask hash-password``). Passwords are never
stored or logged in plain text.
"""

from __future__ import annotations

import hashlib
import hmac
from functools import wraps
from urllib.parse import urlparse

from flask import current_app, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .errors import ApiError

# Used to keep the timing of "unknown user" close to "wrong password".
_DUMMY_HASH = generate_password_hash("talentlens-timing-guard")


def _state():
    return current_app.extensions["talentlens"]


def session_fingerprint() -> str:
    """Ties a session to the current password hash: changing the password signs everyone out."""
    settings = _state().settings
    material = (settings.admin_username or "") + "\0" + (settings.admin_password_hash or "")
    return hmac.new(settings.secret_key.encode(), material.encode(), hashlib.sha256).hexdigest()[:32]


def is_owner() -> bool:
    settings = _state().settings
    if not settings.owner_login_enabled:
        return False
    user = session.get("owner")
    fingerprint = session.get("auth_v")
    if not user or not fingerprint:
        return False
    return hmac.compare_digest(str(user), settings.admin_username) and hmac.compare_digest(str(fingerprint), session_fingerprint())


def verify_credentials(username: str, password: str) -> bool:
    settings = _state().settings
    if not settings.owner_login_enabled:
        check_password_hash(_DUMMY_HASH, password or "")
        return False
    user_ok = hmac.compare_digest((username or "").strip().encode(), settings.admin_username.encode())
    try:
        password_ok = check_password_hash(settings.admin_password_hash, password or "")
    except (ValueError, TypeError):
        current_app.logger.error("ADMIN_PASSWORD_HASH is not a valid Werkzeug hash; regenerate it with `flask hash-password`.")
        password_ok = False
    return user_ok and password_ok


def login_session(username: str) -> None:
    session.clear()  # prevent session fixation
    session["owner"] = username
    session["auth_v"] = session_fingerprint()
    session.permanent = True


def logout_session() -> None:
    session.clear()


def safe_next(target: str | None) -> str:
    """Only allow same-site relative redirects after login."""
    if not target:
        return url_for("pages.index")
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return url_for("pages.index")
    return target


def owner_required(view):
    """API views: 401 JSON. Page views: redirect to the login page."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        if is_owner():
            return view(*args, **kwargs)
        if request.path.startswith("/api/"):
            raise ApiError(401, "unauthorized", "Sign in as the owner to use saved history.")
        return redirect(url_for("pages.login", next=request.full_path.rstrip("?")))

    return wrapper
