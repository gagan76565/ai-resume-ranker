"""HTML pages and health endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from .auth import is_owner, login_session, logout_session, owner_required, safe_next, verify_credentials
from .db import ping
from .errors import ApiError
from .security import client_key, rotate_csrf_token

pages = Blueprint("pages", __name__)
health = Blueprint("health", __name__)


def _state():
    return current_app.extensions["talentlens"]


@pages.get("/")
def index():
    return render_template("index.html", page="screen")


@pages.get("/results")
def results():
    """Results of the latest screening in this browser tab (kept in sessionStorage, never on the server)."""
    return render_template("results.html", page="results", screening_id=None)


@pages.get("/screenings/<int:screening_id>")
@owner_required
def saved_results(screening_id: int):
    return render_template("results.html", page="history", screening_id=screening_id)


@pages.get("/history")
@owner_required
def history():
    return render_template("history.html", page="history")


@pages.route("/login", methods=["GET", "POST"])
def login():
    settings = _state().settings
    next_url = safe_next(request.values.get("next"))
    if request.method == "GET":
        if is_owner():
            return redirect(next_url)
        return render_template("login.html", page="login", next_url=next_url, error=None)

    if not settings.owner_login_enabled:
        return render_template("login.html", page="login", next_url=next_url,
                               error="Owner login is not configured on this server."), 403
    limiter = _state().limiters["login"]
    wait = limiter.blocked_for(client_key())
    if wait:
        minutes = max(1, round(wait / 60))
        return render_template("login.html", page="login", next_url=next_url,
                               error=f"Too many failed attempts. Try again in about {minutes} minute(s)."), 429, {"Retry-After": str(wait)}
    username = request.form.get("username", "")[:200]
    password = request.form.get("password", "")[:1024]
    if not verify_credentials(username, password):
        limiter.hit(client_key())
        current_app.logger.warning("Failed owner login from %s", client_key())
        return render_template("login.html", page="login", next_url=next_url,
                               error="Incorrect username or password."), 401
    limiter.reset(client_key())
    login_session(settings.admin_username)
    rotate_csrf_token()
    return redirect(next_url)


@pages.post("/logout")
def logout():
    logout_session()
    return redirect(url_for("pages.index"))


@health.get("/healthz")
def healthz():
    """Liveness: no database access, so it stays cheap during cold starts."""
    return jsonify({"status": "ok"})


@health.get("/readyz")
def readyz():
    """Readiness: verifies the database answers ``SELECT 1``."""
    try:
        ping(_state().engine)
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Readiness check failed")
        raise ApiError(503, "database_unavailable", "The database is not reachable yet.") from None
    return jsonify({"status": "ready", "database": "ok"})
