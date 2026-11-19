"""TalentLens web application (Flask app factory)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Mapping

from flask import Flask
from sqlalchemy.engine import Engine
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Settings, load_settings
from .security import RateLimiter

__version__ = "2.0.0"


@dataclass
class AppState:
    settings: Settings
    engine: Engine
    limiters: dict[str, RateLimiter] = field(default_factory=dict)


def create_app(env: Mapping[str, str] | None = None) -> Flask:
    """Create the Flask app. ``env`` replaces ``os.environ`` (used by tests)."""
    from . import db
    from .api import api
    from .auth import is_owner
    from .cli import register_cli
    from .errors import register_error_handlers
    from .security import apply_security_headers, check_csrf, get_csrf_token
    from .views import health, pages

    settings = load_settings(env)
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.upload_limits.max_total_bytes + 2 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=1024 * 1024,
        SESSION_COOKIE_NAME="tl_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.production,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        JSON_SORT_KEYS=False,
        TESTING=settings.env == "testing",
    )
    app.json.sort_keys = False
    if settings.trust_proxy:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=settings.trust_proxy, x_proto=settings.trust_proxy)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    for problem in settings.problems:
        app.logger.warning("Config: %s", problem)

    engine = db.make_engine(settings.database_url)
    if settings.auto_migrate:
        db.upgrade(engine)

    app.extensions["talentlens"] = AppState(
        settings=settings,
        engine=engine,
        limiters={
            "login": RateLimiter(settings.login_attempts, settings.login_window_seconds),
            "screen": RateLimiter(settings.screen_requests, settings.screen_window_seconds),
            "purge": RateLimiter(10, 3600),
        },
    )

    app.before_request(check_csrf)
    app.after_request(apply_security_headers)
    register_error_handlers(app)
    app.register_blueprint(health)
    app.register_blueprint(api)
    app.register_blueprint(pages)
    register_cli(app)

    @app.context_processor
    def _template_globals():
        limits = settings.upload_limits
        return {
            "csrf_token": get_csrf_token,
            "is_owner": is_owner(),
            "owner_login_enabled": settings.owner_login_enabled,
            "retention_days": settings.retention_days,
            "app_version": __version__,
            "limits": {
                "max_file_mb": limits.max_file_bytes // (1024 * 1024),
                "max_files": limits.max_files,
                "max_total_mb": limits.max_total_bytes // (1024 * 1024),
            },
        }

    app.logger.info("TalentLens %s started (%s, database=%s)", __version__, settings.env, settings.database_backend)
    return app
