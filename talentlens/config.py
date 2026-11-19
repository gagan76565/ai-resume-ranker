"""Configuration loaded from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from scoring import ScoringConfig

from .uploads import MB, UploadLimits

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int, problems: list[str]) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        problems.append(f"{name}={raw!r} is not an integer; using {default}.")
        return default
    if not minimum <= value <= maximum:
        problems.append(f"{name}={raw!r} must be between {minimum} and {maximum}; using {default}.")
        return default
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def normalise_database_url(url: str) -> str:
    """Accept the URL styles that Render, Neon and Supabase hand out.

    ``postgres://`` and ``postgresql://`` are rewritten to the explicit
    ``postgresql+psycopg2://`` driver form understood by SQLAlchemy 2.
    """
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


@dataclass(frozen=True)
class Settings:
    env: str
    secret_key: str
    secret_key_generated: bool
    database_url: str
    admin_username: str | None
    admin_password_hash: str | None
    retention_days: int
    purge_token: str | None
    upload_limits: UploadLimits
    scoring: ScoringConfig
    trust_proxy: int
    login_attempts: int
    login_window_seconds: int
    screen_requests: int
    screen_window_seconds: int
    auto_migrate: bool
    problems: tuple[str, ...] = field(default=())

    @property
    def production(self) -> bool:
        return self.env == "production"

    @property
    def owner_login_enabled(self) -> bool:
        return bool(self.admin_username and self.admin_password_hash)

    @property
    def database_backend(self) -> str:
        return "postgresql" if self.database_url.startswith("postgresql") else self.database_url.split(":", 1)[0]


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from ``env`` (defaults to ``os.environ``).

    With no variables set at all the app runs in development mode on a local
    SQLite file under ``instance/``.
    """
    env = os.environ if env is None else env
    problems: list[str] = []

    app_env = (env.get("APP_ENV") or "").strip().lower()
    if not app_env:
        app_env = "production" if env.get("RENDER") else "development"
    if app_env not in {"development", "production", "testing"}:
        problems.append(f"APP_ENV={app_env!r} is unknown; using development.")
        app_env = "development"

    secret = (env.get("SECRET_KEY") or "").strip()
    generated = False
    if not secret:
        secret = secrets.token_urlsafe(48)
        generated = True
        problems.append(
            "SECRET_KEY is not set; generated a random one for this process. "
            "Logins will reset whenever the server restarts."
        )

    database_url = (env.get("DATABASE_URL") or "").strip()
    if database_url:
        database_url = normalise_database_url(database_url)
    else:
        data_dir = Path(env.get("DATA_DIR") or (BASE_DIR / "instance"))
        data_dir.mkdir(parents=True, exist_ok=True)
        database_url = "sqlite:///" + str((data_dir / "talentlens.db").resolve()).replace("\\", "/")

    admin_user = (env.get("ADMIN_USERNAME") or "").strip() or None
    admin_hash = (env.get("ADMIN_PASSWORD_HASH") or "").strip() or None
    if bool(admin_user) != bool(admin_hash):
        problems.append("Owner login needs both ADMIN_USERNAME and ADMIN_PASSWORD_HASH; login is disabled.")
        admin_user = admin_hash = None

    limits = UploadLimits(
        max_file_bytes=_int(env, "MAX_FILE_MB", 5, 1, 20, problems) * MB,
        max_files=_int(env, "MAX_FILES", 20, 1, 50, problems),
        max_total_bytes=_int(env, "MAX_TOTAL_MB", 30, 1, 100, problems) * MB,
        max_pages=_int(env, "MAX_PDF_PAGES", 30, 1, 200, problems),
        max_chars=_int(env, "MAX_RESUME_CHARS", 100_000, 1_000, 500_000, problems),
    )
    scoring = ScoringConfig.from_env(env)
    problems.extend(scoring.warnings)

    purge_token = (env.get("PURGE_TOKEN") or "").strip() or None
    if purge_token and len(purge_token) < 24:
        problems.append("PURGE_TOKEN is shorter than 24 characters; the purge endpoint is disabled.")
        purge_token = None

    return Settings(
        env=app_env,
        secret_key=secret,
        secret_key_generated=generated,
        database_url=database_url,
        admin_username=admin_user,
        admin_password_hash=admin_hash,
        retention_days=_int(env, "RETENTION_DAYS", 30, 1, 3650, problems),
        purge_token=purge_token,
        upload_limits=limits,
        scoring=scoring,
        trust_proxy=_int(env, "TRUST_PROXY", 1 if env.get("RENDER") else 0, 0, 5, problems),
        login_attempts=_int(env, "LOGIN_RATE_LIMIT", 5, 1, 1000, problems),
        login_window_seconds=_int(env, "LOGIN_RATE_WINDOW", 900, 10, 86400, problems),
        screen_requests=_int(env, "SCREEN_RATE_LIMIT", 30, 1, 10000, problems),
        screen_window_seconds=_int(env, "SCREEN_RATE_WINDOW", 600, 10, 86400, problems),
        auto_migrate=_bool(env, "AUTO_MIGRATE", True),
        problems=tuple(problems),
    )
