"""Database schema (SQLAlchemy Core) and engine/migration helpers.

The schema is created and evolved only through Alembic migrations in
``migrations/versions``; ``metadata`` here mirrors it for queries and tests
(``tests/test_db.py`` checks they agree).
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Engine

BASE_DIR = Path(__file__).resolve().parent.parent

metadata = sa.MetaData()

screenings = sa.Table(
    "screenings",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("job_title", sa.String(120), nullable=False),
    sa.Column("job_description", sa.Text, nullable=False),
    sa.Column("candidate_count", sa.Integer, nullable=False, server_default="0"),
    sa.Column("must_have", sa.JSON, nullable=False),
    sa.Column("settings", sa.JSON, nullable=False),
    sa.Column("scoring_version", sa.String(20), nullable=False),
    sa.Column("legacy_id", sa.Integer, nullable=True, unique=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
)

candidates = sa.Table(
    "candidates",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("screening_id", sa.Integer, sa.ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("rank", sa.Integer, nullable=False),
    sa.Column("filename", sa.String(255), nullable=False),
    sa.Column("final_score", sa.Float, nullable=False),
    sa.Column("tfidf_score", sa.Float, nullable=True),
    sa.Column("skill_coverage", sa.Float, nullable=True),
    sa.Column("section_signal", sa.Float, nullable=True),
    sa.Column("label", sa.String(40), nullable=False),
    sa.Column("explanation", sa.Text, nullable=False),
    sa.Column("breakdown", sa.JSON, nullable=False),
    sa.Column("shortlisted", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("notes", sa.Text, nullable=False, server_default=""),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
)


def _sqlite_pragmas(dbapi_connection, _record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def make_engine(url: str) -> Engine:
    """Create an engine with settings suited to a 512 MB single-worker instance."""
    if url.startswith("sqlite"):
        engine = sa.create_engine(url, connect_args={"check_same_thread": False, "timeout": 15})
        event.listen(engine, "connect", _sqlite_pragmas)
        return engine
    return sa.create_engine(
        url,
        pool_size=3,
        max_overflow=2,
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args={"connect_timeout": 10, "application_name": "talentlens"},
    )


def alembic_config(url: str | None = None):
    from alembic.config import Config

    config = Config(str(BASE_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BASE_DIR / "migrations"))
    if url:
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Apply Alembic migrations on ``engine`` (idempotent)."""
    from alembic import command

    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def current_revision(engine: Engine) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def ping(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 1"))
