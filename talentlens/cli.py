"""``flask`` CLI commands: hash-password, purge-old, db-upgrade, import-sqlite."""

from __future__ import annotations

import click
from flask import current_app
from werkzeug.security import generate_password_hash

from . import db, repository
from .importer import import_legacy_sqlite


def register_cli(app):
    @app.cli.command("hash-password")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True,
                  help="Password to hash (prompted without echo if omitted).")
    def hash_password(password: str):
        """Print a Werkzeug (scrypt) hash for ADMIN_PASSWORD_HASH."""
        if len(password) < 12:
            raise click.BadParameter("Use at least 12 characters.", param_hint="password")
        click.echo(generate_password_hash(password))

    @app.cli.command("purge-old")
    @click.option("--days", type=int, default=None, help="Override RETENTION_DAYS for this run.")
    @click.option("--dry-run", is_flag=True, help="Only report how many screenings would be deleted.")
    def purge_old(days: int | None, dry_run: bool):
        """Delete screenings older than RETENTION_DAYS (default 30)."""
        state = current_app.extensions["talentlens"]
        days = days if days is not None else state.settings.retention_days
        if days < 1:
            raise click.BadParameter("--days must be at least 1.")
        count = repository.purge_older_than(state.engine, days, dry_run=dry_run)
        verb = "Would delete" if dry_run else "Deleted"
        click.echo(f"{verb} {count} screening(s) older than {days} day(s).")

    @app.cli.command("db-upgrade")
    def db_upgrade():
        """Apply database migrations (also runs automatically at start-up unless AUTO_MIGRATE=0)."""
        state = current_app.extensions["talentlens"]
        db.upgrade(state.engine)
        click.echo(f"Database is at revision {db.current_revision(state.engine)} ({state.settings.database_backend}).")

    @app.cli.command("import-sqlite")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False))
    @click.option("--dry-run", is_flag=True)
    def import_sqlite(path: str, dry_run: bool):
        """Import a v1.x SQLite file (e.g. instance/resume_ranker.db) into the current database."""
        state = current_app.extensions["talentlens"]
        summary = import_legacy_sqlite(state.engine, path, dry_run=dry_run)
        prefix = "[dry run] " if dry_run else ""
        click.echo(
            f"{prefix}Found {summary['screenings_found']} screening(s): imported {summary['imported']} "
            f"({summary['candidates_imported']} candidates), skipped {summary['skipped_existing']} already imported."
        )
