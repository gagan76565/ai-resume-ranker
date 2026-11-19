"""One-off: import a v1.x SQLite database into the v2 schema.

Usage (from the project root, with DATABASE_URL set to the target database
or unset for local SQLite):

    python scripts/import_sqlite.py instance/resume_ranker.db [--dry-run]

Equivalent to ``flask import-sqlite instance/resume_ranker.db``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens.config import load_settings  # noqa: E402
from talentlens.db import make_engine, upgrade  # noqa: E402
from talentlens.importer import import_legacy_sqlite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to the v1.x SQLite file")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be imported without writing")
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = make_engine(settings.database_url)
    upgrade(engine)
    summary = import_legacy_sqlite(engine, args.path, dry_run=args.dry_run)
    prefix = "[dry run] " if args.dry_run else ""
    print(
        f"{prefix}Target: {settings.database_backend}. Found {summary['screenings_found']} screening(s); "
        f"imported {summary['imported']} ({summary['candidates_imported']} candidates); "
        f"skipped {summary['skipped_existing']} already imported."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
