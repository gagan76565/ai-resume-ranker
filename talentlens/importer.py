"""Import a v1.x SQLite database (``instance/resume_ranker.db``) into the v2 schema.

v1.x rows only stored a TF-IDF similarity, matched/missing *terms* and an
explanation. They are imported as-is and marked ``scoring_version =
"1.x-import"``; they are not re-scored, because the original PDFs were never
stored. Re-running the import skips screenings already imported (tracked by
``screenings.legacy_id``).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import Engine

from scoring import label_for

from . import repository

LEGACY_VERSION = "1.x-import"


def _parse_time(value: str | None) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return repository.utcnow()


def _json_list(value) -> list:
    try:
        data = json.loads(value or "[]")
        return [str(x) for x in data] if isinstance(data, list) else []
    except (TypeError, ValueError):
        return []


def import_legacy_sqlite(engine: Engine, path: str | Path, dry_run: bool = False) -> dict:
    """Copy screenings and candidates from a v1.x SQLite file. Returns counts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No SQLite file at {path}")
    source = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"screenings", "candidates"} <= tables:
            raise ValueError("This file does not contain the v1.x 'screenings' and 'candidates' tables.")
        already = repository.legacy_ids(engine)
        summary = {"screenings_found": 0, "imported": 0, "skipped_existing": 0, "candidates_imported": 0}
        for row in source.execute("SELECT * FROM screenings ORDER BY id"):
            summary["screenings_found"] += 1
            if row["id"] in already:
                summary["skipped_existing"] += 1
                continue
            cand_rows = source.execute(
                "SELECT * FROM candidates WHERE screening_id = ? ORDER BY score DESC, filename ASC", (row["id"],)
            ).fetchall()
            ranked = []
            for rank, c in enumerate(cand_rows, start=1):
                score = float(c["score"] or 0)
                matched = _json_list(c["matched_terms"])
                missing = _json_list(c["missing_terms"])
                breakdown = {
                    "final_score": score, "label": label_for(score), "tfidf_score": score,
                    "skill_coverage": None, "section_signal": None,
                    "matched_skills": matched, "missing_skills": missing, "must_have_missing": [],
                    "top_terms": matched, "snippets": [], "capped": False, "low_text": False,
                    "warnings": ["Imported from v1.x: score is TF-IDF similarity only (not re-scored)."],
                    "scoring_version": LEGACY_VERSION,
                }
                ranked.append({
                    "rank": rank, "filename": c["filename"], "score": score, "explanation": c["explanation"] or "",
                    "breakdown": breakdown,
                })
            summary["candidates_imported"] += len(ranked)
            summary["imported"] += 1
            if not dry_run:
                repository.create_screening(
                    engine,
                    job_title=(row["job_title"] or "Untitled role")[:120],
                    job_description=row["job_description"] or "",
                    must_have=[], settings={"imported_from": "v1.x"}, scoring_version=LEGACY_VERSION,
                    ranked=ranked, created_at=_parse_time(row["created_at"]), legacy_id=int(row["id"]),
                )
        return summary
    finally:
        source.close()
