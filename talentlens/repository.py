"""Data access functions. Every function takes an ``Engine`` explicitly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .db import candidates, screenings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    """ISO-8601 in UTC. SQLite returns naive datetimes; they are stored as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _like(term: str) -> str:
    escaped = term.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def create_screening(engine: Engine, *, job_title: str, job_description: str, must_have: list[str],
                     settings: dict, scoring_version: str, ranked: list[dict],
                     created_at: datetime | None = None, legacy_id: int | None = None) -> int:
    """Insert a screening and its candidates in one transaction; return the new id.

    ``ranked`` items are API candidate objects (see ``screening.candidate_payload``).
    """
    created_at = created_at or utcnow()
    with engine.begin() as conn:
        screening_id = conn.execute(
            screenings.insert().values(
                job_title=job_title,
                job_description=job_description,
                candidate_count=len(ranked),
                must_have=list(must_have),
                settings=settings,
                scoring_version=scoring_version,
                legacy_id=legacy_id,
                created_at=created_at,
            )
        ).inserted_primary_key[0]
        if ranked:
            conn.execute(
                candidates.insert(),
                [
                    {
                        "screening_id": screening_id,
                        "rank": item["rank"],
                        "filename": item["filename"],
                        "final_score": item["score"],
                        "tfidf_score": item["breakdown"].get("tfidf_score"),
                        "skill_coverage": item["breakdown"].get("skill_coverage"),
                        "section_signal": item["breakdown"].get("section_signal"),
                        "label": item["breakdown"].get("label", ""),
                        "explanation": item["explanation"],
                        "breakdown": item["breakdown"],
                        "shortlisted": bool(item.get("shortlisted", False)),
                        "notes": item.get("notes", ""),
                    }
                    for item in ranked
                ],
            )
    return int(screening_id)


def _screening_row(row) -> dict:
    return {
        "id": row.id,
        "job_title": row.job_title,
        "candidate_count": row.candidate_count,
        "created_at": iso(row.created_at),
        "must_have": list(row.must_have or []),
        "scoring_version": row.scoring_version,
    }


def candidate_row(row) -> dict:
    breakdown = dict(row.breakdown or {})
    return {
        "id": row.id,
        "screening_id": row.screening_id,
        "rank": row.rank,
        "filename": row.filename,
        "score": row.final_score,
        "matched_terms": breakdown.get("matched_skills", []),
        "missing_terms": breakdown.get("missing_skills", []),
        "explanation": row.explanation,
        "shortlisted": bool(row.shortlisted),
        "notes": row.notes or "",
        "breakdown": breakdown,
    }


def list_screenings(engine: Engine, query: str | None = None, limit: int = 100) -> list[dict]:
    top = (
        sa.select(candidates.c.screening_id, sa.func.max(candidates.c.final_score).label("top_score"),
                  sa.func.avg(candidates.c.final_score).label("average_score"),
                  sa.func.sum(sa.case((candidates.c.shortlisted.is_(True), 1), else_=0)).label("shortlisted"))
        .group_by(candidates.c.screening_id)
        .subquery()
    )
    stmt = (
        sa.select(screenings, top.c.top_score, top.c.average_score, top.c.shortlisted)
        .select_from(screenings.outerjoin(top, top.c.screening_id == screenings.c.id))
        .order_by(screenings.c.created_at.desc(), screenings.c.id.desc())
        .limit(max(1, min(limit, 500)))
    )
    if query:
        pattern = _like(query.strip()[:100])
        stmt = stmt.where(sa.or_(
            sa.func.lower(screenings.c.job_title).like(pattern, escape="\\"),
            sa.func.lower(screenings.c.job_description).like(pattern, escape="\\"),
        ))
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    out = []
    for row in rows:
        item = _screening_row(row)
        item["top_score"] = round(row.top_score, 1) if row.top_score is not None else None
        item["average_score"] = round(row.average_score, 1) if row.average_score is not None else None
        item["shortlisted_count"] = int(row.shortlisted or 0)
        out.append(item)
    return out


def get_screening(engine: Engine, screening_id: int) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(sa.select(screenings).where(screenings.c.id == screening_id)).first()
        if row is None:
            return None
        cand_rows = conn.execute(
            sa.select(candidates).where(candidates.c.screening_id == screening_id)
            .order_by(candidates.c.rank, candidates.c.filename)
        ).all()
    result = _screening_row(row)
    result["job_description"] = row.job_description
    result["settings"] = row.settings or {}
    result["candidates"] = [candidate_row(r) for r in cand_rows]
    return result


def delete_screening(engine: Engine, screening_id: int) -> bool:
    with engine.begin() as conn:
        conn.execute(candidates.delete().where(candidates.c.screening_id == screening_id))
        return conn.execute(screenings.delete().where(screenings.c.id == screening_id)).rowcount > 0


def update_candidate(engine: Engine, screening_id: int, candidate_id: int, changes: dict) -> dict | None:
    values = {key: changes[key] for key in ("shortlisted", "notes") if key in changes}
    with engine.begin() as conn:
        if values:
            values["updated_at"] = utcnow()
            conn.execute(
                candidates.update()
                .where(candidates.c.id == candidate_id, candidates.c.screening_id == screening_id)
                .values(**values)
            )
        row = conn.execute(
            sa.select(candidates).where(candidates.c.id == candidate_id, candidates.c.screening_id == screening_id)
        ).first()
    return candidate_row(row) if row else None


def stats(engine: Engine, strong_threshold: float = 70.0) -> dict:
    with engine.connect() as conn:
        screening_count = conn.execute(sa.select(sa.func.count()).select_from(screenings)).scalar_one()
        candidate_count = conn.execute(sa.select(sa.func.count()).select_from(candidates)).scalar_one()
        average = conn.execute(sa.select(sa.func.avg(candidates.c.final_score))).scalar_one()
        strong = conn.execute(
            sa.select(sa.func.count()).select_from(candidates).where(candidates.c.final_score >= strong_threshold)
        ).scalar_one()
        shortlisted = conn.execute(
            sa.select(sa.func.count()).select_from(candidates).where(candidates.c.shortlisted.is_(True))
        ).scalar_one()
    return {
        "screenings": int(screening_count),
        "candidates": int(candidate_count),
        "average_score": round(float(average or 0), 1),
        "high_matches": int(strong),
        "shortlisted": int(shortlisted),
    }


def purge_older_than(engine: Engine, days: int, now: datetime | None = None, dry_run: bool = False) -> int:
    """Delete screenings (and their candidates) created more than ``days`` days ago."""
    cutoff = (now or utcnow()) - timedelta(days=days)
    with engine.begin() as conn:
        ids = [r[0] for r in conn.execute(sa.select(screenings.c.id).where(screenings.c.created_at < cutoff)).all()]
        if ids and not dry_run:
            conn.execute(candidates.delete().where(candidates.c.screening_id.in_(ids)))
            conn.execute(screenings.delete().where(screenings.c.id.in_(ids)))
    return len(ids)


def legacy_ids(engine: Engine) -> set[int]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(sa.select(screenings.c.legacy_id).where(screenings.c.legacy_id.is_not(None)))}
