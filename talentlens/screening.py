"""Screening pipeline shared by uploads, bundled samples and the JSON API."""

from __future__ import annotations

from collections import Counter

from scoring import SCORING_VERSION, ScoringConfig, explanation_text, rank_candidates, required_skills
from scoring.skills import load_taxonomy

from .repository import iso, utcnow


def display_name(filename: str) -> str:
    """Readable candidate label derived from the file name (``jane_doe-cv.pdf`` → ``jane doe cv``)."""
    stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
    return " ".join(stem.replace("_", " ").replace("-", " ").split()) or filename


def candidate_payload(item: dict, extra_warnings: list[str] | None = None) -> dict:
    """API candidate object: v1.x keys at the top level plus the full v2 ``breakdown``."""
    breakdown = {k: v for k, v in item.items() if k not in {"name"}}
    if extra_warnings:
        breakdown["warnings"] = list(extra_warnings) + breakdown["warnings"]
    return {
        "id": None,
        "rank": item["rank"],
        "filename": item["name"],
        "display_name": display_name(item["name"]),
        "score": item["final_score"],
        "matched_terms": item["matched_skills"],
        "missing_terms": item["missing_skills"],
        "explanation": explanation_text(item),
        "shortlisted": False,
        "notes": "",
        "breakdown": breakdown,
    }


def summarise(candidates: list[dict], top_n: int = 8) -> dict:
    """Pool-level summary: count, average score and the skills most often missing."""
    count = len(candidates)
    missing = Counter(skill for c in candidates for skill in c["breakdown"].get("missing_skills", []))
    return {
        "candidate_count": count,
        "average_score": round(sum(c["score"] for c in candidates) / count, 1) if count else 0.0,
        "top_score": max((c["score"] for c in candidates), default=0.0),
        "low_text_count": sum(1 for c in candidates if c["breakdown"].get("low_text")),
        "capped_count": sum(1 for c in candidates if c["breakdown"].get("capped")),
        "top_missing_skills": [
            {"skill": skill, "missing_count": n}
            for skill, n in sorted(missing.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        ],
    }


def run_screening(job_title: str, job_description: str, must_have: list[str],
                  documents: list[tuple[str, str, list[str]]], config: ScoringConfig) -> dict:
    """Score ``(filename, text, extraction_warnings)`` documents and build the response body (unsaved)."""
    taxonomy = load_taxonomy()
    warnings_by_name = {name: warns for name, _, warns in documents}
    ranked = rank_candidates(job_description, [(name, text) for name, text, _ in documents],
                             must_have=must_have, config=config, taxonomy=taxonomy)
    payload = [candidate_payload(item, warnings_by_name.get(item["name"])) for item in ranked]
    required = required_skills(job_description, must_have, taxonomy)
    return {
        "screening_id": None,
        "saved": False,
        "job_title": job_title,
        "job_description": job_description,
        "created_at": iso(utcnow()),
        "candidate_count": len(payload),
        "must_have": list(required.must_have),
        "required_skills": list(required.skills),
        "scoring_version": SCORING_VERSION,
        "weights": config.normalised_weights(bool(required.skills)),
        "summary": summarise(payload),
        "candidates": payload,
        "file_errors": [],
    }


def parse_must_have(values: list[str]) -> list[str]:
    """Accept repeated form fields and/or comma-separated values; trim and de-duplicate."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            name = " ".join(part.split())
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
    return out
