"""Screening pipeline shared by uploads, bundled samples and the JSON API.

Two engines are available (``SCORING_ENGINE``): ``v3`` (default, evidence-based hybrid) and ``v2``
(the original formula, kept for comparison and regression). Both produce the same response shape;
v3 adds fields inside each candidate's ``breakdown`` (see docs/api.md).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from scoring import SCORING_VERSION, ScoringConfig, explanation_text, rank_candidates, required_skills
from scoring.engine_v3 import COMPONENT_LABELS, ConfigV3, ResumeInput, rank_candidates_v3
from scoring.skills import load_taxonomy

from .repository import iso, utcnow


@dataclass
class ScreeningDoc:
    name: str
    text: str
    warnings: list[str] = field(default_factory=list)
    page_at: Callable[[int], int | None] | None = None
    quality: dict | None = None
    extraction_score: float | None = None


def display_name(filename: str) -> str:
    """Readable candidate label derived from the file name (``jane_doe-cv.pdf`` → ``jane doe cv``)."""
    stem = filename
    for ext in (".pdf", ".docx"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return " ".join(stem.replace("_", " ").replace("-", " ").split()) or filename


def explanation_v3(b: dict) -> str:
    parts = [f"{b['label']}. Fit {b['final_score']:g}/100 from"]
    used = [f"{c['label'].lower()} {c['value']:g} (×{c['weight'] * 100:.0f}%)" for c in b["components"].values() if c["available"]]
    parts.append(", ".join(used) + ".")
    mandatory = b["mandatory"]
    if mandatory["total"]:
        parts.append(f"Mandatory requirements covered: {mandatory['covered']} of {mandatory['total']}.")
    if mandatory["missing"]:
        parts.append("Not demonstrated: " + ", ".join(mandatory["missing"]) + ".")
    parts.append(f"Evaluation confidence: {b['confidence']['level']}.")
    return " ".join(parts)


def candidate_payload(item: dict, extra_warnings: list[str] | None = None) -> dict:
    """API candidate object: v1.x keys at the top level plus the full ``breakdown``."""
    breakdown = {k: v for k, v in item.items() if k not in {"name"}}
    if extra_warnings:
        breakdown["warnings"] = list(extra_warnings) + [w for w in breakdown["warnings"] if w not in extra_warnings]
    return {
        "id": None,
        "rank": item["rank"],
        "filename": item["name"],
        "display_name": display_name(item["name"]),
        "score": item["final_score"],
        "matched_terms": item["matched_skills"],
        "missing_terms": item["missing_skills"],
        "explanation": explanation_v3(item) if item.get("engine") == "v3" else explanation_text(item),
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
        "mandatory_unmet_count": sum(1 for c in candidates if (c["breakdown"].get("mandatory") or {}).get("met") is False),
        "low_confidence_count": sum(1 for c in candidates if (c["breakdown"].get("confidence") or {}).get("level") == "low"),
        "top_missing_skills": [
            {"skill": skill, "missing_count": n}
            for skill, n in sorted(missing.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        ],
    }


def _as_doc(item) -> ScreeningDoc:
    if isinstance(item, ScreeningDoc):
        return item
    name, text, warnings = item
    return ScreeningDoc(name, text, list(warnings or []))


def run_screening(job_title: str, job_description: str, must_have: list[str], documents, config: ScoringConfig,
                  engine: str = "v2", v3_config: ConfigV3 | None = None, encoder="auto") -> dict:
    """Score documents and build the response body (unsaved). ``documents`` are ScreeningDoc or (name, text, warnings)."""
    docs = [_as_doc(d) for d in documents]
    base = {
        "screening_id": None, "saved": False, "job_title": job_title, "job_description": job_description,
        "created_at": iso(utcnow()), "file_errors": [],
    }
    if engine == "v3":
        v3_config = v3_config or ConfigV3.from_env()
        inputs = [ResumeInput(d.name, d.text, d.page_at, d.extraction_score, d.quality, d.warnings) for d in docs]
        ranked, meta = rank_candidates_v3(job_description, inputs, must_have, v3_config, encoder=encoder)
        payload = [candidate_payload(item) for item in ranked]
        total = sum(v3_config.weights.values()) or 1.0
        return base | {
            "candidate_count": len(payload),
            "must_have": [r["label"] for r in meta["requirements"] if r["origin"] == "recruiter_must_have"],
            "required_skills": [r["label"] for r in meta["requirements"]],
            "requirements": meta["requirements"],
            "scoring_version": meta["scoring_version"],
            "engine": "v3",
            "weights": {k: round(v / total, 4) for k, v in v3_config.weights.items()},
            "component_labels": COMPONENT_LABELS,
            "mandatory_policy": v3_config.mandatory_policy,
            "semantic": meta["semantic"],
            "summary": summarise(payload),
            "candidates": payload,
        }
    taxonomy = load_taxonomy()
    warnings_by_name = {d.name: d.warnings for d in docs}
    ranked = rank_candidates(job_description, [(d.name, d.text) for d in docs], must_have=must_have, config=config, taxonomy=taxonomy)
    payload = [candidate_payload(item, warnings_by_name.get(item["name"])) for item in ranked]
    required = required_skills(job_description, must_have, taxonomy)
    return base | {
        "candidate_count": len(payload),
        "must_have": list(required.must_have),
        "required_skills": list(required.skills),
        "scoring_version": SCORING_VERSION,
        "engine": "v2",
        "weights": config.normalised_weights(bool(required.skills)),
        "summary": summarise(payload),
        "candidates": payload,
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
