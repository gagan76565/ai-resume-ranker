"""JSON API: ``/api/v1/*`` plus the v1.x ``/api/*`` aliases.

Every endpoint is documented with examples in ``docs/api.md``.
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import time

from flask import Blueprint, current_app, jsonify, make_response, request

from scoring import SCORING_VERSION
from scoring import semantic as semantic_mod
from scoring.engine_v3 import ConfigV3, ResumeInput, rank_candidates_v3
from scoring.improve import analyze_resume
from scoring.profile import extract_profile
from scoring.skills import load_taxonomy

from . import repository, samples
from .auth import is_owner, owner_required
from .documents import DocumentError, ocr_enabled, parse_document
from .errors import ApiError
from .screening import ScreeningDoc, parse_must_have, run_screening
from .security import csrf_anonymous_ok, csrf_exempt, rate_limited
from .uploads import validate_uploads
from .validation import CANDIDATE_PATCH, SCORE_REQUEST, SCREEN_FORM, WEIGHTS, validate

api = Blueprint("api", __name__)

VERSION = "2.2.0"

FIELD_LABELS = {
    "$.job_description": "Job description",
    "$.job_title": "Role title",
    "$.must_have": "Must-have skills",
    "$.resumes": "Resumes",
}


def _state():
    return current_app.extensions["talentlens"]


def _engine():
    return _state().engine


def _settings():
    return _state().settings


def route(rule: str, legacy: str | None = None, **options):
    """Register a view under ``/api/v1{rule}`` and optionally a legacy ``/api{legacy}`` alias."""

    def decorator(view):
        api.add_url_rule("/api/v1" + rule, view_func=view, **options)
        if legacy:
            api.add_url_rule("/api" + legacy, endpoint=view.__name__ + "_legacy", view_func=view, **options)
        return view

    return decorator


def _validation_error(problems: list[dict]):
    readable = []
    for problem in problems:
        field = problem["field"]
        if field == "$.job_description":
            text = "Enter a job description of 20 to 20,000 characters"
        else:
            label = next((v for k, v in FIELD_LABELS.items() if field == k or field.startswith((k + "[", k + "."))), field)
            text = f"{label} {problem['message']}"
        if text not in readable:
            readable.append(text)
    message = "; ".join(readable[:3]) + "."
    raise ApiError(400, "validation_error", message, problems)


# --------------------------------------------------------------------------- health & metadata

@route("/health", legacy="/health", methods=["GET"])
def health():
    if request.path == "/api/health":
        return jsonify({"status": "ok", "service": "AI Resume Ranker"})  # v1.x shape
    return jsonify({"status": "ok", "service": "TalentLens", "version": VERSION})


@route("/config", methods=["GET"])
def public_config():
    settings = _settings()
    limits = settings.upload_limits
    return jsonify({
        "version": VERSION,
        "scoring_version": SCORING_VERSION,
        "weights": settings.scoring.normalised_weights(),
        "must_have_cap": settings.scoring.must_have_cap,
        "low_text_threshold": settings.scoring.low_text_threshold,
        "limits": {
            "max_file_mb": limits.max_file_bytes // (1024 * 1024),
            "max_files": limits.max_files,
            "max_total_mb": limits.max_total_bytes // (1024 * 1024),
            "max_pages": limits.max_pages,
        },
        "retention_days": settings.retention_days,
        "owner_login_enabled": settings.owner_login_enabled,
        "signed_in": is_owner(),
        "config_warnings": [p for p in settings.problems if p.startswith(("SCORE_", "V3_", "MANDATORY", "SCORING_ENGINE")) or "weights" in p],
        "engine": settings.scoring_engine,
        "v3": {
            "weights": {k: round(v / (sum(settings.scoring_v3.weights.values()) or 1), 4) for k, v in settings.scoring_v3.weights.items()},
            "mandatory_policy": settings.scoring_v3.mandatory_policy,
            "semantic": semantic_mod.status(),
        },
        "accepted_formats": ["pdf", "docx"],
        "ocr_available": ocr_enabled(),
        "llm_enabled": False,
    })


@route("/skills", methods=["GET"])
def skills():
    taxonomy = load_taxonomy()
    items = [{"canonical": s.canonical, "category": s.category, "aliases": list(s.aliases) + list(s.case_sensitive)}
             for s in taxonomy.skills.values()]
    response = jsonify({"count": len(items), "skills": items})
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@route("/samples", methods=["GET"])
def sample_info():
    return jsonify(samples.public_manifest())


# --------------------------------------------------------------------------- screening

def _parse(name: str, data: bytes, errors: list) -> ScreeningDoc | None:
    """Parse one file. Unreadable or scanned (no OCR) files are reported in ``errors`` and never scored."""
    settings = _settings()
    try:
        doc = parse_document(name, data, settings.upload_limits, settings.scoring.low_text_threshold)
    except DocumentError as exc:
        errors.append({"filename": name, "error": f"{name}: {exc}", "code": "unreadable"})
        return None
    if not doc.quality.evaluable:
        errors.append({"filename": name, "code": "needs_ocr",
                       "error": f"{name}: no readable text (probably scanned). {' '.join(doc.quality.reasons[1:]) or 'OCR is not available.'} "
                                "It was not scored."})
        return None
    return ScreeningDoc(name, doc.text, list(doc.warnings), doc.page_at, doc.quality.to_dict(), doc.quality.score)


def _documents_from_uploads():
    validation = validate_uploads(request.files.getlist("resumes"), _settings().upload_limits)
    if validation.batch_error:
        raise ApiError(400, "invalid_upload", validation.batch_error)
    errors = list(validation.errors)
    documents = [d for d in (_parse(a.name, a.data, errors) for a in validation.accepted) if d]
    return documents, errors


def _documents_from_samples():
    errors: list = []
    return [d for d in (_parse(name, data, errors) for name, data in samples.sample_files()) if d], errors


def _weights_override(raw) -> dict | None:
    if raw in (None, "", {}):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ApiError(400, "validation_error", "weights must be a JSON object.", [{"field": "$.weights", "message": "invalid JSON"}]) from None
    problems = validate(raw, WEIGHTS)
    if problems:
        _validation_error([{**p, "field": p["field"].replace("$", "$.weights", 1)} for p in problems])
    return raw


def _run(job_title, job_description, must_have, documents, weights=None) -> dict:
    settings = _settings()
    v3_config = ConfigV3.from_env(overrides=weights) if weights else settings.scoring_v3
    encoder = semantic_mod.get_encoder(dict(settings.semantic_env)) if settings.scoring_engine == "v3" else None
    result = run_screening(job_title, job_description, must_have, documents, settings.scoring,
                           engine=settings.scoring_engine, v3_config=v3_config, encoder=encoder)
    if weights:
        result["weights_override"] = weights
    return result


def _maybe_save(result: dict, save_requested: bool) -> dict:
    if not (save_requested and is_owner()):
        return result
    settings = _settings()
    screening_id = repository.create_screening(
        _engine(),
        job_title=result["job_title"],
        job_description=result["job_description"],
        must_have=result["must_have"],
        settings={"weights": result["weights"], "must_have_cap": settings.scoring.must_have_cap,
                  "file_errors": result["file_errors"]},
        scoring_version=result["scoring_version"],
        ranked=result["candidates"],
    )
    saved = repository.get_screening(_engine(), screening_id)
    result.update({"screening_id": screening_id, "saved": True, "created_at": saved["created_at"]})
    by_rank = {c["rank"]: c for c in saved["candidates"]}
    for candidate in result["candidates"]:
        candidate["id"] = by_rank[candidate["rank"]]["id"]
    return result


@route("/screen", legacy="/screen", methods=["POST"])
@csrf_anonymous_ok
@rate_limited("screen")
def screen():
    """Multipart upload (or ``use_samples=1``) → ranked candidates. Saves only for the signed-in owner."""
    form = request.form
    job_title = (form.get("job_title") or "").strip()
    job_description = (form.get("job_description") or "").strip()
    must_have = parse_must_have(form.getlist("must_have"))
    problems = validate({"job_title": job_title, "job_description": job_description, "must_have": must_have}, SCREEN_FORM)
    if problems:
        _validation_error(problems)
    weights = _weights_override(form.get("weights"))
    use_samples = form.get("use_samples") == "1"
    documents, errors = _documents_from_samples() if use_samples else _documents_from_uploads()
    if not documents:
        raise ApiError(422, "no_readable_resumes", "None of the uploaded files could be read. See the per-file messages.", errors)
    result = _run(job_title or "Untitled role", job_description, must_have, documents, weights)
    result["file_errors"] = errors
    result["sample_data"] = use_samples
    result = _maybe_save(result, form.get("save", "1") != "0" and not use_samples)
    return jsonify(result), 201 if result["saved"] else 200


@route("/samples/screen", methods=["POST"])
@csrf_anonymous_ok
@rate_limited("screen")
def screen_samples():
    """Rank the bundled synthetic resumes against the sample (or a supplied) job description. Never saved."""
    data = samples.manifest()
    body = request.get_json(silent=True) or {}
    job_description = (body.get("job_description") or data["job_description"]).strip()
    must_have = body.get("must_have", data["must_have"])
    problems = validate({"job_description": job_description, "must_have": must_have}, SCREEN_FORM)
    if problems:
        _validation_error(problems)
    documents, errors = _documents_from_samples()
    result = _run(body.get("job_title") or data["job_title"], job_description, parse_must_have(must_have), documents,
                  _weights_override(body.get("weights")))
    result["file_errors"] = errors
    result["sample_data"] = True
    return jsonify(result)


@route("/score", methods=["POST"])
@csrf_anonymous_ok
@rate_limited("screen")
def score_text():
    """Plain-text scoring for API clients (no PDFs). Never saved."""
    body = request.get_json(silent=True)
    if body is None:
        raise ApiError(400, "invalid_json", "Send a JSON body with Content-Type: application/json.")
    problems = validate(body, SCORE_REQUEST)
    if problems:
        _validation_error(problems)
    names = [r["name"] for r in body["resumes"]]
    if len(set(names)) != len(names):
        raise ApiError(400, "validation_error", "Resume names must be unique.", [{"field": "$.resumes", "message": "duplicate name"}])
    documents = [ScreeningDoc(r["name"], r["text"]) for r in body["resumes"]]
    result = _run((body.get("job_title") or "Untitled role").strip(), body["job_description"].strip(),
                  parse_must_have(body.get("must_have", [])), documents, _weights_override(body.get("weights")))
    for candidate in result["candidates"]:  # plain text was supplied, so "scanned PDF" wording does not apply
        b = candidate["breakdown"]
        b["warnings"] = [
            f"Only {b['text_chars']} characters of text were supplied; scores on very short text are unreliable."
            if w.startswith("Very little extractable text") else w
            for w in b["warnings"]
        ]
    return jsonify(result)


@route("/resume/analyze", methods=["POST"])
@csrf_anonymous_ok
@rate_limited("screen")
def analyze_single_resume():
    """One resume (multipart ``resume``) → structured profile, extraction quality and improvement suggestions.

    Optional ``job_description`` (+ ``must_have``) adds requirement evidence and job-specific gaps. Never stored.
    """
    files = request.files.getlist("resume")
    if len(files) != 1:
        raise ApiError(400, "invalid_upload", "Upload exactly one resume in the 'resume' field.")
    validation = validate_uploads(files, _settings().upload_limits)
    if validation.batch_error or not validation.accepted:
        raise ApiError(422, "no_readable_resumes", "The resume could not be read.", validation.errors)
    accepted = validation.accepted[0]
    errors: list = []
    doc = _parse(accepted.name, accepted.data, errors)
    if doc is None:
        raise ApiError(422, "no_readable_resumes", errors[0]["error"], errors)
    job_description = (request.form.get("job_description") or "").strip()
    requirements = []
    if job_description:
        problems = validate({"job_description": job_description}, SCREEN_FORM)
        if problems:
            _validation_error(problems)
        ranked, _meta = rank_candidates_v3(job_description, [ResumeInput(doc.name, doc.text, doc.page_at, doc.extraction_score, doc.quality)],
                                           parse_must_have(request.form.getlist("must_have")), _settings().scoring_v3,
                                           encoder=semantic_mod.get_encoder(dict(_settings().semantic_env)))
        requirements = ranked[0]["requirements"]
    return jsonify({
        "filename": doc.name,
        "quality": doc.quality,
        "warnings": doc.warnings,
        "profile": extract_profile(doc.text),
        "requirements": requirements,
        "improvements": analyze_resume(doc.text, requirements),
        "stored": False,
    })


# --------------------------------------------------------------------------- owner history

_last_lazy_purge = {"at": 0.0}


def _lazy_purge():
    """Opportunistic retention purge (at most hourly) since Render free has no background jobs."""
    now = time.monotonic()
    if now - _last_lazy_purge["at"] < 3600:
        return
    _last_lazy_purge["at"] = now
    deleted = repository.purge_older_than(_engine(), _settings().retention_days)
    if deleted:
        current_app.logger.info("Retention purge removed %d screening(s)", deleted)


@route("/stats", legacy="/stats", methods=["GET"])
@owner_required
def stats():
    return jsonify(repository.stats(_engine()))


@route("/screenings", legacy="/screenings", methods=["GET"])
@owner_required
def list_screenings():
    _lazy_purge()
    limit = request.args.get("limit", default=100, type=int) or 100
    items = repository.list_screenings(_engine(), request.args.get("q"), limit)
    if request.path.startswith("/api/v1/"):
        return jsonify({"count": len(items), "screenings": items})
    return jsonify(items)  # v1.x returned a bare list


def _get_or_404(screening_id: int) -> dict:
    screening = repository.get_screening(_engine(), screening_id)
    if screening is None:
        raise ApiError(404, "not_found", "Screening not found.")
    return screening


@route("/screenings/<int:screening_id>", legacy="/screenings/<int:screening_id>", methods=["GET"])
@owner_required
def screening_detail(screening_id: int):
    screening = _get_or_404(screening_id)
    from .screening import summarise

    screening["screening_id"] = screening["id"]
    screening["saved"] = True
    screening["summary"] = summarise(screening["candidates"])
    screening["required_skills"] = (screening["candidates"][0]["breakdown"].get("required_skills", [])
                                    if screening["candidates"] else [])
    screening["weights"] = screening["settings"].get("weights")
    screening["file_errors"] = screening["settings"].get("file_errors", [])
    return jsonify(screening)


@route("/screenings/<int:screening_id>", methods=["DELETE"])
@owner_required
def delete_screening(screening_id: int):
    if not repository.delete_screening(_engine(), screening_id):
        raise ApiError(404, "not_found", "Screening not found.")
    return jsonify({"deleted": True, "screening_id": screening_id})


@route("/screenings/<int:screening_id>/candidates/<int:candidate_id>", methods=["PATCH"])
@owner_required
def update_candidate(screening_id: int, candidate_id: int):
    body = request.get_json(silent=True)
    if body is None:
        raise ApiError(400, "invalid_json", "Send a JSON body with Content-Type: application/json.")
    problems = validate(body, CANDIDATE_PATCH)
    if problems:
        _validation_error(problems)
    updated = repository.update_candidate(_engine(), screening_id, candidate_id, body)
    if updated is None:
        raise ApiError(404, "not_found", "Candidate not found in this screening.")
    return jsonify(updated)


CSV_HEADER = [
    # v1.x columns, unchanged order and names
    "Candidate PDF", "Text similarity (%)", "Matched terms", "Terms not found", "Explanation",
    # v2 additions
    "Rank", "Final score", "Label", "Skill coverage (%)", "Section signal (%)", "Must-have missing",
    "Shortlisted", "Notes", "Warnings",
    # v2.2 additions (appended; earlier columns unchanged)
    "Engine", "Mandatory coverage (%)", "Mandatory missing", "Evaluation confidence",
]


def csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection (cells starting with = + - @ tab or CR)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def candidates_csv(candidates: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(CSV_HEADER)
    for c in candidates:
        b = c["breakdown"]
        writer.writerow([csv_safe(v) for v in (
            c["filename"], b.get("tfidf_score", ""), ", ".join(c["matched_terms"]), ", ".join(c["missing_terms"]),
            c["explanation"], c["rank"], c["score"], b.get("label", ""), b.get("skill_coverage", ""),
            b.get("section_signal", ""), ", ".join(b.get("must_have_missing", [])), "yes" if c["shortlisted"] else "no",
            c.get("notes", ""), " | ".join(b.get("warnings", [])),
            b.get("engine", "v2"), (b.get("mandatory") or {}).get("coverage", ""), ", ".join((b.get("mandatory") or {}).get("missing", [])),
            (b.get("confidence") or {}).get("level", ""),
        )])
    return "﻿" + output.getvalue()


def _csv_response(screening_id: int):
    screening = _get_or_404(screening_id)
    response = make_response(candidates_csv(screening["candidates"]))
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="screening-{screening_id}.csv"'
    return response


@route("/screenings/<int:screening_id>/export.csv", methods=["GET"])
@owner_required
def export_csv(screening_id: int):
    return _csv_response(screening_id)


@api.get("/api/export.csv")
@owner_required
def export_csv_legacy():
    screening_id = request.args.get("screening_id", type=int)
    if not screening_id:
        raise ApiError(400, "validation_error", "Provide a screening_id query parameter.")
    return _csv_response(screening_id)


# --------------------------------------------------------------------------- retention (cron)

@route("/admin/purge", methods=["POST"])
@csrf_exempt
@rate_limited("purge")
def purge():
    """Delete screenings older than RETENTION_DAYS. Auth: ``Authorization: Bearer $PURGE_TOKEN``."""
    settings = _settings()
    if not settings.purge_token:
        raise ApiError(404, "not_found", "The purge endpoint is disabled (PURGE_TOKEN is not set).")
    header = request.headers.get("Authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied.encode(), settings.purge_token.encode()):
        raise ApiError(401, "unauthorized", "Missing or invalid bearer token.")
    deleted = repository.purge_older_than(_engine(), settings.retention_days)
    current_app.logger.info("Retention purge via API removed %d screening(s)", deleted)
    return jsonify({"deleted": deleted, "retention_days": settings.retention_days})
