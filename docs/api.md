# TalentLens API (v1)

Base path: `/api/v1`. All responses are JSON unless stated. Responses under
`/api/` are sent with `Cache-Control: no-store`.

The examples below were captured from a local run of v2.0.0. Long arrays
are shortened with `…`.

## Conventions

### Errors

Every `/api/v1` error has the same shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Enter a job description of 20 to 20,000 characters; Resumes must contain at least 1 item(s).",
    "details": [
      {"field": "$.job_description", "message": "must be at least 20 characters"},
      {"field": "$.resumes", "message": "must contain at least 1 item(s)"}
    ]
  }
}
```

| HTTP | `code` | When |
| --- | --- | --- |
| 400 | `validation_error` | Input failed schema validation; `details` lists each field |
| 400 | `invalid_json` | Body is not JSON (JSON endpoints) |
| 400 | `invalid_upload` | No files, or too many files |
| 400 | `csrf_failed` | Missing or invalid CSRF token (see below) |
| 401 | `unauthorized` | Owner-only endpoint without an owner session, or a bad bearer token |
| 404 | `not_found` | Unknown screening, candidate or route |
| 413 | `payload_too_large` | Request body over the total upload cap (default 30 MB + 2 MB form overhead) |
| 422 | `no_readable_resumes` | Every uploaded file was rejected; `details` holds per-file errors |
| 429 | `rate_limited` | Too many requests; see the `Retry-After` header |
| 500 | `internal_error` | Unexpected error (logged server-side, no internals in the response) |
| 503 | `database_unavailable` | `/readyz` could not reach the database |

### Authentication and CSRF

* **Public endpoints** (screening, scoring, samples, config, skills, health)
  need no login and store nothing.
* **Owner endpoints** need the session cookie set by `POST /login`.
* **CSRF:** every `POST`/`PATCH`/`DELETE` made with an owner session must send
  the session's token in the header `X-CSRF-Token` (or the form field
  `csrf_token`). Pages expose it in `<meta name="csrf-token">`. Anonymous calls to
  the stateless screening endpoints do not need a token, because they carry
  no cookie authority and write nothing. That is why `curl` works without
  one.
* **Purge** uses `Authorization: Bearer $PURGE_TOKEN` instead of a cookie.

### Rate limits (in memory, per client IP, per process)

| Scope | Default | Env vars |
| --- | --- | --- |
| Screening and scoring endpoints | 30 requests / 10 min | `SCREEN_RATE_LIMIT`, `SCREEN_RATE_WINDOW` |
| Failed owner logins | 5 / 15 min | `LOGIN_RATE_LIMIT`, `LOGIN_RATE_WINDOW` |
| Purge endpoint | 10 / hour | — |

### The candidate object

```json
{
  "id": null,
  "rank": 1,
  "filename": "cv.pdf",
  "display_name": "cv",
  "score": 70.2,
  "matched_terms": ["Python", "SQL", "Docker"],
  "missing_terms": [],
  "explanation": "Strong match. Score 70.2/100 = 40% text similarity (25.4) + 45% skill coverage (100) + 15% section signal (100). Matched skills: Python, SQL, Docker.",
  "shortlisted": false,
  "notes": "",
  "breakdown": { "...": "see below" }
}
```

`score`, `matched_terms`, `missing_terms` and `explanation` keep their v1.x
names. `id` is set only when the screening was saved. `breakdown` is documented in
[scoring.md](scoring.md#the-breakdown-object). Here is an example:

```json
"breakdown": {
  "final_score": 67.9,
  "label": "Partial match",
  "tfidf_score": 19.8,
  "skill_coverage": 100.0,
  "section_signal": 100.0,
  "matched_skills": ["Python", "PostgreSQL", "Docker", "REST APIs"],
  "missing_skills": [],
  "must_have_missing": [],
  "skills_in_sections": ["Python", "PostgreSQL", "Docker", "REST APIs"],
  "required_skills": ["Python", "PostgreSQL", "Docker", "REST APIs"],
  "top_terms": ["apis", "docker", "python", "rest", "rest apis"],
  "warnings": [],
  "weights": {"tfidf": 0.4, "skills": 0.45, "sections": 0.15},
  "contributions": {"tfidf": 7.9, "skills": 45.0, "sections": 15.0},
  "capped": false,
  "text_chars": 84,
  "low_text": true,
  "snippets": [
    {"skill": "Python", "prefix_ellipsis": false, "suffix_ellipsis": false,
     "segments": [{"text": "Python", "mark": true}, {"text": ", Flask, ", "mark": false}, {"text": "postgres", "mark": true}]}
  ],
  "scoring_version": "2.0",
  "rank": 1
}
```

---

## Health

### `GET /healthz`
Liveness. Never touches the database, so it stays cheap during cold starts.
```json
{"status": "ok"}
```

### `GET /readyz`
Readiness. Runs `SELECT 1`.
```json
{"status": "ready", "database": "ok"}
```
When the database is unreachable it returns `503` with `{"error": {"code": "database_unavailable", …}}`.

### `GET /api/v1/health`
```json
{"status": "ok", "service": "TalentLens", "version": "2.0.0"}
```

## Metadata

### `GET /api/v1/config`
Public, non-secret settings used by the UI.
```json
{
  "version": "2.0.0",
  "scoring_version": "2.0",
  "weights": {"tfidf": 0.4, "skills": 0.45, "sections": 0.15},
  "must_have_cap": 40.0,
  "low_text_threshold": 200,
  "limits": {"max_file_mb": 5, "max_files": 20, "max_total_mb": 30, "max_pages": 30},
  "retention_days": 30,
  "owner_login_enabled": true,
  "signed_in": false,
  "config_warnings": []
}
```

### `GET /api/v1/skills`
The skills taxonomy (cached for 1 hour).
```json
{"count": 207, "skills": [{"canonical": "Python", "category": "language", "aliases": ["python3", "python 3"]}, "…"]}
```

### `GET /api/v1/samples`
Metadata for the bundled **synthetic** sample resumes and sample job description.
```json
{
  "label": "Synthetic demo data: fictional people and a sample job description generated for TalentLens.",
  "job_title": "Junior Machine Learning Engineer (sample)",
  "job_description": "Junior Machine Learning Engineer - SAMPLE JOB DESCRIPTION (synthetic demo data) …",
  "must_have": ["Python", "SQL"],
  "resumes": [{"filename": "sample-01-aarav-placeholder.pdf", "description": "Strong match: clear Skills, Experience and Projects sections."}, "…"]
}
```

## Screening

### `POST /api/v1/screen` (alias: `POST /api/screen`)
`multipart/form-data`:

| Field | Required | Notes |
| --- | --- | --- |
| `job_description` | yes | 20–20,000 characters |
| `job_title` | no | ≤ 120 characters (default "Untitled role") |
| `must_have` | no | repeat the field or comma-separate; ≤ 20 items, each ≤ 60 characters |
| `resumes` | yes* | repeat once per PDF; `.pdf`, `%PDF-` header, ≤ 5 MB each, ≤ 20 files, ≤ 30 MB total |
| `use_samples` | no | `1` = use the bundled synthetic resumes instead of `resumes` |
| `save` | no | owner only: `0` to skip saving (default saves) |

```bash
curl -s -X POST http://127.0.0.1:5000/api/v1/screen \
  -F job_title="Data engineer" \
  -F job_description="Data engineer with Python, SQL and Docker." \
  -F must_have=Python \
  -F resumes=@cv1.pdf -F resumes=@cv2.pdf
```

The response is `200` (not saved) or `201` (saved for the owner):

```json
{
  "screening_id": null,
  "saved": false,
  "job_title": "Backend developer",
  "job_description": "Backend developer with Python, PostgreSQL, Docker and REST APIs.",
  "created_at": "2026-11-19T06:59:42+00:00",
  "candidate_count": 1,
  "must_have": ["Python"],
  "required_skills": ["Python", "PostgreSQL", "Docker", "REST APIs"],
  "scoring_version": "2.0",
  "weights": {"tfidf": 0.4, "skills": 0.45, "sections": 0.15},
  "summary": {"candidate_count": 1, "average_score": 67.9, "top_score": 67.9, "low_text_count": 1, "capped_count": 0, "top_missing_skills": []},
  "candidates": ["…candidate objects…"],
  "file_errors": [{"filename": "bad.pdf", "error": "bad.pdf does not look like a real PDF (it is missing the %PDF header). Re-export it as PDF."}],
  "sample_data": false
}
```

Files that fail validation are listed in `file_errors` and skipped. If none can be read,
the response is `422 no_readable_resumes`.

### `POST /api/v1/score`
JSON scoring of plain text (no PDFs). This endpoint is never saved.

```bash
curl -s -X POST http://127.0.0.1:5000/api/v1/score -H 'Content-Type: application/json' -d '{
  "job_title": "Backend developer",
  "job_description": "Backend developer with Python, PostgreSQL, Docker and REST APIs.",
  "must_have": ["Python"],
  "resumes": [{"name": "alex.txt", "text": "Skills\nPython, Flask, postgres, Docker\nExperience\nBuilt REST APIs for a booking app."}]
}'
```

Schema: `job_description` (string, 20–20,000), `resumes` (1–20 items of
`{name: 1–120 chars, unique; text: ≤ 100,000 chars}`), and the optional
`job_title` and `must_have`. Unknown fields are rejected. The response has the same shape as `/screen`.

### `POST /api/v1/samples/screen`
Ranks the bundled synthetic resumes. The JSON body is optional:
`{"job_description": "...", "must_have": ["..."], "job_title": "..."}`.
By default the sample job description is used. This endpoint is never saved.

## Owner history (sign-in required)

### `GET /api/v1/screenings?q=&limit=` (alias: `GET /api/screenings`, which returns a bare list)
`q` searches the role title and job description, case-insensitive. `limit` defaults to 100, maximum 500.
```json
{
  "count": 1,
  "screenings": [
    {"id": 1, "job_title": "Data engineer", "candidate_count": 1, "created_at": "2026-11-19T06:59:42+00:00",
     "must_have": [], "scoring_version": "2.0", "top_score": 70.2, "average_score": 70.2, "shortlisted_count": 0}
  ]
}
```

### `GET /api/v1/screenings/<id>` (alias: `GET /api/screenings/<id>`)
Same shape as the screen response, plus `id`, `settings` and the saved `candidates`, with ids, notes and shortlist state.

### `DELETE /api/v1/screenings/<id>`
```json
{"deleted": true, "screening_id": 1}
```

### `PATCH /api/v1/screenings/<id>/candidates/<candidate_id>`
Body: `{"shortlisted": true, "notes": "Strong SQL"}`. Both fields are optional, and `notes` can be up to 5,000 characters. The response is the updated candidate:
```json
{"id": 1, "screening_id": 1, "rank": 1, "filename": "cv.pdf", "score": 70.2, "matched_terms": ["Python", "SQL", "Docker"],
 "missing_terms": [], "explanation": "…", "shortlisted": true, "notes": "Strong SQL", "breakdown": {"…": "…"}}
```

### `GET /api/v1/screenings/<id>/export.csv` (alias: `GET /api/export.csv?screening_id=<id>`)
UTF-8 CSV with a BOM. The first five columns are unchanged from v1.x:

`Candidate PDF, Text similarity (%), Matched terms, Terms not found, Explanation, Rank, Final score, Label, Skill coverage (%), Section signal (%), Must-have missing, Shortlisted, Notes, Warnings`

"Text similarity (%)" is the TF-IDF component, which is what v1.x exported. "Matched terms" now
lists matched *skills*. Cells that start with `= + - @` are prefixed with `'`, to
block spreadsheet formula injection.

### `GET /api/v1/stats` (alias: `GET /api/stats`)
```json
{"screenings": 1, "candidates": 1, "average_score": 70.2, "high_matches": 1, "shortlisted": 1}
```
`high_matches` counts candidates with a score of 70 or more ("Strong match").

## Retention

### `POST /api/v1/admin/purge`
Header: `Authorization: Bearer $PURGE_TOKEN`. The endpoint is disabled (`404`) unless `PURGE_TOKEN`
is set, and the token must be at least 24 characters. It deletes screenings older than `RETENTION_DAYS`.
```json
{"deleted": 0, "retention_days": 30}
```
It is called daily by `.github/workflows/retention.yml` when the repository secrets are set.
The same job is available as the CLI command `flask purge-old [--days N] [--dry-run]`.

## Legacy (v1.x) aliases

| v1.x route | Now | Change |
| --- | --- | --- |
| `GET /api/health` | same response `{"status":"ok","service":"AI Resume Ranker"}` | none |
| `POST /api/screen` | `/api/v1/screen` | saves only for the signed-in owner; `score` is now the v2 final score |
| `GET /api/stats` | `/api/v1/stats` | **owner only** |
| `GET /api/screenings` | `/api/v1/screenings` | **owner only**; still a bare list |
| `GET /api/screenings/<id>` | `/api/v1/screenings/<id>` | **owner only** |
| `GET /api/export.csv?screening_id=` | `/api/v1/screenings/<id>/export.csv` | **owner only**; extra columns appended |

Legacy routes keep the v1.x error shape `{"error": "message"}`, with extra
`code` and `details` keys.

---

## v2.2 additions (scoring engine v3)

All earlier fields and endpoints are unchanged. The screening responses (`/screen`, `/score`, `/samples/screen`) add
top-level fields, and each candidate's `breakdown` gains v3 fields. With `SCORING_ENGINE=v2` the responses look like v2.1.

### New top-level fields
```json
{
  "engine": "v3",
  "scoring_version": "3.0",
  "weights": {"required": 0.3, "semantic": 0.25, "experience": 0.15, "projects": 0.15, "preferred": 0.1, "lexical": 0.05},
  "mandatory_policy": "flag",
  "requirements": [{"id": "r2", "label": "Python", "cls": "mandatory", "skills": ["Python"], "source": {"text": "- Python and SQL", "start": 14, "end": 30}, "origin": "job_description"}],
  "semantic": {"used": false, "note": null, "enabled": false, "loaded": false, "model": null, "error": null},
  "file_errors": [{"filename": "scan.pdf", "code": "needs_ocr", "error": "scan.pdf: no readable text (probably scanned). … It was not scored."}]
}
```
`file_errors[].code` is `needs_ocr` or `unreadable` for files the document pipeline rejects. Validation errors, as before, have no code.

### Optional `weights` override
`/screen` takes it as a form field holding a JSON string. `/score` and `/samples/screen` take it as a JSON object.

* **Keys:** a subset of `required, semantic, experience, projects, preferred, lexical`.
* **Values:** numbers from 0 to 10, re-normalised.
* **Response:** echoed back as `weights_override`.
* **Errors:** unknown keys return `400 validation_error`.

### v3 candidate `breakdown` (real output, shortened)
```json
{
  "engine": "v3",
  "fit_score": 56.8,
  "evidence_fit": 56.8,
  "final_score": 56.8,
  "label": "Partial match",
  "components": {
    "required":  {"label": "Required-skill evidence", "value": 65.0, "weight": 0.5,    "contribution": 32.5, "available": true,  "note": ""},
    "preferred": {"label": "Preferred skills",        "value": 60.0, "weight": 0.1667, "contribution": 10.0, "available": true,  "note": ""},
    "projects":  {"label": "Project relevance",       "value": null, "weight": 0.0,    "contribution": 0.0,  "available": false,
                  "note": "No Projects section; weight redistributed because Experience is present."}
  },
  "mandatory": {"total": 4, "covered": 3, "coverage": 75.0, "missing": ["Docker"], "met": false, "policy": "flag"},
  "confidence": {"score": 0.61, "level": "moderate", "factors": {"extraction": 0.4, "structure": 1.0, "clarity": 0.6, "semantic": 0.7},
                 "reasons": ["…"], "calibrated": false},
  "requirements": [
    {"id": "r2", "label": "Python", "cls": "mandatory", "status": "supported", "strength": 1.0,
     "evidence": [{"skill": "Python", "quote": "Developed REST APIs using Python and Flask for a booking app",
                   "start": 42, "end": 102, "match_start": 68, "match_end": 74, "surface": "Python", "match_kind": "exact",
                   "section": "experience", "page": null, "type": "demonstrated", "strength": 1.0}],
     "related_found": [], "inferred_from": [], "explanation": "The Experience section describes work using Python."},
    {"id": "r1", "label": "Docker", "cls": "mandatory", "status": "negated", "strength": 0.0,
     "explanation": "The resume states that this experience is absent."}
  ],
  "semantic": {"used": false, "matches": []},
  "profile": {"contact": {"name": null, "email": null, "phone": null, "location": null, "used_for_scoring": false}},
  "injection_suspected": false
}
```
Notes on these fields:

* **Quotes:** `quote == text[start:end]` of the extracted text. `page` is set for PDFs.
* **Statuses:** `supported`, `unclear`, `inferred`, `negated` or `not_found`. `not_found` means "not demonstrated in the resume".
* **v2-compatible keys remain, with these meanings:**
  * `score` and `final_score` are the fit score.
  * `skill_coverage` is the % of mandatory and general requirements covered.
  * `matched_skills` and `missing_skills` hold requirement labels.
  * `must_have_missing` lists the missing mandatory requirements.
  * `capped` is true only with `MANDATORY_POLICY=cap`.
* **Ordering:** candidates who meet every mandatory requirement are ranked first, then by fit.

### `POST /api/v1/resume/analyze`
Send a multipart form with exactly one `resume` (PDF or DOCX), plus optional `job_description` and `must_have`. The endpoint is
stateless and nothing is stored. It needs no CSRF token when anonymous, and it shares the screening rate limit.

```json
{
  "filename": "cv.pdf",
  "stored": false,
  "quality": {"grade": "excellent", "score": 1.0, "reasons": ["Clean, complete text extraction."], "pages": 1, "evaluable": true},
  "profile": {"experience": {"status": "found", "items": [{"title": "Backend Developer", "company": "Example Co"}]}},
  "requirements": ["… same objects as above, only when job_description is given …"],
  "improvements": {
    "findings": [{"type": "weak_verb", "severity": "medium", "message": "“Worked on” hides what you actually did.",
                  "original": {"text": "Worked on a website using React.", "start": 120, "end": 152},
                  "suggestion": "[Built/Implemented/…] a website using React — [what it does], [your specific contribution]. Choose the verb that is true.",
                  "is_recommendation": true}],
    "total_findings": 6,
    "disclaimer": "Suggestions are recommendations generated from rules. They are not part of your resume. …"
  }
}
```

### `GET /api/v1/config` additions
New fields:

* `engine`, plus `v3.weights`, `v3.mandatory_policy` and `v3.semantic`.
* `accepted_formats`, which is `["pdf","docx"]`.
* `ocr_available`.
* `llm_enabled`, which is always `false` because no LLM provider ships.

### CSV
These columns are appended after the existing ones: `Engine`, `Mandatory coverage (%)`, `Mandatory missing`, `Evaluation confidence`.
