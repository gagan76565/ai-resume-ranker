# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.1.1] - 2026-11-19

### Fixed
- `python app.py` now binds `0.0.0.0` automatically when `RENDER` is set, so a Render service whose start
  command is `python app.py` (instead of the Procfile gunicorn command) no longer fails with "No open ports
  detected". Elsewhere it still defaults to `127.0.0.1`; `HOST` overrides both.

## [2.1.0] - 2026-11-19

UI/UX redesign only. There are no changes to scoring, extraction, the API, routes or privacy
behaviour. Scores for the same input are byte-identical to 2.0.0.

### Changed
- New design system in `static/css/app.css`:
  - vendored Inter variable font (SIL OFL) and a type scale;
  - refined indigo accent and three surface levels per theme;
  - elevation, radius and motion tokens;
  - components: buttons, fields, switch, tabs, tooltips, badges, chips, meters, panels, toasts, skeletons, empty states and modal.
- Shell: brand mark and wordmark, segmented navigation with a clear active state,
  an animated sun/moon theme switch, and a two-row header on mobile.
- Screening page:
  - "Find the right candidate, backed by evidence" introduction;
  - live three-step workflow indicator derived from the form state;
  - two-column workspace, upload capacity meter and richer file rows;
  - integrated privacy panel (same claims as before);
  - sticky action bar that summarises what will be submitted. **Rank candidates** stays disabled
    until a job description (20+ characters) and a valid PDF or the samples are present.
- Results page:
  - header with metadata chips;
  - metric cards with a score-band distribution;
  - skill-gap bars;
  - tabbed requirements accordion (skills, weights, methodology, job description);
  - refined toolbar;
  - re-proportioned table with avatars, score bars with band thresholds, and coverage meters.
    It becomes cards below 1080 px.
- Candidate details are organised into panels: overview ring and formula, component bars, matched and
  missing skills, evidence with `<mark>` highlights, top terms, extracted-text facts, warnings and notes.
- Compare dialog: aligned rows, score deltas, shared vs unique skills, evidence, warnings, and buttons
  to remove a candidate.
- History, login and error pages use the same components. The icon SVG sprite lives in `templates/_icons.html`.
- `scripts/ui_check.py` now checks 375/768/1024/1440 px and the new readiness behaviour (47 checks).

### Fixed
- An earlier stylesheet edit had duplicated utility classes inside `@media print`.

## [2.0.0] - 2026-11-19

### Added
- **Scoring v2** (`scoring/` package): a hybrid score of TF-IDF similarity (0.40),
  skill coverage (0.45) and section signal (0.15), with weights configurable by
  environment variable. It has a 207-skill alias taxonomy, boundary-safe matching,
  must-have skills with a score cap, a per-candidate breakdown and highlighted
  snippets. The formula is documented in `docs/scoring.md`.
- Upload validation: `.pdf` extension, `%PDF-` magic bytes, 5 MB per file,
  20 files, 30 MB total, page and character caps. Errors name the file.
  Low-text (likely scanned) PDFs are scored with a visible warning.
- Public demo mode that stores nothing, plus **synthetic** sample resumes
  and a sample job description.
- Owner login (`ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, `flask hash-password`)
  with login rate limiting, CSRF protection, secure cookies, CSP and security headers.
- Saved history for the owner: notes, shortlist, delete with Undo, search, CSV export.
- PostgreSQL support via `DATABASE_URL`, with SQLite fallback. SQLAlchemy Core and Alembic migrations run automatically at start-up.
- Retention: `RETENTION_DAYS` (default 30), `flask purge-old`, a bearer-token
  purge endpoint and an optional GitHub Actions cron.
- Versioned API `/api/v1/*` with JSON-schema validation and consistent error
  objects; `/healthz`, `/readyz`, `/api/v1/config`, `/api/v1/skills`,
  `/api/v1/score` (plain-text JSON scoring) and `/api/v1/samples`. See `docs/api.md`.
- New UI: light and dark themes, drag-and-drop with file chips, must-have tag
  input, progress state, a sortable and filterable results table, "Why this score?"
  panel, two-way compare, summary bar, CSV export of the current view, print
  layout, empty states, skeleton loaders, toasts, and a cold-start wake-up banner
  with auto-retry.
- Test suite: 141 tests (pytest, also runnable with `python -m unittest`).
  CI runs on Python 3.10 to 3.12 with `ruff`, plus a PostgreSQL job.
- `LICENSE` (MIT), `.env.example`, `docs/`, the `scripts/import_sqlite.py`
  migration tool and the optional `scripts/ui_check.py` browser check.

### Changed
- PDF extraction now uses `pypdf` 7.1.1. `PyPDF2` is deprecated upstream.
- `score` in API responses and the "Score" column are now the v2 final score.
  The v1 TF-IDF value is still available as `breakdown.tfidf_score`, and in
  the CSV column "Text similarity (%)", whose meaning is unchanged.
- `matched_terms` and `missing_terms` now list taxonomy **skills** instead of raw tokens.
- `POST /api/screen` saves only when the owner is signed in. It returns `200` when
  nothing was saved and `201` when it was.
- CSV export keeps the five v1.x columns, in the same order, and appends new columns.
- `python app.py` binds to `127.0.0.1` by default (set `HOST=0.0.0.0` to expose it).
  Gunicorn and the Procfile are unchanged.
- The default SQLite file is now `instance/talentlens.db`.
- The CI workflow was renamed `tests.yml` → `ci.yml`.

### Security / breaking
- **History is no longer public.** `GET /api/stats`, `/api/screenings`,
  `/api/screenings/<id>` and `/api/export.csv` now need the owner session
  and return `401` otherwise. In v1.x any visitor could read every candidate's
  results.
- Legacy routes keep the `{"error": "message"}` shape, with extra `code` and `details` keys.

### Removed
- `templates/result.html`, an unused page that loaded the Tailwind CDN.

### Migration from 1.x
1. Deploy v2. Migrations create the new tables automatically.
2. If you have a v1.x SQLite file, run `flask import-sqlite instance/resume_ranker.db`
   (or `python scripts/import_sqlite.py …`). It sets `DATABASE_URL` to the target. Imported rows are marked
   `scoring_version = "1.x-import"` and are not re-scored, because the PDFs were never stored.
   Re-running the import skips rows that were already imported.

## [1.0.0] - 2026-10-11
- Flask dashboard with TF-IDF cosine similarity, SQLite history, CSV export and the JSON API.
