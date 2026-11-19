# TalentLens v2 — upgrade plan

## Current state (inspected 2026-11-19, commit `eda2e6f` on `main`)

| Area | Today |
| --- | --- |
| Backend | One 290-line `app.py` (Flask 3.1.1). Raw `sqlite3`, tables `screenings` and `candidates`. |
| Scoring | `TfidfVectorizer(stop_words="english", ngram_range=(1,2))` fitted on the JD and one resume, cosine × 100. Keyword overlap is a literal set intersection of tokens. |
| PDF | `PyPDF2==3.0.1` (deprecated upstream; the project moved to `pypdf`). Extension and size checks only; no magic-byte check. |
| Frontend | `templates/index.html`: a single 279-line page with inline CSS and JS. `templates/result.html` is an unused legacy page that loads the Tailwind CDN. |
| API | `/api/health`, `/api/stats`, `/api/screenings`, `/api/screenings/<id>`, `POST /api/screen`, `/api/export.csv`. Errors are `{"error": "message"}`. |
| Auth | None. Every visitor can read every saved screening. |
| Tests | 5 `unittest` tests. All pass locally (Python 3.11.9). |
| CI | GitHub Actions, Python 3.12, `python -m unittest`. |
| Deploy | Render free, `gunicorn app:app --workers 1 --threads 4 --timeout 120`. |
| Repo | No LICENSE, CHANGELOG, `.env.example`, or screenshots. |

Baseline check: `python app.py` serves `/` (HTTP 200) and `/api/health` returns `{"status":"ok"}`.

## Gaps found

1. Literal keyword matching misses synonyms ("JS" vs "JavaScript") and can match substrings badly.
2. The score formula is undocumented, and one number mixes all signals.
3. Privacy: any visitor can list every uploaded candidate's results.
4. SQLite on Render free is wiped on every deploy or restart.
5. Scanned PDFs return "no usable text", but nothing explains why.
6. No CSRF, auth, rate limiting, or secure cookie configuration.
7. Inline monolithic UI: no dark mode, colour-only score cues, and keyboard gaps (clickable `div`s).
8. Old pins: Flask 3.1.1, PyPDF2 (deprecated).

## Milestone plan

| Milestone | Deliverables | Commit prefix |
| --- | --- | --- |
| A Scoring v2 | `scoring/` package (text, skills, sections, tfidf, engine), `skills.json` with 150+ skills, hybrid score, must-have cap, breakdown object, `docs/scoring.md`, unit tests | `feat(scoring)` |
| B Uploads | `talentlens/uploads.py`: extension, magic bytes, 5 MB/file, 20 files, total cap, page cap; low-text warning; `pypdf` pinned | `feat(uploads)` |
| C Accounts/persistence | App factory package `talentlens/`, SQLAlchemy Core + Alembic, `DATABASE_URL` (Postgres) with SQLite fallback, owner login, custom CSRF, login rate limiting, retention + `flask purge-old`, SQLite import script, synthetic samples | `feat(auth)`, `feat(db)` |
| D API/platform | `/api/v1/*` with legacy aliases, small JSON schema validator, uniform errors, `/healthz` and `/readyz`, wake-up splash, pytest + ruff, CI matrix 3.10–3.12, `docs/api.md` | `feat(api)`, `ci:` |
| E UI/UX | `base.html` design system (CSS variables, light/dark), upload/results/compare/history/login/404/500 pages, vanilla JS modules in `static/` | `feat(ui)` |
| F Docs | README rewrite, LICENSE (MIT, I. Gagan), CHANGELOG, `.env.example`, `docs/*`, repo metadata | `docs:` |

## Key design decisions

- **Score is pairwise.** TF-IDF is fitted on (JD, one resume), so a candidate's score never depends on who else is in the batch.
- **Order-invariant n-grams.** Bigrams are built inside a line or sentence only, so reordering sections cannot change the score (bias test).
- **Public demo stores nothing.** `POST /api/v1/screen` only writes to the database when the owner is logged in and asks to save.
- **CSRF.** A session-bound token is checked on every state-changing request that carries the owner session, and on the login form. Anonymous screening is stateless (no cookie authority, no writes), so API clients can call it with curl.
- **Migrations.** Alembic runs automatically at start-up (`AUTO_MIGRATE=1`), because Render free has no pre-deploy step. v2 uses a new default SQLite file (`instance/talentlens.db`). The old `instance/resume_ranker.db` is imported with `scripts/import_sqlite.py`.
- **Undo for deletes** is client-side: the DELETE request is sent after a 6-second undo window, or immediately when the page is closed.

## Risks

| Risk | Mitigation |
| --- | --- |
| scikit-learn memory on 512 MB | Measure peak RSS; one worker; cap pages and extracted characters per PDF. |
| Malicious or huge PDFs | Magic-byte check, 5 MB cap, 30-page cap, 100k-character cap, every parser error caught per file. |
| Breaking changes to `/api/screen` (no longer saves anonymously; history needs login) | Keep the routes and response keys, and record the change in CHANGELOG. |
| Python 3.10 in CI | numpy ≥2.3 and SQLAlchemy 2.1 drop 3.10, so pins are chosen to resolve on 3.10–3.13 (verified with pip dry runs). |
| Gunicorn cannot run on Windows | Verify the Procfile command inside WSL Ubuntu 24.04. |
| Docker daemon not running | Test Postgres with a throwaway `initdb` cluster from the local PostgreSQL 18 binaries. |

## Assumptions

1. The local folder is a ZIP-style copy of GitHub `main` with no `.git`. I made it a clone (same history and `origin`, no remote edits) and branched `upgrade/v2` from `eda2e6f`.
2. "Owner" means a single admin account, with no multi-user tenancy.
3. A missing must-have caps the final score at `MUST_HAVE_CAP` (default 40).
4. `templates/result.html` is dead code and is removed.
5. Public demo rate limiting: 30 screenings per 10 minutes per IP, in memory.
