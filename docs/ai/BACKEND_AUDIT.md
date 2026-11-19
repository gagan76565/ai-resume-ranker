# TalentLens backend audit (before the Prompt 2 engine upgrade)

Audited on 2026-11-19 at commit `b16e325` (branch `fix/render-host`, which is `main` plus a Render bind fix).
Every statement below was checked against the code, not against screenshots.

## 1. Map of the backend

| Concern | Where | Notes |
| --- | --- | --- |
| Flask app factory | `talentlens/__init__.py` (`create_app`) | `app.py` is a thin WSGI entry for gunicorn |
| Config | `talentlens/config.py` | All settings come from env vars; scoring weights via `ScoringConfig.from_env` |
| HTTP API | `talentlens/api.py` | `/api/v1/*` + v1.x aliases; `POST /screen`, `/score`, `/samples/screen` |
| Pages | `talentlens/views.py` | `/`, `/results`, `/screenings/<id>`, `/history`, `/login`, `/healthz`, `/readyz` |
| Upload validation + PDF text | `talentlens/uploads.py` | `.pdf`, `%PDF-` magic, 5 MB/file, 20 files, 30 MB total, 30 pages, 100k chars; **pypdf 7.1.1** (not PyPDF2) |
| Screening pipeline | `talentlens/screening.py` | builds the response, pool summary, `parse_must_have` |
| Scoring engine | `scoring/` (pure Python + scikit-learn, no Flask) | `text.py`, `skills.py`, `sections.py`, `tfidf.py`, `engine.py` |
| Skill taxonomy | `scoring/data/skills.json` | 207 skills, aliases, 12 categories, case-sensitive aliases for ambiguous words |
| Persistence | `talentlens/db.py`, `repository.py`, `migrations/` | SQLAlchemy Core + Alembic; **owner-only** history (public demo stores nothing) |
| Auth, CSRF, rate limits, headers | `talentlens/auth.py`, `security.py` | single owner from env; in-memory limiter |
| Samples | `samples/` + `scripts/make_samples.py` | 6 synthetic resume PDFs + sample job description |
| CSV export | `talentlens/api.py` (`candidates_csv`) and `static/js/results.js` (current view) | formula-injection guard on both |
| Tests | `tests/` | 144 tests (pytest and unittest), CI on 3.10–3.12 + Postgres |

## 2. Pipeline as implemented today

1. **Validate** each upload: extension, magic bytes, sizes. Errors name the file.
2. **Extract** text with `pypdf.PdfReader(...).pages[i].extract_text()` for at most 30 pages. Pages are
   joined with `\n` and **page numbers are discarded**. Encrypted PDFs are tried with an empty password.
3. **Normalise** text: NFKC, bullets become newlines, Unicode dashes become `-`, whitespace is collapsed.
4. **Required skills** = every taxonomy skill found in the job description, plus must-haves
   (resolved to canonical names, or kept literal if unknown).
5. **Per resume:**
   * TF-IDF cosine similarity `T` of the (JD, resume) pair. Unigrams and within-line bigrams,
     sublinear TF, English stop words.
   * Skill occurrences found with boundary-safe regexes, with the longest alias winning.
   * Sections from an English heading lexicon (skills / experience / projects / other).
   * `K` = matched required ÷ required. `S` = matched skills that appear inside a signal section ÷ matched skills.
6. **Score** = `100 × (0.40·T + 0.45·K + 0.15·S)`, with weights normalised. Without JD skills, `T` is used alone.
   A missing must-have **caps** the final score at 40.
7. **Explain:** label bands (70/45/20), top shared TF-IDF terms, up to 4 snippets (±70 characters) with
   marked matches, and warnings (low text under 200 characters, no headings, no JD skills, must-have cap).
8. **Rank** by final score, then TF-IDF, then file name. The order is deterministic and independent of upload order.

## 3. Baseline (recorded before any change)

`python benchmarks/record_baseline.py`, saved to `benchmarks/baseline/scoring-v2.0-samples.json`,
fingerprint `969636fa3f7654f1`:

| Rank | Sample | Final | T | K | S | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | aarav-placeholder (strong) | 60.4 | 20.8 | 82.4 | 100 | |
| 2 | bea-example (aliases) | 42.3 | 8.6 | 52.9 | 100 | |
| 3 | dana-keywords (keyword list, no sections) | 41.0 | 23.0 | 70.6 | 0 | **keyword stuffing ranks 3rd** |
| 4 | eli-analyst (partial) | 25.2 | 5.7 | 17.6 | 100 | |
| 5 | chen-fictional (front-end) | 19.8 | 5.3 | 5.9 | 100 | missing must-haves (below the cap anyway) |
| 6 | scanned-simulation | 4.5 | 11.3 | 0 | 0 | low-text warning, still ranked |

## 4. Strengths
* Pure, deterministic, order-invariant scoring with fairness tests (name, section order and upload order don't change scores).
* Boundary-safe matching with false-positive guards (`java`≠`javascript`, `C`/`R` rules, longest match).
* A transparent formula and a per-candidate breakdown; evidence rendered safely (`textContent` + `<mark>`).
* Small footprint: peak worker RSS 162.5–162.7 MB, 20 PDFs in 1.1–1.8 s (measured in WSL with gunicorn).
* Safe uploads (magic bytes, caps), and no resume content is logged.

## 5. Limitations and accuracy risks
| # | Limitation | Effect |
| --- | --- | --- |
| L1 | A skill counts as "present" whether it sits in a keyword list or in a described implementation | keyword lists score as well as real experience (Dana ranks 3rd) |
| L2 | No negation handling | "no experience with Java" counts as Java |
| L3 | Every JD skill is "required"; there's no preferred / nice-to-have split | "Nice to have: Kubernetes" penalises like a hard requirement; "Flask or FastAPI" requires both |
| L4 | TF-IDF fitted on two documents | IDF is crude (shared terms get the minimum weight); `T` is small (5–23) and length-sensitive |
| L5 | No semantic matching | "built REST services" doesn't match "API development" unless taxonomy aliases cover it |
| L6 | The section signal is binary per skill and uses an English heading lexicon | multi-column PDFs and unusual headings give `S = 0` |
| L7 | The must-have cap replaces the score | the evidence-based score is hidden behind 40; there's no separate "mandatory coverage" number |
| L8 | Extraction is flat text: page numbers are lost, there's no DOCX, no OCR, and the only quality grade is "fewer than 200 characters" | evidence can't cite pages; scanned resumes are still ranked (at the bottom) |
| L9 | No structured profile (experience entries, dates, education, projects) | no experience or project relevance |
| L10 | Evidence = first occurrence ±70 characters per matched skill | no per-requirement status (supported / unclear / not found) |
| L11 | No evaluation confidence separate from fit | a "60" from a clean resume and a "60" from a garbled one look identical |
| L12 | No ground-truth benchmark | accuracy has never been measured |
| L13 | Ambiguous aliases (`Go` at a sentence start, `Express` as an employer) | occasional false positives |

## 6. Performance limits
* Single gunicorn worker with 4 threads; scoring is CPU-bound and sequential per request.
* TF-IDF vectoriser per pair: cheap (milliseconds).
* Render free: 512 MB RAM, a slow shared CPU, an ephemeral disk, no `apt-get` (so no system Tesseract), no workers.

## 7. Backward-compatibility surface the frontend depends on
`static/js/results.js` and `upload.js` read these response fields (they must keep working):
`screening_id, saved, job_title, job_description, created_at, candidate_count, must_have, required_skills,
scoring_version, weights, summary{top_missing_skills}, file_errors, sample_data, settings.must_have_cap` and, per
candidate, `id, rank, filename, display_name, score, matched_terms, missing_terms, explanation, shortlisted, notes`,
`breakdown{label, tfidf_score, skill_coverage, section_signal, matched_skills, missing_skills, must_have_missing,
skills_in_sections, required_skills, top_terms, warnings, weights, contributions, capped, text_chars, low_text, snippets[]}`.
The CSV keeps its first five v1.x columns. Saved owner screenings in Postgres store `breakdown` as JSON, so old
rows must keep rendering after the upgrade.

## 8. Notes on the Prompt 2 brief vs reality
* The brief mentions PyPDF2. The code already uses **pypdf 7.1.1** (BSD).
* The brief asks not to add PostgreSQL "for appearance". Postgres already exists from Prompt 1, **for owner history only**,
  via `DATABASE_URL` (Neon, live). The public demo still stores nothing. Nothing in Prompt 2 needs new storage.
* The shared ground rules (free tier, ≤512 MB, no torch/transformers on Render, no system packages) conflict with
  some suggestions in the brief (sentence-transformers, Tesseract, PyMuPDF). See `AI_UPGRADE_PLAN.md` §Decisions.
