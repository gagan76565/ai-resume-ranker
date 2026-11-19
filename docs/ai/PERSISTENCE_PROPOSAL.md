# Optional proposal: persistent recruitment projects (NOT implemented)

The brief asks to keep this separate. **Nothing below is built, and current behaviour is unchanged:**
* **Public demo:** files are processed in memory, nothing is stored, and results stay in the browser tab.
* **Owner mode** (from Prompt 1): saved screenings are stored in Postgres (`DATABASE_URL`, currently Neon free).
  Each candidate's breakdown is stored, including evidence quotes and the extracted profile (contact details included), and is
  deleted after `RETENTION_DAYS` (30). **Uploaded files themselves are never stored.**

## What "projects" would add
Long-lived requisitions grouping several screenings, candidate pipeline states (new, reviewing, interview…), multiple
reviewers, and comments.

## Implications
| Area | Implication |
| --- | --- |
| Privacy | Candidate data would live for weeks or months. That needs a lawful basis, a privacy notice for candidates, deletion on request, an export function and audit logs. The UI's "nothing is stored" claim would apply only to the demo. |
| Security | Multiple users need real accounts (password reset, maybe SSO), role-based access, per-project authorisation checks and rate limits per user. |
| Database | New tables (`projects`, `project_members`, `candidate_states`, `comments`, `audit_log`). The current schema and Alembic can absorb these. Postgres is already in place, so no new database is needed. |
| File storage | Keeping the original PDFs would need object storage (for example a free S3-compatible tier), encryption at rest and stricter retention. **Recommendation: don't store originals.** Keep the extracted evidence only, as today. |
| Cost | Neon and Render free tiers cover a single small team. Storing files or keeping the service always on would exceed the free tiers. |
| Effort | About 2–3 weeks: models, migrations, authorisation, UI and tests. |

## Recommendation
Don't build this until there is a concrete multi-user need. If it is built, keep "no original files stored"
and keep retention-based deletion as the default.
