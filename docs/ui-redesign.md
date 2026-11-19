# TalentLens UI redesign: audit and plan (v2.1)

## Audit of the v2.0 frontend
| Area | Finding |
| --- | --- |
| Structure | `templates/base.html` + 5 pages, `static/css/app.css` (one file, tokens on `:root`), `static/js/{core,upload,results,history,theme-init}.js`. No framework, strict CSP (`'self'` only, no inline style/script). |
| Tokens | Colour tokens with light/dark sets; spacing scale; no type scale, no elevation scale, system font only. |
| Shell | Plain top bar; nav items look like text links; theme button is a bare icon; no mobile menu (nav wraps). |
| Screening | Two equal cards plus a separate full-width card that only holds the button, so the primary action sits far below the fold and leaves a wide empty strip. The tall textarea makes the left card much taller than the right. No sense of progress. |
| Results | Metric cards are tall, empty boxes; "missing skills" is a cloud of identical chips; the requirements panel is a bare `<details>`; table columns are cramped (candidate name wraps, skills column dominates); the expanded detail is one long wall of headings. |
| Compare | Two plain columns; differences are not highlighted; no way to remove a candidate. |
| Dark mode | Token-driven and accessible, but surfaces are flat (little separation between levels). |
| Motion | Only toasts/skeleton/spinner. |
| Verified features (kept) | sample JD, sample resumes, drag-and-drop + picker, per-file validation (extension, size, `%PDF`), total/count limits, must-have tags with autocomplete, upload progress, wake-up retry, results summary, filters, sorting, shortlist (local or saved), compare (max 2), CSV of current view, print, explanation panel with `<mark>` evidence, owner notes, delete with Undo, history search, theme persistence. |

## Plan
1. **Design system:** vendored Inter (variable, OFL) with tabular figures; type scale; refined indigo accent; neutral slate greys; 3 surface levels per theme; elevation scale; radius scale; motion tokens; components: button, icon button with tooltip, input/select/textarea, card, panel, badge, chip, tabs, switch, progress, toast, skeleton, empty state, modal.
2. **Shell:** brand mark + wordmark, segmented nav with clear active state, sun/moon switch, mobile menu button.
3. **Screening:** intro with headline and a *live* 3-step workflow indicator (derived from form state), two-column workspace (role / resumes), integrated privacy panel, sticky action bar that summarises what will be submitted.
4. **Results:** header with metadata chips, metric cards with real distributions (score bands, top candidate), skill-gap bars, tabbed requirements panel (skills, weights, method, JD), refined toolbar, re-proportioned table (desktop) and cards (mobile), redesigned detail drawer in sections.
5. **Compare:** side-by-side workspace with aligned rows, score deltas, shared vs unique skills, evidence, remove buttons.
6. **Themes, motion, responsive:** test at 1440/1024/768/375 in both themes; reduced motion.
7. **QA:** pytest, `scripts/ui_check.py`, scoring fingerprint before/after, before/after screenshots.

No backend, scoring, extraction, API or privacy behaviour changes are needed: every new visual (score-band distribution, gap bars, deltas, workflow state) is computed client-side from data the API already returns.
