# AI upgrade plan (Prompt 2)

Status legend: ✅ done · 🔄 in progress · ⏳ planned · ⛔ blocked on a decision

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Backend audit + v2.0 baseline (`benchmarks/baseline/`) | ✅ |
| 2 | Document processing: per-page PDF, DOCX, quality grades, optional OCR | ⛔ D2, D3 |
| 3 | Skill ontology relations, context strength, negation, required/preferred | ⏳ |
| 4 | Hybrid lexical + semantic matching | ⛔ D1 |
| 5 | v3 explainable scoring, mandatory coverage replacing the hard cap | ⏳ |
| 6 | Evidence engine + evaluation confidence | ⏳ |
| 7 | Optional LLM adapter + resume improvement engine | ⛔ D4 |
| 8 | Benchmark (50–100 synthetic annotated pairs), v2 vs improved lexical vs hybrid | ⏳ |
| 9 | Performance (RSS, latency on CPU) + security review | ⏳ |
| 10 | Frontend integration (new fields only, no redesign) + regression | ⏳ |

## Decisions needed (they conflict with the standing free-tier ground rules)
* **D1 Semantic model runtime.** The brief suggests sentence-transformers (torch), but the ground rules forbid torch on Render
  free (512 MB). Recommended: the same MiniLM-L6-v2 run through **ONNX Runtime** (no torch), enabled by env,
  with lexical fallback. Memory fit to be measured before it's enabled on Render.
* **D2 PDF library.** The brief suggests **PyMuPDF**, which is **AGPL-3.0** (or commercial). Using it in a public web
  service would effectively require AGPL for this MIT project. Recommended: keep **pypdf (BSD)** and add per-page
  extraction and layout-tolerant quality checks.
* **D3 OCR.** Tesseract needs a system binary, which Render free can't install. Recommended: optional local-only fallback,
  auto-disabled when the binary is missing. On Render, scanned PDFs are flagged as "unreadable: needs OCR" and excluded from ranking.
* **D4 LLM provider.** The core never depends on it. Recommended: a provider-independent adapter, off by default, with a fake
  provider for tests and at most one real provider.

## Migration risks
| Risk | Mitigation |
| --- | --- |
| The frontend reads v2 fields | v3 adds fields; v2 field names keep their meaning (`score` = fit, `skill_coverage` = requirement coverage) |
| Saved owner screenings hold v2 breakdowns | the renderer already branches on `scoring_version`; old rows keep rendering |
| Score values change for the same input | expected; v2 stays selectable, and the benchmark shows the change |
| Memory on Render | semantic is opt-in until measured; v3 lexical runs on the current footprint |
| More dependencies | each one pinned and checked on Python 3.10–3.13 Linux wheels before it's added |

## Implementation sequence
2 → 3 → 6 (evidence) → 5 (v3 scoring) → 4 (semantic plugs into 5) → 8 (benchmark) → 7 → 9 → 10.
Evidence comes before scoring because v3 components are computed *from* evidence items.
