# AI upgrade plan (Prompt 2): status

Status legend: ✅ done · ⚠️ done with stated limitations · ⏳ not done

| # | Milestone | Status | Where |
| --- | --- | --- | --- |
| 1 | Backend audit + v2.0 baseline | ✅ | `BACKEND_AUDIT.md`, `benchmarks/baseline/` (fingerprint `969636fa3f7654f1`) |
| 2 | Document processing: PyMuPDF per page, DOCX, quality grades, optional OCR, structured profile | ⚠️ OCR is optional; the local path is tested with mocks because no Tesseract binary was available | `talentlens/documents/`, `scoring/profile.py` |
| 3 | Ontology (10 groups, related, implied), context strength, negation, mandatory/preferred/general | ✅ | `scoring/ontology.py`, `context.py`, `job.py`, `data/relations.json` |
| 4 | Hybrid lexical + semantic (MiniLM-L6-v2, ONNX, no torch) with fallback | ✅ | `scoring/semantic.py`, `scripts/fetch_model.py` |
| 5 | v3 scoring: six components, configurable weights, mandatory reported separately | ✅ | `scoring/engine_v3.py` |
| 6 | Evidence engine + evaluation confidence | ✅ (confidence is not calibrated) | `scoring/evidence.py`, `confidence.py` |
| 7 | LLM adapter (interface only, per decision D4) + resume-improvement engine | ✅ interface and fake provider; **no real provider shipped** | `talentlens/llm.py`, `scoring/improve.py` |
| 8 | Benchmark: 72 synthetic pairs; v2 vs v3-lexical vs v3-hybrid | ⚠️ labels by construction, not independent | `benchmarks/`, `AI_BENCHMARK_REPORT.md` |
| 9 | Performance + security review | ✅ | below |
| 10 | Frontend integration (no redesign) + regression | ✅ | `static/js/results.js`, `upload.js` |

## Decisions taken (2026-11-19)
| | Decision | Consequence |
| --- | --- | --- |
| D1 | ONNX Runtime + MiniLM-L6-v2 int8 (Apache-2.0), optional | No PyTorch. The model is fetched by script (23 MB, SHA-256 pinned) and isn't in Git |
| D2 | **PyMuPDF** | **The project licence changed MIT → AGPL-3.0-or-later** (separate commit `dbc4b01`) |
| D3 | OCR optional, local only | On Render, scanned PDFs are reported `needs_ocr` and never scored |
| D4 | LLM: interface only | No external calls. `llm_enabled: false` in `/api/v1/config` |

## Performance (measured)
Gunicorn with the Procfile command, on WSL Ubuntu 24.04 and Python 3.12, with 20 two-page PDF resumes per request:

| Mode | Peak worker RSS | 20-file screening (3 runs) | Cold start to `/healthz` |
| --- | ---: | ---: | ---: |
| v3, semantic off | 182 MB | 2.5–2.9 s | 2.6 s |
| v3, semantic on (used) | 263 MB | 2.5–3.2 s | 1.0 s |
| v2.1 baseline (earlier run) | 162.7 MB | 1.1–1.8 s | 1.1–1.7 s |

Other measurements:
* **Encoder throughput:** MiniLM int8 runs at 704 sentences/s on one thread on Linux. On the Windows development laptop it ran at about 60/s; that is an environment quirk, not representative.
* **Semantic budget:** semantic matching runs only when a batch needs ≤ `SEMANTIC_MAX_SENTENCES` (default 700) sentences. Otherwise it's skipped for the **whole** batch, so all candidates are compared on the same components, and every candidate's warnings say so.
* **Load once:** the model loads once per process, embeddings are LRU-cached in memory, and nothing is written to disk.
* **Profile fix:** profiling found that the profile extractor rescanned the taxonomy for every line. Fixing that roughly halved v3 latency.
* **Render free:** this has 512 MB and a slower shared CPU. The measured 263 MB peak leaves headroom, and latency there will be higher than on WSL.

## Security and responsible-use review
| Check | Status |
| --- | --- |
| File type from magic bytes (PDF `%PDF-`, DOCX zip with `word/document.xml`), per-file and total caps, page and character caps | ✅ tests |
| DOCX zip-bomb guard (40 MB unpacked, 2,000 entries) | ✅ test |
| Malicious filenames (path traversal, HTML) reduced to safe display names | ✅ tests |
| Uploaded content never executed, written to disk or rendered as HTML (evidence is rendered with `textContent` and `<mark>`) | ✅ |
| Prompt injection: resume text that looks like instructions is flagged to the recruiter and never obeyed (scoring is deterministic; the LLM interface gets evidence quotes only and rejects non-verbatim citations) | ✅ tests (5 attack and 5 benign phrases) |
| Name, email, phone, labelled location and protected-attribute lines (age, gender, marital status, nationality, religion, …) blanked from scoring text | ✅ fairness test: identical scores |
| Resume content and contact details never logged | ✅ test |
| "Not demonstrated in the resume" wording, never "does not have" | ✅ test |
| Decision support only: no auto-reject. Missing mandatory skills are flagged, not hidden | ✅ |
| PyMuPDF CPU cost on hostile PDFs has no per-file timeout, only the page cap and gunicorn's 120 s worker timeout | ⚠️ known limitation |

## Remaining limitations
* The benchmark labels aren't independent (see `AI_BENCHMARK_REPORT.md`), and there are no real-resume results.
* Case-sensitive aliases without context ("Express" as an employer) still give false positives.
* The components are correlated (required, experience and projects at 0.71–0.82), so the weights need re-validation on real data.
* Confidence is a heuristic and isn't calibrated.
* Profile extraction (titles, companies, education) is rule-based and English-only. Uncommon layouts give `unparsed`.
* No real LLM provider exists. Adding one needs a key, a privacy disclosure to users, and review.
