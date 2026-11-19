# Scoring methodology

This file documents **the engine that is live today (v2.0)** precisely, and the **proposed v3
engine**. The v3 section is a design and is marked as such until it's implemented and benchmarked.
TalentLens produces decision support for a human reviewer. **No score is a probability of hiring
success, and the tool never hires or rejects anyone.**

## Part A: v2.0 (current, baseline)

```
final = 100 × (wT·T + wK·K + wS·S) / (wT + wK + wS)        defaults wT=0.40, wK=0.45, wS=0.15
```
* `T`: TF-IDF cosine of the job description and the resume. The vectoriser is fitted on that pair only, with unigrams and within-line bigrams, sublinear TF and English stop words.
* `K`: matched required skills ÷ required skills. Required skills are the taxonomy skills found in the JD plus must-haves.
* `S`: the share of matched skills found inside a Skills, Experience or Projects section.
* With no recognised JD skills, `final = 100 × T`.
* **Must-have cap:** if any must-have is missing, `final = min(final, MUST_HAVE_CAP)` (default 40).
* Labels: ≥70 Strong, ≥45 Partial, ≥20 Weak, otherwise Very weak.
* Weights come from `SCORE_WEIGHT_TFIDF`, `SCORE_WEIGHT_SKILLS` and `SCORE_WEIGHT_SECTIONS`. Invalid values fall back to the defaults.

The full description and its limits are in `docs/scoring.md`. The recorded baseline is in
`benchmarks/baseline/scoring-v2.0-samples.json`.

## Part B: v3 (proposed; status tracked in `AI_UPGRADE_PLAN.md`)

### B1. Requirements model
The job description is parsed into requirements, each with a **class**:
* **mandatory**: must-haves ticked by the recruiter, plus skills in lines under "Required" or "Must have" headings or phrasing;
* **preferred**: skills under "Nice to have", "Preferred", "Bonus" or "Plus";
* **general**: other skills mentioned in the JD.

Alternatives such as "Flask or FastAPI" become one requirement satisfied by either.

### B2. Evidence per requirement
Each requirement gets a status, the strongest evidence item found, and a reason:

| Status | Rule (deterministic) |
| --- | --- |
| **Supported** (strength 1.0) | the skill appears in an Experience or Projects sentence that also contains an action verb (built, developed, implemented…) |
| **Mentioned** (0.6) | the skill appears in an Experience or Projects sentence without an action verb, or in a Skills section |
| **Unclear** (0.3) | the skill appears only outside the recognised sections, or in a keyword list with no context |
| **Negated** (0) | the mention sits inside a negation window ("no", "not", "without", "lack of" …) |
| **Not found** (0) | no mention at all, reported as "not demonstrated in the resume" (never "does not have") |

Every evidence item carries the **exact substring** of the extracted text, its character offsets,
section, page number (when the extractor knows it) and evidence type. Quotations are copied, never generated.

### B3. Components (each normalised to 0–1)
| Component | Default weight | Computed from |
| --- | ---: | --- |
| Required-skill evidence `R` | 0.30 | mean evidence strength over mandatory + general requirements |
| Semantic relevance `M` | 0.25 | section-aware embedding similarity (JD requirement sentences vs experience, project and summary sentences), max-pooled per requirement then averaged. **Optional**: when the model is unavailable the weight is redistributed and the response says so |
| Relevant experience `E` | 0.15 | share of JD requirements with evidence inside Experience entries (lexical or semantic) |
| Project relevance `P` | 0.15 | share of JD requirements with evidence inside Projects with implementation detail (title-only mentions don't count) |
| Preferred skills `Q` | 0.10 | mean evidence strength over preferred requirements (weight redistributed if the JD has none) |
| Lexical relevance `L` | 0.05 | the v2 TF-IDF cosine (kept as a minor signal) |

`fit = 100 × Σ w·c / Σ w(available components)`. A component that can't be computed (for example no Projects
section, or no preferred skills in the JD) is **excluded and reported**. It isn't counted as zero.
These weights are design choices, not validated constants, and they're configurable per job and via env.

**Double counting:** `R` measures *which* requirements are evidenced, while `E` and `P` measure *where*. The benchmark
reports the correlation between components so overlap is visible rather than assumed away.

### B4. Mandatory requirements (replaces the hard cap)
The fit score is never silently replaced. Each candidate shows **separately**:
* `fit_score` (above);
* `mandatory_coverage` = supported or mentioned mandatory requirements ÷ mandatory requirements;
* `missing_mandatory[]`;
* a `mandatory_met` flag. The default sort is `mandatory_met` first, then `fit_score`. The UI badge reads
  "Missing mandatory: X (not demonstrated in resume)".

For backward compatibility, v2.0's `MUST_HAVE_CAP` remains available as an opt-in rule (`MANDATORY_POLICY=cap`).

### B5. Evaluation confidence (separate from fit)
`confidence` (0–1) combines extraction quality (excellent 1, partial 0.7, poor 0.3, scanned 0), section-detection
quality, the share of requirements with unambiguous evidence, and whether the semantic model was used. It's
presented as **High / Moderate / Low**. It's a heuristic and **not statistically calibrated**.

### B6. What is never a signal
Name, email, phone, location, photo, age, gender, nationality, religion, marital status, disability, and the
institution's prestige. Education counts only if the JD explicitly requires a qualification.
