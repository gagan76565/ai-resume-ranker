# Scoring methodology

TalentLens produces decision support for a human reviewer. **No score is a probability of hiring success, and
the tool never hires or rejects anyone.** Two engines exist. `SCORING_ENGINE=v3` (the default since v2.2.0) is described in
Part B. `v2` is kept for comparison and regression, and screenings saved before v2.2.0 use it.

## Part A: v2.0 (baseline, still selectable)

```
final = 100 × (wT·T + wK·K + wS·S) / (wT + wK + wS)        defaults wT=0.40, wK=0.45, wS=0.15
```
* `T`: TF-IDF cosine of the JD and the resume, fitted on that pair only (unigrams and within-line bigrams, sublinear TF, English stop words).
* `K`: matched required skills ÷ required skills. Required skills are the taxonomy skills found in the JD plus must-haves.
* `S`: the share of matched skills found inside a Skills, Experience or Projects section.
* With no recognised JD skills, `final = 100 × T`. If a must-have is missing, `final = min(final, MUST_HAVE_CAP)` (default 40).
* The weights come from `SCORE_WEIGHT_TFIDF`, `SCORE_WEIGHT_SKILLS` and `SCORE_WEIGHT_SECTIONS`. Details are in `docs/scoring.md`. The baseline outputs are in `benchmarks/baseline/`.

## Part B: v3.0 (implemented in v2.2.0; `scoring/engine_v3.py`)

### B1. Requirements (`scoring/job.py`)
Each recognised skill in the JD becomes a requirement with a class:

| Class | Source |
| --- | --- |
| **mandatory** | must-haves ticked by the recruiter (listed first), and skills in a line or block introduced by "required", "requirements", "must (have)", "essential", "mandatory" or "minimum qualifications" |
| **preferred** | skills in a line or block introduced by "nice to have", "preferred", "bonus", "a plus", "desirable", "good to have", "optional" or "ideally" |
| **general** | every other recognised JD skill |

Parsing rules:
* "X or Y", "X/Y" and "A, B or C" become **one** requirement satisfied by any of the options.
* A comma list *without* "or" stays as separate requirements.
* Negated JD mentions ("Java is not required") are dropped.

### B2. Evidence per requirement (`scoring/evidence.py`, `scoring/context.py`)
Every mention of a requirement skill is classified by its clause, which ends at sentence punctuation followed
by a space, a newline, ";", "but" or "however". "Node.js" is safe because a "." without a following space is not a break.

| Mention type | Strength | Rule |
| --- | ---: | --- |
| demonstrated | 1.0 | the clause has an action verb (built, developed, deployed, used …) and is not in a Skills section |
| listed | 0.6 | the mention is in a Skills section, or in an Experience/Projects clause without an action verb |
| bare | 0.3 | the mention is anywhere else (no recognised section, summary without an action verb) |
| negated | 0 | a negation cue governs it within three words ("no experience with", "never used", "without any") |

Each requirement's verdict comes from its strongest non-negated mention:

| Verdict | Strength | Meaning |
| --- | ---: | --- |
| **supported** | 1.0 | described in use |
| **unclear** | 0.6 or 0.3 | mentioned, but how it was used isn't established |
| **inferred** | 0.25 | not mentioned; implied by another mentioned skill (`relations.json`, e.g. PostgreSQL ⇒ SQL). This is unverified |
| **negated** | 0 | the resume states the experience is absent |
| **not_found** | 0 | not demonstrated in the resume (which doesn't mean the candidate lacks it) |

Evidence items carry the **exact** clause quote (`text[start:end]`), the match span, alias or exact form, section,
page (PDF), type and strength. Related technologies (Angular for a React requirement) are listed as
`related_found` and **never** count as the requirement.

### B3. Components and fit
| Key | Default weight | Value (0–1) |
| --- | ---: | --- |
| required | 0.30 | mean verdict strength over **mandatory + general** requirements |
| semantic | 0.25 | mean over JD sentences (≥ 25 characters; the title line of a multi-line JD is skipped; at most 15) of the best MiniLM cosine against resume sentences from Experience, Projects and Summary (at most 30; the whole text if there are no headings), rescaled as `clip((cos − 0.15) / 0.50, 0, 1)` |
| experience | 0.15 | mean over mandatory + general requirements of the strongest evidence located **inside Experience** |
| projects | 0.15 | mean over **all** requirements of the strongest evidence inside Projects. Only described use counts fully; listed or title mentions count 0.3 |
| preferred | 0.10 | mean verdict strength over preferred requirements |
| lexical | 0.05 | the v2 TF-IDF cosine |

```
fit = 100 × Σ wᵢ·cᵢ / Σ wᵢ      over available components only
```

When a component isn't available, it's **excluded and its reason reported**:

| Component | Unavailable when |
| --- | --- |
| required | there are no mandatory or general requirements |
| preferred | there are no preferred requirements |
| semantic | the model is missing or disabled, or the batch exceeds `SEMANTIC_MAX_SENTENCES` (default 700). In the last case it's skipped for the whole batch |
| experience | there's no Experience section but there is a Projects section |
| projects | there's no Projects section but there is an Experience section |

If *both* Experience and Projects are missing, both count as 0. There's no located evidence at all.

The weights come from `V3_WEIGHT_<KEY>` or a per-request `weights` object, and are always re-normalised. They're
design choices, not validated constants. The benchmark measured strong correlation between required, experience and
projects (0.71–0.82), so the effective weight of "skill evidence" is higher than its nominal 30%.

Contact details and protected-attribute lines are blanked (offsets preserved) before the lexical and semantic signals
are computed. Skills are never taken from them, so they can't influence the score.

### B4. Mandatory requirements (replaces the hard cap)
Each candidate reports, **separately from fit**:
* `mandatory.total` and `mandatory.covered` (supported or unclear verdicts);
* `mandatory.coverage` (%) and the `missing` list;
* `mandatory.met`.

With `MANDATORY_POLICY=flag` (the default):
* the fit score is **not** capped;
* candidates who meet all mandatory requirements are sorted first, then by fit;
* the UI badge reads "Missing mandatory: n", and the warning says "(not demonstrated in the resume)".

`MANDATORY_POLICY=cap` restores the v2 cap at `MUST_HAVE_CAP`, and still reports the uncapped `evidence_fit`.

### B5. Evaluation confidence (`scoring/confidence.py`), separate from fit
```
confidence = 0.40·extraction + 0.20·structure + 0.30·clarity + 0.10·semantic
```

| Input | Value |
| --- | --- |
| extraction | the document-quality score from 0 to 1: excellent is 0.85 or more, partial 0.5 to 0.85, and poor below 0.5. needs_ocr documents aren't scored at all. Plain text gives 1.0 at 200+ characters, otherwise 0.4 |
| structure | 1.0 if Skills, Experience or Projects headings exist; 0.6 if only other headings; 0.3 if none |
| clarity | 1 − the share of requirements with unclear or inferred verdicts |
| semantic | 1.0 if used; 0.7 if not |

Levels: High ≥ 0.75, Moderate ≥ 0.5, otherwise Low. This is a transparent heuristic and is **not calibrated**.

### B6. Never a signal
Name, email, phone, location, URLs, photo, age or date of birth, gender, marital status, nationality, religion,
ethnicity, disability and institution prestige. Education is extracted for display only. No component scores it.

### B7. Labels
These are the same bands as v2: ≥70 Strong, ≥45 Partial, ≥20 Weak, otherwise Very weak. The number is always shown next to the label.
