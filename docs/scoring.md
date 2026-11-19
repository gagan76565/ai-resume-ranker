# How TalentLens scores a resume (scoring v2)

TalentLens ranks resumes by **how closely their text matches a job description**.
It is a reading aid for a human reviewer. **It is not a hiring decision, it does
not predict job performance, and similarity is not suitability.**

The code lives in [`scoring/`](../scoring). It is pure Python plus scikit-learn,
with no Flask dependency, and every function has a docstring. The scoring
version is reported as `scoring_version` (currently `2.0`).

## The formula

```
final_score = 100 × (w_t·T + w_k·K + w_s·S) / (w_t + w_k + w_s)
```

| Symbol | Name | Range | Meaning |
| --- | --- | --- | --- |
| `T` | TF-IDF similarity | 0–1 | Cosine similarity of the JD and resume TF-IDF vectors |
| `K` | Skill coverage | 0–1 | `matched required skills ÷ required skills` |
| `S` | Section signal | 0–1 | `matched skills found inside a Skills / Experience / Projects section ÷ matched skills` |
| `w_t` | `SCORE_WEIGHT_TFIDF` | ≥ 0 | default **0.40** |
| `w_k` | `SCORE_WEIGHT_SKILLS` | ≥ 0 | default **0.45** |
| `w_s` | `SCORE_WEIGHT_SECTIONS` | ≥ 0 | default **0.15** |

Weights are normalised, so `1, 1, 2` means 25% / 25% / 50%. A value that is not a
number, is negative or is NaN falls back to that weight's default. If all three
are zero, all three fall back to the defaults. In both cases a warning is logged
at start-up and shown in `/api/v1/config`.

The result is clamped to 0–100 and rounded to one decimal place.

### Rules applied after the blend

1. **No recognised skills in the JD** (and no must-haves): `K` and `S` are
   undefined, so the score is `100 × T`. The candidate gets the warning *"No
   known skills were found in the job description…"*.
2. **Must-have cap.** If any skill ticked as must-have is missing, the final
   score is capped at `MUST_HAVE_CAP` (default **40**). The UI shows a
   "Capped" badge and the warning *"Missing must-have skill: X. The score is
   capped at 40."*. A cap is used rather than a subtraction because it is
   easy to explain: a candidate without a must-have never ranks as a
   "Partial" (45+) or "Strong" (70+) match.

### Score labels

The numeric score is always shown next to a text label. Colour is never the
only cue.

| Score | Label |
| --- | --- |
| ≥ 70 | Strong match |
| 45–69.9 | Partial match |
| 20–44.9 | Weak match |
| < 20 | Very weak match |

## Components in detail

### T — TF-IDF cosine similarity

* Fitted on **exactly two documents**, the JD and one resume. A candidate's
  score therefore never depends on who else is in the batch.
* Features are unigrams plus **bigrams formed only within a line or sentence**.
  Moving whole lines or sections around therefore cannot change the vector
  (see the bias tests).
* English stop words (scikit-learn's list) are removed. So are pure numbers
  and 1-character tokens. Tokens keep `+`, `#`, and inner `.`, `-` or `/`
  (`c++`, `node.js`, `ci/cd`).
* `sublinear_tf=True` (`1 + log tf`), smooth IDF, L2 normalisation.
* `top_terms` lists the shared features with the largest
  `tfidf_jd × tfidf_resume` product, which are the terms that contribute most to the
  cosine. Ties are broken alphabetically.

Real documents rarely exceed `T ≈ 0.5` even when they match well, because resumes
contain much text that a JD does not. That is why `T` is blended with skill
coverage instead of being used alone, as v1 did.

### K — skill coverage

**Required skills** are every taxonomy skill found in the job description, plus
any must-have skills the user ticks. Must-haves are resolved to canonical
names (`k8s` → `Kubernetes`); a must-have that is not in the taxonomy is
matched literally, with the same boundary rules.

The taxonomy is [`scoring/data/skills.json`](../scoring/data/skills.json): 207
skills, each with a `canonical` name, `aliases` and a `category`.

Matching rules:

* Case-insensitive, except for a short list of `case_sensitive` aliases.
  These are single letters (`C`, `R`) and words that are also everyday
  English (`Go`, `Swift`, `Excel`, `Rust`, `Dart`, `Express`).
* Word-boundary safe: a match cannot touch letters, digits, `_`, `+`, `#` or `&`,
  and cannot be followed by `.x`. So `java` ≠ `javascript`, `c` ≠ `c++`,
  and `node` inside `node.js` is not double-counted.
* Single-letter skills also cannot touch `-`, `'`, `.` or `&`. So `R&D`,
  `C-level` and initials such as `C.` do not count as skills.
* When aliases overlap, the longest wins (`SQL Server`, not `SQL`).

### S — section signal

A heading is a short line, or the part before a colon, such as `Skills`,
`Technical Skills`, `Work Experience` or `Projects` (see
[`scoring/sections.py`](../scoring/sections.py) for the full list). Each section
runs until the next recognised heading, including non-signal headings such as
`Education`. A matched skill counts towards `S` if **any** of its occurrences is
inside a Skills, Experience or Projects section. If no such headings are found,
`S = 0` and the candidate gets a warning.

## The breakdown object

Every candidate in every API response carries these fields:

| Field | Type | Notes |
| --- | --- | --- |
| `final_score` | number | 0–100, one decimal |
| `tfidf_score`, `skill_coverage`, `section_signal` | number | 0–100, one decimal |
| `label` | string | see the labels table |
| `matched_skills[]`, `missing_skills[]` | string[] | in the order the JD lists them, must-haves first |
| `must_have_missing[]` | string[] | |
| `top_terms[]` | string[] | up to 10 |
| `warnings[]` | string[] | low text, must-have cap, no headings, no JD skills |
| `weights`, `contributions` | object | the normalised weights and each component's points |
| `capped`, `low_text`, `text_chars` | | |
| `snippets[]` | object[] | short excerpts as `segments` of `{text, mark}`, rendered with `textContent` and `<mark>` (never as HTML) |

## Low-text (likely scanned) PDFs

If fewer than `LOW_TEXT_THRESHOLD` characters (default **200**) are extracted,
the candidate is still scored and listed, with the warning *"Very little
extractable text… likely scanned. OCR is not available on the free tier…"*.
OCR needs system packages such as Tesseract, which Render's free tier cannot
install, so OCR is not offered.

## Known limits

* **English only.** Stop words, headings and aliases are English.
* **Can miss context.** "Taught a Python workshop" and "5 years of production
  Python" both match `Python`. Seniority, recency and depth are not
  measured, and neither are negations such as "no Java experience".
* **The taxonomy is finite.** Skills outside the 207 entries only count when
  they are typed as must-haves. Some aliases can be ambiguous. For example, a
  sentence starting with "Go" counts as the Go language, and "Express" can be
  an employer name.
* **Layout-dependent extraction.** Multi-column PDFs may extract in an odd
  order. Only the section signal is affected, because `T` and `K` do not depend on order.
* **Not validated against hiring outcomes.** No accuracy figure is claimed.

## Fairness checks in the test suite

[`tests/test_scoring_engine.py`](../tests/test_scoring_engine.py) asserts:

* identical content in a different section order → identical score;
* the same batch in a different upload order → identical scores and ranking;
* a different candidate name on the same content → identical score;
* five repeated runs → byte-identical output;
* invalid or zero weights → safe defaults.
