"""Evaluation confidence: how much the fit score can be trusted. Separate from fit.

``confidence = 0.40·extraction + 0.20·structure + 0.30·clarity + 0.10·semantic``

* extraction — document-quality score (1 excellent … 0 needs OCR); 1.0 for plain text with ≥ 200 characters;
* structure  — 1.0 if Skills/Experience/Projects headings were found, 0.6 if only other headings, 0.3 if none;
* clarity    — 1 − share of requirements whose verdict is ambiguous (``unclear`` or ``inferred``);
* semantic   — 1.0 when semantic matching ran, 0.7 when it was unavailable (lexical-only evidence).

Levels: High ≥ 0.75, Moderate ≥ 0.5, otherwise Low. This is a transparent heuristic; it is **not**
statistically calibrated against hiring outcomes or reviewer agreement.
"""

from __future__ import annotations


def evaluation_confidence(extraction_score: float, sections: list, requirement_results: list, semantic_used: bool) -> dict:
    kinds = {s.kind for s in sections}
    structure = 1.0 if kinds & {"skills", "experience", "projects"} else 0.6 if kinds else 0.3
    total = len(requirement_results)
    ambiguous = sum(1 for r in requirement_results if r.status in {"unclear", "inferred"})
    clarity = 1.0 - (ambiguous / total if total else 0.0)
    semantic = 1.0 if semantic_used else 0.7
    score = 0.40 * extraction_score + 0.20 * structure + 0.30 * clarity + 0.10 * semantic
    level = "high" if score >= 0.75 else "moderate" if score >= 0.5 else "low"
    reasons = []
    if extraction_score < 0.85:
        reasons.append("Text extraction was incomplete or noisy.")
    if structure < 1.0:
        reasons.append("No Skills, Experience or Projects headings were detected." if structure < 0.6
                       else "Only non-core section headings were detected.")
    if total and ambiguous / total > 0.3:
        reasons.append(f"{ambiguous} of {total} requirements have only ambiguous evidence (listed or inferred).")
    if not semantic_used:
        reasons.append("Semantic matching was unavailable, so paraphrased experience may be missed.")
    return {
        "score": round(score, 2),
        "level": level,
        "factors": {"extraction": round(extraction_score, 2), "structure": structure, "clarity": round(clarity, 2), "semantic": semantic},
        "reasons": reasons or ["Clean extraction, recognised structure and unambiguous evidence."],
        "calibrated": False,
    }
