"""Hybrid, explainable resume score.

``final = 100 * (w_t * T + w_k * K + w_s * S) / (w_t + w_k + w_s)``

* ``T`` — TF-IDF cosine similarity of job description and resume (0..1).
* ``K`` — skill coverage: share of required skills found in the resume (0..1).
* ``S`` — section signal: share of *matched* skills that appear inside a
  Skills / Experience / Projects section (0..1).

If the job description contains no recognised skills and no must-haves were
given, ``K`` and ``S`` are undefined and the score falls back to ``T`` alone
(with a warning). If any must-have skill is missing, the final score is
capped at ``must_have_cap``. See ``docs/scoring.md`` for the full rationale.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

from .sections import detect_sections, in_signal_section
from .skills import Taxonomy, custom_skill_pattern, load_taxonomy
from .text import normalize_text
from .tfidf import tfidf_similarity

SCORING_VERSION = "2.0"

DEFAULT_WEIGHTS = {"tfidf": 0.40, "skills": 0.45, "sections": 0.15}
DEFAULT_MUST_HAVE_CAP = 40.0
DEFAULT_LOW_TEXT_THRESHOLD = 200

LABELS = ((70.0, "Strong match"), (45.0, "Partial match"), (20.0, "Weak match"), (0.0, "Very weak match"))

LOW_TEXT_WARNING = (
    "Very little extractable text ({chars} characters); this PDF is likely scanned. "
    "OCR is not available on the free tier, so the score may be unfairly low."
)


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and thresholds. Weights are normalised to sum to 1 at scoring time."""

    weight_tfidf: float = DEFAULT_WEIGHTS["tfidf"]
    weight_skills: float = DEFAULT_WEIGHTS["skills"]
    weight_sections: float = DEFAULT_WEIGHTS["sections"]
    must_have_cap: float = DEFAULT_MUST_HAVE_CAP
    low_text_threshold: int = DEFAULT_LOW_TEXT_THRESHOLD
    warnings: tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ScoringConfig:
        """Read ``SCORE_WEIGHT_TFIDF``, ``SCORE_WEIGHT_SKILLS``, ``SCORE_WEIGHT_SECTIONS``,
        ``MUST_HAVE_CAP`` and ``LOW_TEXT_THRESHOLD``.

        Invalid values (non-numeric, negative, NaN) fall back to the default for
        that setting; if every weight is zero all three fall back to defaults.
        Problems are recorded in ``warnings`` rather than raised, so a typo in
        an environment variable cannot take the service down.
        """
        env = os.environ if env is None else env
        problems: list[str] = []

        def number(name: str, default: float, minimum: float = 0.0, maximum: float = math.inf) -> float:
            raw = env.get(name)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                problems.append(f"{name}={raw!r} is not a number; using {default}.")
                return default
            if math.isnan(value) or value < minimum or value > maximum:
                problems.append(f"{name}={raw!r} is out of range; using {default}.")
                return default
            return value

        weights = (
            number("SCORE_WEIGHT_TFIDF", DEFAULT_WEIGHTS["tfidf"]),
            number("SCORE_WEIGHT_SKILLS", DEFAULT_WEIGHTS["skills"]),
            number("SCORE_WEIGHT_SECTIONS", DEFAULT_WEIGHTS["sections"]),
        )
        if sum(weights) <= 0:
            problems.append("All score weights are zero; using defaults.")
            weights = tuple(DEFAULT_WEIGHTS.values())
        return cls(
            weight_tfidf=weights[0],
            weight_skills=weights[1],
            weight_sections=weights[2],
            must_have_cap=number("MUST_HAVE_CAP", DEFAULT_MUST_HAVE_CAP, 0.0, 100.0),
            low_text_threshold=int(number("LOW_TEXT_THRESHOLD", DEFAULT_LOW_TEXT_THRESHOLD, 0.0, 100000.0)),
            warnings=tuple(problems),
        )

    def normalised_weights(self, skills_available: bool = True) -> dict[str, float]:
        """Weights rescaled to sum to 1. Without required skills only TF-IDF remains."""
        if not skills_available:
            return {"tfidf": 1.0, "skills": 0.0, "sections": 0.0}
        raw = {"tfidf": self.weight_tfidf, "skills": self.weight_skills, "sections": self.weight_sections}
        total = sum(raw.values())
        if total <= 0:
            raw, total = dict(DEFAULT_WEIGHTS), sum(DEFAULT_WEIGHTS.values())
        return {key: value / total for key, value in raw.items()}


@dataclass(frozen=True)
class RequiredSkills:
    """Skills a resume is checked against: from the JD plus user must-haves."""

    skills: tuple[str, ...]
    must_have: tuple[str, ...]
    custom: tuple[str, ...]  # must-haves that are not in the taxonomy


def label_for(score: float) -> str:
    """Human-readable band for a 0..100 score."""
    for threshold, label in LABELS:
        if score >= threshold:
            return label
    return LABELS[-1][1]


def required_skills(job_description: str, must_have: Iterable[str] = (), taxonomy: Taxonomy | None = None) -> RequiredSkills:
    """Skills found in the job description (in order of first appearance) plus must-haves.

    Must-haves are resolved to canonical taxonomy names when possible (so
    ``k8s`` becomes ``Kubernetes``); unknown ones are kept as typed.
    Duplicates are removed case-insensitively.
    """
    taxonomy = taxonomy or load_taxonomy()
    ordered: list[str] = []
    seen: set[str] = set()
    must: list[str] = []
    custom: list[str] = []
    for raw in must_have:
        name = normalize_text(raw)[:60]
        if not name:
            continue
        canonical = taxonomy.canonical_for(name)
        resolved = canonical or name
        if resolved.lower() in seen:
            continue
        seen.add(resolved.lower())
        ordered.append(resolved)
        must.append(resolved)
        if canonical is None:
            custom.append(resolved)
    for canonical in taxonomy.found_skills(normalize_text(job_description)):
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            ordered.append(canonical)
    return RequiredSkills(tuple(ordered), tuple(must), tuple(custom))


def _skill_occurrences(text: str, required: RequiredSkills, taxonomy: Taxonomy) -> dict[str, list[tuple[int, int]]]:
    occurrences: dict[str, list[tuple[int, int]]] = {}
    for canonical, matches in taxonomy.found_skills(text).items():
        occurrences[canonical] = [(m.start, m.end) for m in matches]
    for name in required.custom:
        spans = [(m.start(), m.end()) for m in custom_skill_pattern(name).finditer(text)]
        if spans:
            occurrences[name] = spans
    return occurrences


def _snippets(text: str, matched: list[str], occurrences, sections, limit: int = 4, radius: int = 70) -> list[dict]:
    """Short excerpts around matched skills, split into plain/marked segments.

    The client renders segments with ``textContent`` and wraps marked ones in
    ``<mark>``, so no resume text is ever interpreted as HTML.
    """
    all_spans = sorted(span for name in matched for span in occurrences.get(name, []))
    snippets = []
    used_windows: list[tuple[int, int]] = []
    for name in matched:
        spans = occurrences.get(name) or []
        if not spans:
            continue
        preferred = [s for s in spans if in_signal_section(s[0], sections)] or spans
        start, end = preferred[0]
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        lo = max(line_start, start - radius)
        hi = min(line_end, end + radius)
        if any(lo < u_hi and hi > u_lo for u_lo, u_hi in used_windows):
            continue
        used_windows.append((lo, hi))
        segments = []
        cursor = lo
        for s_start, s_end in all_spans:
            if s_start < lo or s_end > hi or s_start < cursor:
                continue
            if s_start > cursor:
                segments.append({"text": text[cursor:s_start], "mark": False})
            segments.append({"text": text[s_start:s_end], "mark": True})
            cursor = s_end
        if cursor < hi:
            segments.append({"text": text[cursor:hi], "mark": False})
        snippets.append({
            "skill": name,
            "prefix_ellipsis": lo > line_start,
            "suffix_ellipsis": hi < line_end,
            "segments": segments,
        })
        if len(snippets) >= limit:
            break
    return snippets


def score_resume(
    job_description: str,
    resume_text: str,
    must_have: Iterable[str] = (),
    config: ScoringConfig | None = None,
    taxonomy: Taxonomy | None = None,
    required: RequiredSkills | None = None,
) -> dict:
    """Score one resume against a job description and return the full breakdown.

    Returned keys: ``final_score``, ``tfidf_score``, ``skill_coverage``,
    ``section_signal`` (all 0..100, one decimal), ``label``,
    ``matched_skills``, ``missing_skills``, ``must_have_missing``,
    ``required_skills``, ``top_terms``, ``warnings``, ``weights``,
    ``contributions``, ``capped``, ``text_chars``, ``low_text``, ``snippets``,
    ``scoring_version``.
    """
    config = config or ScoringConfig()
    taxonomy = taxonomy or load_taxonomy()
    required = required or required_skills(job_description, must_have, taxonomy)
    text = normalize_text(resume_text)
    warnings: list[str] = []

    text_chars = len(text)
    low_text = text_chars < config.low_text_threshold
    if low_text:
        warnings.append(LOW_TEXT_WARNING.format(chars=text_chars))

    tfidf = tfidf_similarity(job_description, text)
    occurrences = _skill_occurrences(text, required, taxonomy)
    sections = detect_sections(text)

    matched = [name for name in required.skills if name in occurrences]
    missing = [name for name in required.skills if name not in occurrences]
    must_missing = [name for name in required.must_have if name not in occurrences]

    skills_available = bool(required.skills)
    coverage = len(matched) / len(required.skills) if skills_available else 0.0
    in_section = [name for name in matched if any(in_signal_section(s, sections) for s, _ in occurrences[name])]
    section_signal = len(in_section) / len(matched) if matched else 0.0

    if not skills_available:
        warnings.append("No known skills were found in the job description, so the score uses text similarity only.")
    elif matched and not any(s.kind in ("skills", "experience", "projects") for s in sections):
        warnings.append("No Skills, Experience or Projects headings were detected, so the section signal is 0.")

    weights = config.normalised_weights(skills_available)
    contributions = {
        "tfidf": weights["tfidf"] * tfidf.similarity * 100,
        "skills": weights["skills"] * coverage * 100,
        "sections": weights["sections"] * section_signal * 100,
    }
    final = sum(contributions.values())
    capped = False
    if must_missing and final > config.must_have_cap:
        final = config.must_have_cap
        capped = True
    if must_missing:
        warnings.append(
            "Missing must-have skill" + ("s" if len(must_missing) > 1 else "") + ": " + ", ".join(must_missing)
            + f". The score is capped at {config.must_have_cap:g}."
        )

    final = round(min(max(final, 0.0), 100.0), 1)
    return {
        "final_score": final,
        "label": label_for(final),
        "tfidf_score": round(tfidf.similarity * 100, 1),
        "skill_coverage": round(coverage * 100, 1),
        "section_signal": round(section_signal * 100, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "must_have_missing": must_missing,
        "skills_in_sections": in_section,
        "required_skills": list(required.skills),
        "top_terms": tfidf.top_terms,
        "warnings": warnings,
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "contributions": {key: round(value, 1) for key, value in contributions.items()},
        "capped": capped,
        "text_chars": text_chars,
        "low_text": low_text,
        "snippets": _snippets(text, matched, occurrences, sections),
        "scoring_version": SCORING_VERSION,
    }


def explanation_text(breakdown: dict) -> str:
    """One-paragraph plain-text explanation (used for CSV export and the legacy ``explanation`` field)."""
    w = breakdown["weights"]
    parts = [
        f"{breakdown['label']}. Score {breakdown['final_score']:g}/100 = "
        f"{w['tfidf'] * 100:.0f}% text similarity ({breakdown['tfidf_score']:g}) + "
        f"{w['skills'] * 100:.0f}% skill coverage ({breakdown['skill_coverage']:g}) + "
        f"{w['sections'] * 100:.0f}% section signal ({breakdown['section_signal']:g})."
    ]
    if breakdown["capped"]:
        parts.append("Capped because a must-have skill is missing.")
    if breakdown["matched_skills"]:
        parts.append("Matched skills: " + ", ".join(breakdown["matched_skills"]) + ".")
    if breakdown["missing_skills"]:
        parts.append("Skills not found: " + ", ".join(breakdown["missing_skills"]) + ".")
    return " ".join(parts)


def rank_candidates(
    job_description: str,
    resumes: Iterable[tuple[str, str]],
    must_have: Iterable[str] = (),
    config: ScoringConfig | None = None,
    taxonomy: Taxonomy | None = None,
) -> list[dict]:
    """Score ``(name, text)`` pairs and return them sorted by score (then name).

    Each item is ``{"name": ..., "rank": n, **breakdown}``. The ranking is
    stable and independent of input order.
    """
    taxonomy = taxonomy or load_taxonomy()
    config = config or ScoringConfig()
    required = required_skills(job_description, must_have, taxonomy)
    results = [
        {"name": name, **score_resume(job_description, text, config=config, taxonomy=taxonomy, required=required)}
        for name, text in resumes
    ]
    results.sort(key=lambda item: (-item["final_score"], -item["tfidf_score"], item["name"]))
    for index, item in enumerate(results, start=1):
        item["rank"] = index
    return results


def config_summary(config: ScoringConfig) -> dict:
    """Serializable view of the configuration (for API responses and docs)."""
    data = asdict(config)
    data["warnings"] = list(config.warnings)
    data["normalised_weights"] = config.normalised_weights()
    return data
