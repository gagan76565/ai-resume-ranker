"""Scoring engine v3: evidence-based, hybrid (lexical + optional semantic), explainable.

``fit = 100 × Σ wᵢ·cᵢ / Σ wᵢ`` over the *available* components (each cᵢ ∈ [0, 1]):

| key        | default | component |
|------------|--------:|-----------|
| required   | 0.30 | mean evidence strength over mandatory + general requirements |
| semantic   | 0.25 | mean best semantic match of JD requirement sentences in Experience/Projects/Summary (optional) |
| experience | 0.15 | mean evidence strength of non-preferred requirements *inside Experience* |
| projects   | 0.15 | mean evidence strength of *all* requirements inside Projects; only described use counts fully (0.3 otherwise) |
| preferred  | 0.10 | mean evidence strength over preferred requirements |
| lexical    | 0.05 | TF-IDF cosine of JD and resume (the v2 text-similarity signal) |

Unavailable components are excluded and reported, never silently scored as zero (except Experience and
Projects when *both* sections are missing: then there is no contextual evidence and both count as 0).
Mandatory requirements are reported separately (coverage, missing, met); by default they do **not** cap
the fit score (``MANDATORY_POLICY=flag``); ``cap`` restores the v2 behaviour. Weights are design choices,
not validated constants. Full methodology: docs/ai/SCORING_METHODOLOGY.md.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from . import semantic as semantic_mod
from .confidence import evaluation_confidence
from .engine import label_for
from .evidence import evaluate_requirements
from .job import analyze_job
from .ontology import load_ontology
from .profile import extract_profile
from .redact import injection_suspected, redact_for_scoring
from .sections import SIGNAL_SECTIONS, detect_sections
from .skills import load_taxonomy
from .text import normalize_text
from .tfidf import tfidf_similarity

SCORING_VERSION = "3.0"
DEFAULT_WEIGHTS_V3 = {"required": 0.30, "semantic": 0.25, "experience": 0.15, "projects": 0.15, "preferred": 0.10, "lexical": 0.05}
COMPONENT_LABELS = {
    "required": "Required-skill evidence", "semantic": "Semantic relevance", "experience": "Relevant experience",
    "projects": "Project relevance", "preferred": "Preferred skills", "lexical": "Lexical relevance (TF-IDF)",
}
PROJECT_LISTED_STRENGTH = 0.3


@dataclass(frozen=True)
class ConfigV3:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS_V3))
    mandatory_policy: str = "flag"  # flag | cap
    must_have_cap: float = 40.0
    semantic_max_sentences: int = 400  # per request; above this semantic is skipped for the whole batch
    max_resume_sentences: int = 30
    max_jd_sentences: int = 15
    problems: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, overrides: Mapping[str, float] | None = None) -> ConfigV3:
        env = os.environ if env is None else env
        problems: list[str] = []
        weights = dict(DEFAULT_WEIGHTS_V3)
        for key in weights:
            raw = env.get(f"V3_WEIGHT_{key.upper()}")
            if raw not in (None, ""):
                try:
                    value = float(raw)
                    if math.isnan(value) or value < 0:
                        raise ValueError
                    weights[key] = value
                except ValueError:
                    problems.append(f"V3_WEIGHT_{key.upper()}={raw!r} is invalid; using {weights[key]}.")
        for key, value in (overrides or {}).items():
            if key in weights and isinstance(value, (int, float)) and value >= 0:
                weights[key] = float(value)
        if sum(weights.values()) <= 0:
            problems.append("All v3 weights are zero; using defaults.")
            weights = dict(DEFAULT_WEIGHTS_V3)
        policy = str(env.get("MANDATORY_POLICY", "flag")).strip().lower()
        if policy not in {"flag", "cap"}:
            problems.append(f"MANDATORY_POLICY={policy!r} is unknown; using 'flag'.")
            policy = "flag"
        try:
            cap = float(env.get("MUST_HAVE_CAP", 40) or 40)
        except ValueError:
            cap = 40.0
        try:
            budget = int(env.get("SEMANTIC_MAX_SENTENCES", 400) or 400)
        except ValueError:
            budget = 400
        return cls(weights=weights, mandatory_policy=policy, must_have_cap=cap, semantic_max_sentences=budget, problems=tuple(problems))


@dataclass
class ResumeInput:
    name: str
    text: str
    page_at: Callable[[int], int | None] | None = None
    extraction_score: float | None = None  # from document quality; None → derived from text length
    quality: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _jd_queries(job_text: str, limit: int) -> list[str]:
    seen, out = set(), []
    lines = job_text.split("\n")
    if len(lines) > 3:
        lines = lines[1:]  # the first line of a multi-line JD is normally the job title, not a requirement
    for line in lines:
        line = line.strip(" -•*\t")
        if len(line) >= 25 and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
    return out[:limit]


def _resume_sentences(text: str, sections, limit: int) -> list[tuple[str, int, int, str]]:
    wanted = [s for s in sections if (s.label or s.kind) in {"experience", "projects", "summary"}]
    spans = [(s.start, s.end, s.label or s.kind) for s in wanted] or [(0, len(text), "none")]
    out = []
    for start, end, label in spans:
        for sentence, s_start, s_end in semantic_mod.sentences(text, start, end):
            stripped_start = text.find(sentence, s_start, s_end)
            out.append((sentence, stripped_start, stripped_start + len(sentence), label))
    return out[:limit]


def _snippets(text: str, results, skill_spans: dict, limit: int = 6) -> list[dict]:
    """v2-compatible snippet objects built from evidence quotes (exact substrings, marked matches)."""
    covered_spans = sorted({(s, e) for r in results if r.covered for skill in r.skills for s, e, _ in skill_spans.get(skill, [])})
    snippets, used = [], []
    for r in results:
        if not r.covered or not r.evidence:
            continue
        item = r.evidence[0]
        if any(item.start < u_end and item.end > u_start for u_start, u_end in used):
            continue
        used.append((item.start, item.end))
        segments, cursor = [], item.start
        for s, e in covered_spans:
            if s < cursor or e > item.end or s < item.start:
                continue
            if s > cursor:
                segments.append({"text": text[cursor:s], "mark": False})
            segments.append({"text": text[s:e], "mark": True})
            cursor = e
        if cursor < item.end:
            segments.append({"text": text[cursor:item.end], "mark": False})
        snippets.append({"skill": r.label, "prefix_ellipsis": False, "suffix_ellipsis": False, "segments": segments,
                         "section": item.section, "page": item.page})
        if len(snippets) >= limit:
            break
    return snippets


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_resume_v3(job_text: str, requirements, resume: ResumeInput, config: ConfigV3, encoder=None,
                    jd_queries: list[str] | None = None, taxonomy=None, ontology=None) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    ontology = ontology or load_ontology()
    text = normalize_text(resume.text)
    sections = detect_sections(text)
    profile = extract_profile(text, taxonomy, sections)
    scoring_text = redact_for_scoring(text, profile)  # same length/offsets; contact + protected lines blanked
    skill_spans: dict[str, list[tuple[int, int, str]]] = {}
    for m in taxonomy.find(text):
        skill_spans.setdefault(m.canonical, []).append((m.start, m.end, m.surface))
    results = evaluate_requirements(text, requirements, sections, skill_spans, page_at=resume.page_at, ontology=ontology)
    core = [r for r in results if r.cls != "preferred"]
    preferred = [r for r in results if r.cls == "preferred"]
    mandatory = [r for r in results if r.cls == "mandatory"]
    has_exp = any(s.kind == "experience" for s in sections)
    has_proj = any(s.kind == "projects" for s in sections)

    def located(section: str, listed_strength: float | None = None, pool=None) -> float | None:
        pool = core if pool is None else pool
        if not pool:
            return None
        scores = []
        for r in pool:
            best = 0.0
            for item in r.evidence:
                if item.section == section:
                    strength = item.strength if (listed_strength is None or item.type == "demonstrated") else listed_strength
                    best = max(best, strength)
            scores.append(best)
        return _mean(scores)

    components: dict[str, dict] = {}

    def put(key: str, value: float | None, note: str = "") -> None:
        components[key] = {"label": COMPONENT_LABELS[key], "value": None if value is None else round(value * 100, 1),
                           "available": value is not None, "note": note, "_raw": value}

    put("required", _mean([r.strength for r in core]), "" if core else "The job description has no mandatory or general skill requirements.")
    put("preferred", _mean([r.strength for r in preferred]), "" if preferred else "The job description lists no preferred skills.")
    if has_exp or not has_proj:
        put("experience", located("experience") if has_exp else 0.0, "" if has_exp else "No Experience section was detected.")
    else:
        put("experience", None, "No Experience section; weight redistributed because Projects are present.")
    if has_proj or not has_exp:
        put("projects", located("projects", PROJECT_LISTED_STRENGTH, results) if has_proj else 0.0, "" if has_proj else "No Projects section was detected.")
    else:
        put("projects", None, "No Projects section; weight redistributed because Experience is present.")
    if core and components["experience"]["available"] and components["experience"]["_raw"] is None:
        components["experience"].update({"value": 0.0, "_raw": 0.0})

    tf = tfidf_similarity(job_text, scoring_text)
    put("lexical", tf.similarity)

    semantic_matches: list[dict] = []
    if encoder is not None and jd_queries:
        candidates = _resume_sentences(scoring_text, sections, config.max_resume_sentences)
        semantic_matches = semantic_mod.match(encoder, jd_queries, candidates)
        for item in semantic_matches:
            if item["best"]:
                best = item["best"]
                best["text"] = text[best["start"]:best["end"]]  # quote the original, never the redacted copy
                best["page"] = resume.page_at(best["start"]) if resume.page_at else None
        put("semantic", _mean([m["score"] for m in semantic_matches]) if candidates else 0.0,
            "" if candidates else "No sentences were available for semantic comparison.")
    else:
        put("semantic", None, "Semantic matching is unavailable or was skipped; its weight is redistributed.")

    avail = {k: config.weights.get(k, 0.0) for k, c in components.items() if c["available"]}
    total_w = sum(avail.values()) or 1.0
    weights = {k: (avail.get(k, 0.0) / total_w) for k in DEFAULT_WEIGHTS_V3}
    fit = 0.0
    for key, comp in components.items():
        comp["weight"] = round(weights[key], 4)
        contribution = weights[key] * (comp["_raw"] or 0.0) * 100 if comp["available"] else 0.0
        comp["contribution"] = round(contribution, 1)
        fit += contribution
        comp.pop("_raw")
    fit = max(0.0, min(100.0, fit))
    evidence_fit = round(fit, 1)

    missing_mandatory = [r.label for r in mandatory if not r.covered]
    capped = False
    if missing_mandatory and config.mandatory_policy == "cap" and fit > config.must_have_cap:
        fit, capped = config.must_have_cap, True
    fit = round(fit, 1)

    extraction = resume.extraction_score
    if extraction is None:
        extraction = 1.0 if len(text) >= 200 else 0.4
    confidence = evaluation_confidence(extraction, sections, results, encoder is not None and bool(semantic_matches))

    warnings = list(resume.warnings)
    if resume.quality and resume.quality.get("grade") not in (None, "excellent"):
        warnings.extend(resume.quality.get("reasons", []))
    if missing_mandatory:
        warnings.append("Missing mandatory requirement" + ("s" if len(missing_mandatory) > 1 else "") + ": " +
                        ", ".join(missing_mandatory) + " (not demonstrated in the resume)." +
                        (f" Score capped at {config.must_have_cap:g}." if capped else ""))
    if not any(s.kind in SIGNAL_SECTIONS for s in sections):
        warnings.append("No Skills, Experience or Projects headings were detected.")
    if not results:
        warnings.append("No known skills were found in the job description; scoring relies on text and semantic similarity.")
    suspicious = injection_suspected(text)
    if suspicious:
        warnings.append("The resume contains text that looks like instructions to an AI system. It was ignored: scoring is "
                        "deterministic and is not influenced by instructions in resumes.")

    covered = [r for r in results if r.covered]
    signal = [r for r in covered if r.evidence and r.evidence[0].section in SIGNAL_SECTIONS]
    return {
        # v2-compatible keys (meanings documented in docs/ai/SCORING_METHODOLOGY.md)
        "final_score": fit,
        "label": label_for(fit),
        "tfidf_score": round(tf.similarity * 100, 1),
        "skill_coverage": round(100 * len([r for r in core if r.covered]) / len(core), 1) if core else 0.0,
        "section_signal": round(100 * len(signal) / len(covered), 1) if covered else 0.0,
        "matched_skills": [r.label for r in covered],
        "missing_skills": [r.label for r in results if not r.covered],
        "must_have_missing": missing_mandatory,
        "skills_in_sections": [r.label for r in signal],
        "required_skills": [r.label for r in results],
        "top_terms": tf.top_terms,
        "warnings": warnings,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "contributions": {k: c["contribution"] for k, c in components.items()},
        "capped": capped,
        "text_chars": len(text),
        "low_text": len(text) < 200,
        "snippets": _snippets(text, results, skill_spans),
        "scoring_version": SCORING_VERSION,
        # v3 additions
        "engine": "v3",
        "fit_score": fit,
        "evidence_fit": evidence_fit,
        "components": components,
        "requirements": [r.to_dict() for r in results],
        "mandatory": {"total": len(mandatory), "covered": len(mandatory) - len(missing_mandatory),
                      "coverage": round(100 * (len(mandatory) - len(missing_mandatory)) / len(mandatory), 1) if mandatory else None,
                      "missing": missing_mandatory, "met": not missing_mandatory, "policy": config.mandatory_policy},
        "confidence": confidence,
        "semantic": {"used": bool(semantic_matches), "matches": semantic_matches},
        "quality": resume.quality,
        "injection_suspected": suspicious,
        "profile": profile,
    }


def rank_candidates_v3(job_description: str, resumes: Iterable[ResumeInput], must_have: Iterable[str] = (),
                       config: ConfigV3 | None = None, encoder: object | None = "auto") -> tuple[list[dict], dict]:
    """Score and rank resumes. Returns (ranked breakdowns with "name"/"rank", run metadata)."""
    config = config or ConfigV3.from_env()
    taxonomy, ontology = load_taxonomy(), load_ontology()
    job_text = normalize_text(job_description)
    requirements = analyze_job(job_description, list(must_have), taxonomy)
    resumes = list(resumes)
    if encoder == "auto":
        encoder = semantic_mod.get_encoder()
    queries = _jd_queries(job_text, config.max_jd_sentences)
    semantic_note = None
    if encoder is not None:
        projected = len(queries) + len(resumes) * config.max_resume_sentences
        if projected > config.semantic_max_sentences and len(resumes) * config.max_resume_sentences > config.semantic_max_sentences:
            semantic_note = (f"Semantic matching was skipped for this batch ({len(resumes)} resumes exceed the "
                             f"{config.semantic_max_sentences}-sentence budget); all candidates use lexical evidence only.")
            encoder = None
    started = time.perf_counter()
    results = []
    for resume in resumes:
        item = score_resume_v3(job_text, requirements, resume, config, encoder, queries, taxonomy, ontology)
        if semantic_note:
            item["warnings"].append(semantic_note)
        results.append({"name": resume.name, **item})
    results.sort(key=lambda r: (not r["mandatory"]["met"], -r["final_score"], -r["tfidf_score"], r["name"]))
    for index, item in enumerate(results, start=1):
        item["rank"] = index
    meta = {
        "engine": "v3", "scoring_version": SCORING_VERSION,
        "requirements": [r.to_dict() for r in requirements],
        "weights_configured": dict(config.weights), "mandatory_policy": config.mandatory_policy,
        "semantic": {"used": encoder is not None, "note": semantic_note, **semantic_mod.status()},
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "config_problems": list(config.problems),
    }
    return results, meta
