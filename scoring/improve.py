"""Deterministic resume-improvement suggestions.

Principles:
* Every finding quotes the exact resume text it refers to (``original`` = text[start:end]).
* Suggestions are **recommendations**, never presented as resume content. Rewrites keep only facts that are
  already in the original line and use ``[placeholders]`` for anything the candidate must supply. Nothing is
  invented: no metrics, user counts, achievements or skills.
* Missing job skills are phrased as "if you have used X, describe how", never as a claim.
"""

from __future__ import annotations

import re

from .context import ACTION_VERBS
from .sections import detect_sections
from .text import normalize_text

WEAK_STARTS = re.compile(
    r"^(?P<weak>worked\s+on|helped(?:\s+with)?|responsible\s+for|was\s+involved\s+in|involved\s+in|assisted(?:\s+with)?|"
    r"participated\s+in|tasked\s+with|duties\s+included|did)\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
CLICHES = ("hard-working", "hardworking", "team player", "motivated", "passionate", "results-driven", "detail-oriented",
           "go-getter", "self-starter", "think outside the box", "dynamic", "synergy", "fast learner")
OUTCOME = re.compile(r"\d+(?:\.\d+)?\s?%|\b\d{2,}\b|\b(?:reduced|increased|improved|saved|grew|cut)\b", re.IGNORECASE)
DATE_STYLES = {
    "month-year": re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d\d", re.IGNORECASE),
    "year-only": re.compile(r"(?<![a-z]\s)\b(?:19|20)\d\d\s*(?:-|–|to)\s*(?:(?:19|20)\d\d|present|current)", re.IGNORECASE),
}
STRONG_VERBS = "Built, Developed, Designed, Implemented, Led, Automated, Deployed, Maintained"


def _lines(text: str, start: int, end: int):
    offset = start
    for line in text[start:end].split("\n"):
        stripped = line.strip()
        if stripped:
            lead = len(line) - len(line.lstrip())
            yield stripped, offset + lead, offset + lead + len(stripped)
        offset += len(line) + 1


def _finding(kind, severity, message, original=None, suggestion=None):
    return {"type": kind, "severity": severity, "message": message, "original": original, "suggestion": suggestion,
            "is_recommendation": True}


def analyze_resume(text: str, requirement_results: list | None = None, max_findings: int = 25) -> dict:
    text = normalize_text(text)
    sections = detect_sections(text)
    findings: list[dict] = []
    kinds = {s.kind for s in sections}
    labels = {s.label or s.kind for s in sections}

    # Structure
    missing = [name for name, key in (("Skills", "skills"), ("Experience", "experience")) if key not in kinds]
    if missing:
        findings.append(_finding("structure", "high", f"No {' or '.join(missing)} heading was detected. Clear headings help "
                                 "reviewers and parsers find your evidence.", suggestion="Add short headings such as 'Skills', "
                                 "'Experience' and 'Projects' on their own lines."))

    for section in sections:
        if section.kind not in {"experience", "projects"}:
            continue
        styles = set()
        for line, start, end in _lines(text, section.start, section.end):
            original = {"text": line, "start": start, "end": end}
            for style, pattern in DATE_STYLES.items():
                if pattern.search(line):
                    styles.add(style)
            weak = WEAK_STARTS.match(line)
            if weak and len(line.split()) >= 3:
                rest = weak.group("rest").rstrip(".")
                findings.append(_finding("weak_verb", "medium", f"“{weak.group('weak')}” hides what you actually did.", original,
                                         f"[{STRONG_VERBS.split(', ')[0]}/{STRONG_VERBS.split(', ')[3]}/…] {rest} — "
                                         "[what it does], [your specific contribution]. Choose the verb that is true."))
                continue
            words = len(line.split())
            if section.kind == "projects" and words >= 3 and not ACTION_VERBS.search(line):
                findings.append(_finding("weak_project", "medium", "This project line names the project but not what you built or how.",
                                         original, f"{line.rstrip('.')} — [what you built], [technologies you actually used], "
                                         "[a result, only if you can support it]."))
            elif section.kind == "experience" and ACTION_VERBS.search(line) and not OUTCOME.search(line) and words >= 6:
                findings.append(_finding("missing_outcome", "low", "This achievement has no visible outcome.", original,
                                         f"{line.rstrip('.')}, [which led to … — add a measurable result only if you have one]."))
            if words > 40:
                findings.append(_finding("long_line", "low", "Very long line; reviewers skim.", original,
                                         "Split into two or three bullet points, each starting with a verb."))
        if len(styles) > 1:
            findings.append(_finding("inconsistent_dates", "low", f"Mixed date formats in the {section.heading} section.",
                                     suggestion="Use one format throughout, e.g. 'Jan 2022 – Present'."))

    for section in sections:
        if (section.label or section.kind) != "summary":
            continue
        body = text[section.start:section.end]
        hits = [c for c in CLICHES if re.search(rf"\b{re.escape(c)}\b", body, re.IGNORECASE)]
        if hits:
            first = re.search(rf"\b{re.escape(hits[0])}\b", body, re.IGNORECASE)
            start = section.start + first.start()
            findings.append(_finding("generic_summary", "medium", f"Generic phrases ({', '.join(hits)}) don't tell a reviewer much.",
                                     {"text": text[start:start + len(first.group(0))], "start": start, "end": start + len(first.group(0))},
                                     "Replace with your focus and evidence: '[role] who [what you build] with [technologies you use]'."))
    if "summary" not in labels and len(text) > 400:
        findings.append(_finding("missing_summary", "low", "No summary section was detected.",
                                 suggestion="Optional: add a two-line summary of your focus, using only facts from your experience."))

    for result in requirement_results or []:
        status = result["status"] if isinstance(result, dict) else result.status
        label = result["label"] if isinstance(result, dict) else result.label
        cls = result["cls"] if isinstance(result, dict) else result.cls
        if status in {"not_found", "inferred"} and cls != "preferred":
            findings.append(_finding("missing_skill", "medium" if cls == "mandatory" else "low",
                                     f"The job asks for {label}, but the resume does not demonstrate it.",
                                     suggestion=f"If you have genuinely used {label}, add one sentence describing what you built with it. "
                                                "If you have not, do not add it."))
        elif status == "unclear":
            findings.append(_finding("unclear_skill", "low", f"{label} is listed but its use is not described.",
                                     suggestion=f"If accurate, describe one piece of work where you used {label}."))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {
        "findings": findings[:max_findings],
        "total_findings": len(findings),
        "disclaimer": "Suggestions are recommendations generated from rules. They are not part of your resume. "
                      "Only add information that is true and that you can support.",
    }
