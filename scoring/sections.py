"""Very small resume section detector.

A *heading* is a short line (at most 48 characters, at most 5 words) whose
text, after stripping decoration such as ``#``, ``:``, ``-`` or ``|``, is one
of the known heading phrases below. ``Skills: Python, SQL`` also counts: the
part before the colon is the heading and the rest of the line belongs to the
section.

Only three section kinds earn the section signal: ``skills``, ``experience``
and ``projects``. Other headings (education, summary, ...) end the previous
section. Text before the first heading belongs to no section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SIGNAL_SECTIONS = ("skills", "experience", "projects")

HEADINGS: dict[str, tuple[str, ...]] = {
    "skills": (
        "skills", "technical skills", "core skills", "key skills", "skill set", "skillset",
        "skills summary", "core competencies", "competencies", "technologies", "tech stack",
        "tools", "tools and technologies", "tools & technologies", "technical proficiencies",
        "areas of expertise", "expertise",
    ),
    "experience": (
        "experience", "work experience", "professional experience", "employment",
        "employment history", "work history", "career history", "relevant experience",
        "internships", "internship", "internship experience", "professional background",
    ),
    "projects": (
        "projects", "personal projects", "academic projects", "key projects", "selected projects",
        "side projects", "project experience", "open source", "open-source contributions",
        "portfolio",
    ),
    "other": (
        "education", "academic background", "summary", "professional summary", "profile",
        "objective", "career objective", "about", "about me", "certifications", "certificates",
        "licenses and certifications", "awards", "achievements", "honors", "honours",
        "publications", "interests", "hobbies", "languages", "references", "contact",
        "contact information", "volunteering", "volunteer experience", "activities",
        "extracurricular activities", "courses", "coursework", "relevant coursework",
        "leadership", "declaration", "personal details",
    ),
}

_LOOKUP = {phrase: kind for kind, phrases in HEADINGS.items() for phrase in phrases}

# Finer labels for the "other" headings (used by the profile extractor; ``kind`` stays unchanged).
LABELS: dict[str, tuple[str, ...]] = {
    "education": ("education", "academic background", "courses", "coursework", "relevant coursework"),
    "summary": ("summary", "professional summary", "profile", "objective", "career objective", "about", "about me"),
    "certifications": ("certifications", "certificates", "licenses and certifications"),
    "achievements": ("awards", "achievements", "honors", "honours"),
    "publications": ("publications",),
    "contact": ("contact", "contact information", "personal details"),
}
_LABEL_LOOKUP = {phrase: label for label, phrases in LABELS.items() for phrase in phrases}
_DECORATION = re.compile(r"^[\s#*=_\-|:>•]+|[\s#*=_\-|:>•]+$")


@dataclass(frozen=True)
class Section:
    """A detected section: ``kind`` plus [start, end) character offsets in the text.

    ``label`` is a finer category (education, summary, certifications, …) for "other" sections and
    equals ``kind`` for skills / experience / projects.
    """

    kind: str
    heading: str
    start: int
    end: int
    label: str = ""


def _clean_heading(candidate: str) -> str | None:
    cleaned = _DECORATION.sub("", candidate).lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or len(cleaned) > 48 or len(cleaned.split()) > 5:
        return None
    return cleaned


def _heading_kind(candidate: str) -> str | None:
    cleaned = _clean_heading(candidate)
    return _LOOKUP.get(cleaned) if cleaned else None


def heading_label(candidate: str) -> str:
    cleaned = _clean_heading(candidate) or ""
    kind = _LOOKUP.get(cleaned, "")
    return kind if kind != "other" else _LABEL_LOOKUP.get(cleaned, "other")


def detect_sections(text: str) -> list[Section]:
    """Return sections found in normalised ``text`` in document order."""
    found: list[tuple[str, str, int, int]] = []  # (kind, heading, line_start, content_start)
    offset = 0
    for line in text.split("\n"):
        line_start = offset
        offset += len(line) + 1
        kind = _heading_kind(line)
        if kind:
            found.append((kind, line.strip(), line_start, line_start + len(line)))
            continue
        if ":" in line:
            head, _, _rest = line.partition(":")
            kind = _heading_kind(head)
            if kind:
                found.append((kind, head.strip(), line_start, line_start + len(head) + 1))
    sections = []
    for index, (kind, heading, _line_start, content_start) in enumerate(found):
        end = found[index + 1][2] if index + 1 < len(found) else len(text)
        sections.append(Section(kind, heading, content_start, max(end, content_start), heading_label(heading)))
    return sections


def section_at(position: int, sections: list[Section]) -> Section | None:
    for section in sections:
        if section.start <= position < section.end:
            return section
    return None


def in_signal_section(position: int, sections: list[Section]) -> bool:
    """True when ``position`` falls inside a skills/experience/projects section."""
    return any(s.kind in SIGNAL_SECTIONS and s.start <= position < s.end for s in sections)
