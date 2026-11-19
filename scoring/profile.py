"""Structured profile extraction from normalised resume text.

Rules
-----
* Every value is a **verbatim substring** of the resume text (with its character span); nothing is
  inferred, paraphrased or invented.
* Each group reports a ``status`` that separates "genuinely absent" from "could not be parsed":

  - ``found``           — values extracted;
  - ``section_missing`` — no heading for this group was detected (information may be absent, or the
                          heading uses words we do not recognise);
  - ``unparsed``        — the section exists but no entries could be recognised (extraction limitation).

* Contact details (name, email, phone, location) are extracted for display only. They are **never**
  used as scoring signals (see ``docs/ai/SCORING_METHODOLOGY.md`` §B6).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .sections import Section, detect_sections
from .skills import Taxonomy, load_taxonomy
from .text import normalize_text

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w])(\+?\d[\d ()./-]{7,}\d)(?![\w])")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+)?(?:19|20)\d\d"
_RANGE = re.compile(rf"(?P<start>{_DATE})\s*(?:-|to|until)\s*(?P<end>{_DATE}|present|current|now|today)", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")
_DEGREE = re.compile(
    r"\b(?:B\.?\s?Sc|BSc|B\.?S\.?|B\.?A\.?|BCom|B\.?Com|B\.?Tech|B\.?E\.?|M\.?\s?Sc|MSc|M\.?S\.?|MBA|M\.?Tech|M\.?A\.?|"
    r"Ph\.?\s?D|PhD|Bachelor(?:'s)?(?:\s+of\s+\w+)?|Master(?:'s)?(?:\s+of\s+\w+)?|Diploma|Associate(?:'s)?\s+Degree)\b"
)
_INSTITUTION = re.compile(r"[^,|;()]*\b(?:University|College|Institute|School|Academy|Polytechnic)\b[^,|;]*", re.IGNORECASE)
_LABELLED_LOCATION = re.compile(r"^(?:location|address|based in)\s*[:\-]\s*(?P<value>.{2,80})$", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(?:built|build|developed|develop|implemented|implement|designed|design|created|deployed|engineered|wrote|"
    r"led|architected|automated|migrated|optimi[sz]ed|integrated|trained|shipped|launched|delivered|maintained|"
    r"served|packaged|containeri[sz]ed|analy[sz]ed|ran|reduced|improved|increased)\b",
    re.IGNORECASE,
)
_OUTCOME = re.compile(r"\d+(?:\.\d+)?\s?%|\b\d{2,}\b|\b(?:improved|reduced|increased|cut|saved|grew)\b", re.IGNORECASE)


@dataclass
class Value:
    text: str
    start: int
    end: int


@dataclass
class Group:
    status: str  # found | section_missing | unparsed
    items: list = field(default_factory=list)


def _lines(text: str, start: int, end: int, merge_wrapped: bool = False) -> list[Value]:
    """Non-empty lines with spans. ``merge_wrapped`` joins layout-wrapped continuation lines
    (a line starting with a lower-case letter continues the previous one); the span then covers both."""
    out: list[Value] = []
    offset = start
    for line in text[start:end].split("\n"):
        stripped = line.strip()
        if stripped:
            lead = len(line) - len(line.lstrip())
            begin = offset + lead
            if merge_wrapped and out and stripped[0].islower() and not out[-1].text.endswith((".", ":")):
                prev = out[-1]
                out[-1] = Value(text[prev.start:begin + len(stripped)].replace("\n", " "), prev.start, begin + len(stripped))
            else:
                out.append(Value(stripped, begin, begin + len(stripped)))
        offset += len(line) + 1
    return out


def _clean_head(head: str) -> str:
    head = head.strip(" ,|-–—")
    return re.sub(r"\(\s*\)", "", head).strip(" ,|-–—")


def _sections_by_label(sections: list[Section], label: str) -> list[Section]:
    return [s for s in sections if (s.label or s.kind) == label]


def _looks_like_name(line: str) -> bool:
    words = line.split()
    if not 2 <= len(words) <= 4 or any(ch.isdigit() for ch in line) or "@" in line:
        return False
    return all(re.fullmatch(r"[A-Z][a-zA-Z'.-]+", w) for w in words)


def _contact(text: str, sections: list[Section]) -> dict:
    head_end = sections[0].start if sections else min(len(text), 400)
    head = _lines(text, 0, head_end)
    name = next((v for v in head[:3] if _looks_like_name(v.text)), None)
    email = _EMAIL.search(text)
    phone = next((m for m in _PHONE.finditer(text) if sum(ch.isdigit() for ch in m.group(1)) >= 8 and not _RANGE.search(m.group(1))), None)
    location = None
    for v in _lines(text, 0, len(text)):
        m = _LABELLED_LOCATION.match(v.text)
        if m:
            location = Value(m.group("value").strip(), v.start + m.start("value"), v.start + m.end("value"))
            break
    return {
        "name": asdict(name) if name else None,
        "email": {"text": email.group(0), "start": email.start(), "end": email.end()} if email else None,
        "phone": {"text": phone.group(1), "start": phone.start(1), "end": phone.end(1)} if phone else None,
        "location": asdict(location) if location else None,
        "used_for_scoring": False,
    }


def _skills_in(text: str, taxonomy: Taxonomy) -> list[str]:
    return list(taxonomy.found_skills(text))


def _experience(text: str, sections: list[Section], taxonomy: Taxonomy) -> Group:
    secs = [s for s in sections if s.kind == "experience"]
    if not secs:
        return Group("section_missing")
    entries = []
    for sec in secs:
        lines = _lines(text, sec.start, sec.end, merge_wrapped=True)
        current = None
        for v in lines:
            m = _RANGE.search(v.text)
            if m:
                head = _clean_head(v.text[: m.start()] + v.text[m.end():])
                parts = [p.strip() for p in re.split(r"\s*(?:,|\||\bat\b|\s-\s|—|–)\s*", head) if p.strip()]
                current = {
                    "header": asdict(v),
                    "title": parts[0] if parts else None,
                    "company": parts[1] if len(parts) > 1 else None,
                    "dates": {"start": m.group("start"), "end": m.group("end"), "raw": m.group(0)},
                    "lines": [],
                }
                entries.append(current)
            elif current is not None:
                current["lines"].append(asdict(v))
            else:
                # Description before any dated header: keep as an undated entry rather than dropping it.
                current = {"header": None, "title": None, "company": None, "dates": None, "lines": [asdict(v)]}
                entries.append(current)
    for entry in entries:
        body = "\n".join(line["text"] for line in entry["lines"])
        full = ((entry["header"] or {}).get("text", "") + "\n" + body).strip()
        entry["technologies"] = _skills_in(full, taxonomy)
        entry["achievements"] = [line for line in entry["lines"] if _OUTCOME.search(line["text"])]
        entry["has_implementation_detail"] = any(_ACTION.search(line["text"]) for line in entry["lines"])
    return Group("found" if entries else "unparsed", entries)


def _education(text: str, sections: list[Section]) -> Group:
    secs = _sections_by_label(sections, "education")
    if not secs:
        return Group("section_missing")
    items = []
    for sec in secs:
        for v in _lines(text, sec.start, sec.end):
            degree = _DEGREE.search(v.text)
            institution = _INSTITUTION.search(v.text)
            if not (degree or institution):
                continue
            years = _YEAR.findall(v.text)
            field_match = re.search(r"\b(?:in|of)\s+([A-Z][A-Za-z &/-]{2,60}?)(?=,|\||\(|$| at )", v.text)
            items.append({
                "raw": asdict(v),
                "degree": degree.group(0) if degree else None,
                "field": field_match.group(1).strip() if field_match else None,
                "institution": institution.group(0).strip() if institution else None,
                "graduation_year": years[-1] if years else None,
            })
    return Group("found" if items else "unparsed", items)


def _projects(text: str, sections: list[Section], taxonomy: Taxonomy) -> Group:
    secs = [s for s in sections if s.kind == "projects"]
    if not secs:
        return Group("section_missing")
    items = []
    for sec in secs:
        lines = _lines(text, sec.start, sec.end, merge_wrapped=True)
        i = 0
        while i < len(lines):
            v = lines[i]
            title, desc = None, v.text
            if ":" in v.text and len(v.text.split(":", 1)[0]) <= 60:
                title, desc = (part.strip() for part in v.text.split(":", 1))
            elif len(v.text) <= 60 and not v.text.endswith(".") and i + 1 < len(lines):
                title, desc = v.text, lines[i + 1].text
                i += 1
            items.append({
                "raw": asdict(v),
                "title": title,
                "description": desc,
                "technologies": _skills_in(v.text + "\n" + desc, taxonomy),
                "has_implementation_detail": bool(_ACTION.search(desc)) and bool(_skills_in(desc, taxonomy)),
                "has_outcome": bool(_OUTCOME.search(desc)),
            })
            i += 1
    return Group("found" if items else "unparsed", items)


def _simple(text: str, sections: list[Section], label: str) -> Group:
    secs = _sections_by_label(sections, label)
    if not secs:
        return Group("section_missing")
    items = [asdict(v) for s in secs for v in _lines(text, s.start, s.end)]
    return Group("found" if items else "unparsed", items)


def extract_profile(text: str, taxonomy: Taxonomy | None = None, sections: list[Section] | None = None) -> dict:
    """Return the structured profile of a resume (see module docstring for the status semantics)."""
    taxonomy = taxonomy or load_taxonomy()
    text = normalize_text(text) if "\r" in text else text
    sections = detect_sections(text) if sections is None else sections
    summary = _simple(text, sections, "summary")
    skills_found = taxonomy.found_skills(text)
    by_category: dict[str, list[str]] = {}
    for canonical in skills_found:
        category = taxonomy.skills[canonical].category
        by_category.setdefault(category, []).append(canonical)
    return {
        "contact": _contact(text, sections),
        "summary": asdict(summary),
        "technical_profile": {"status": "found" if skills_found else "not_found", "by_category": by_category},
        "experience": asdict(_experience(text, sections, taxonomy)),
        "education": asdict(_education(text, sections)),
        "projects": asdict(_projects(text, sections, taxonomy)),
        "certifications": asdict(_simple(text, sections, "certifications")),
        "publications": asdict(_simple(text, sections, "publications")),
        "achievements": asdict(_simple(text, sections, "achievements")),
        "sections_detected": [{"heading": s.heading, "kind": s.kind, "label": s.label or s.kind} for s in sections],
    }
