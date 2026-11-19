"""Evidence engine: one traceable verdict per job requirement.

Statuses (strength in brackets):

* ``supported`` (1.0)  — a clause describes *using* the skill (action verb), outside a Skills list;
* ``unclear``  (0.6/0.3) — the skill is mentioned (Skills list, or outside recognised sections) but how it was
  used is not established;
* ``inferred`` (0.25)  — not mentioned; implied by another skill the resume mentions (PostgreSQL ⇒ SQL). Unverified;
* ``negated``  (0)     — the resume itself says the experience is absent ("no experience with Java");
* ``not_found`` (0)    — no mention: "not demonstrated in the resume", which is **not** the same as
  "the candidate does not have this skill".

Every evidence item's ``quote`` is ``text[start:end]`` of the extracted resume text — copied, never generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .context import Mention, classify
from .job import Requirement
from .ontology import Ontology, load_ontology
from .sections import Section
from .skills import custom_skill_pattern

STATUS_ORDER = {"supported": 4, "unclear": 3, "inferred": 2, "negated": 1, "not_found": 0}
INFERRED_STRENGTH = 0.25
SECTION_NAMES = {"skills": "Skills", "experience": "Experience", "projects": "Projects", "summary": "Summary",
                 "education": "Education", "certifications": "Certifications", "none": "an unlabelled part of the resume"}


@dataclass
class EvidenceItem:
    skill: str
    quote: str
    start: int
    end: int
    match_start: int
    match_end: int
    surface: str
    match_kind: str  # exact | alias
    section: str
    page: int | None
    type: str  # demonstrated | listed | bare | negated
    strength: float


@dataclass
class RequirementResult:
    id: str
    label: str
    cls: str
    skills: list[str]
    status: str
    strength: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    related_found: list[str] = field(default_factory=list)
    inferred_from: list[str] = field(default_factory=list)
    explanation: str = ""
    source: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def covered(self) -> bool:
        """Directly mentioned (supported or unclear). Inferred, negated and not-found are not covered."""
        return self.status in {"supported", "unclear"}


def _mentions_for(skill: str, occurrences: dict, text: str, sections: list[Section], custom: bool) -> list[tuple[Mention, str]]:
    spans = occurrences.get(skill, [])
    if custom and not spans:
        spans = [(m.start(), m.end(), m.group(0)) for m in custom_skill_pattern(skill).finditer(text)]
    return [(classify(text, s, e, sections), surface) for s, e, surface in spans]


def _explain(result: RequirementResult) -> str:
    if result.status == "supported":
        item = result.evidence[0]
        where = SECTION_NAMES.get(item.section, item.section.title())
        return f"The {where} section describes work using {item.skill}."
    if result.status == "unclear":
        item = result.evidence[0]
        if item.section == "skills":
            return f"{item.skill} is listed in the Skills section, but the resume does not show how it was used."
        where = SECTION_NAMES.get(item.section, item.section)
        return f"{item.skill} is mentioned in {where if item.section == 'none' else 'the ' + where + ' section'}, but how it was used is not established."
    if result.status == "inferred":
        return (f"Not mentioned directly. It may be implied by {', '.join(result.inferred_from)}, "
                "but this is an unverified inference.")
    if result.status == "negated":
        return "The resume states that this experience is absent."
    text = "No supporting evidence for this requirement was identified in the resume (not demonstrated ≠ not possessed)."
    if result.related_found:
        text += f" Related, non-equivalent experience: {', '.join(result.related_found)}."
    return text


def evaluate_requirements(text: str, requirements: list[Requirement], sections: list[Section], skill_spans: dict,
                          page_at=None, ontology: Ontology | None = None, max_evidence: int = 3) -> list[RequirementResult]:
    """Return one :class:`RequirementResult` per requirement.

    ``skill_spans`` maps canonical skill → list of (start, end, surface) in ``text``.
    ``page_at(offset)`` returns the 1-based page for an offset (or None).
    """
    ontology = ontology or load_ontology()
    present = set(skill_spans)
    results: list[RequirementResult] = []
    for req in requirements:
        items: list[EvidenceItem] = []
        for skill in req.skills:
            for mention, surface in _mentions_for(skill, skill_spans, text, sections, req.custom):
                items.append(EvidenceItem(
                    skill=skill, quote=text[mention.clause_start:mention.clause_end], start=mention.clause_start,
                    end=mention.clause_end, match_start=mention.start, match_end=mention.end, surface=surface,
                    match_kind="exact" if surface.lower() == skill.lower() else "alias", section=mention.section,
                    page=page_at(mention.start) if page_at else None, type=mention.kind, strength=mention.strength,
                ))
        positive = [i for i in items if i.type != "negated"]
        positive.sort(key=lambda i: (-i.strength, i.start))
        result = RequirementResult(req.id, req.label, req.cls, list(req.skills), "not_found", 0.0, source=req.source)
        if positive:
            best = positive[0]
            result.status = "supported" if best.type == "demonstrated" else "unclear"
            result.strength = best.strength
            result.evidence = positive[:max_evidence]
        elif items:
            result.status = "negated"
            result.evidence = items[:1]
        else:
            implying = sorted({src for skill in req.skills for src in ontology.implying(skill) if src in present})
            if implying:
                result.status = "inferred"
                result.strength = INFERRED_STRENGTH
                result.inferred_from = implying
        if result.status != "supported":
            result.related_found = sorted({r for skill in req.skills for r in ontology.related(skill) if r in present} - set(req.skills))
        result.explanation = _explain(result)
        results.append(result)
    return results
