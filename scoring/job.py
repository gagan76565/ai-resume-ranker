"""Job-description analysis: requirements with a class and alternatives.

* **mandatory** — must-have skills added by the recruiter, and skills in a line or block introduced by
  "required", "must have", "must", "essential" or "minimum qualifications";
* **preferred** — skills in a line or block introduced by "nice to have", "preferred", "bonus", "a plus",
  "desirable", "good to have" or "optional";
* **general**   — every other recognised skill mentioned in the JD.

"X or Y" and "X/Y" between two recognised skills in the same clause become **one** requirement satisfied by
either. Negated JD mentions ("Java is not required") are not requirements.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .context import is_negated
from .skills import Taxonomy, custom_skill_pattern, load_taxonomy
from .text import normalize_text

_PREFERRED = re.compile(r"\b(?:nice[\s-]to[\s-]have|preferred|bonus|a\s+plus|is\s+a\s+plus|desirable|good[\s-]to[\s-]have|optional|ideally)\b", re.IGNORECASE)
_MANDATORY = re.compile(r"\b(?:required|requirements?|must[\s-]have|must|essential|minimum\s+qualifications?|mandatory)\b", re.IGNORECASE)
_NEUTRAL_HEADING = re.compile(
    r"^(?:what\s+you(?:'ll|\s+will)\s+do|responsibilities|about\s+(?:us|the\s+role|you)|the\s+role|benefits|perks|what\s+we\s+offer)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_GAP = re.compile(r"^\s*(?:,?\s*or\s+|/\s*)$", re.IGNORECASE)
_CHAIN_GAP = re.compile(r"^\s*(?:,\s*|,?\s*or\s+|/\s*)$", re.IGNORECASE)


@dataclass
class Requirement:
    id: str
    label: str  # e.g. "Flask or FastAPI"
    skills: list[str]  # canonical names; any one satisfies the requirement
    cls: str  # mandatory | preferred | general
    source: dict | None = None  # {"text", "start", "end"} of the JD line, None for recruiter must-haves
    origin: str = "job_description"  # job_description | recruiter_must_have
    custom: bool = False  # not in the taxonomy (literal must-have)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _block_class(line: str, current: str | None) -> tuple[str | None, str | None]:
    """Return (class for this line, class carried to following lines)."""
    stripped = line.strip()
    is_heading = (stripped.endswith(":") and len(stripped) <= 60) or (len(stripped) <= 40 and not stripped.endswith("."))
    pref, mand = _PREFERRED.search(stripped), _MANDATORY.search(stripped)
    line_cls = "preferred" if pref else "mandatory" if mand else None
    if is_heading and (pref or mand):
        return line_cls, line_cls
    if is_heading and _NEUTRAL_HEADING.search(stripped):
        return None, None
    return line_cls or current, current


def analyze_job(job_description: str, must_have=(), taxonomy: Taxonomy | None = None) -> list[Requirement]:
    taxonomy = taxonomy or load_taxonomy()
    text = normalize_text(job_description)
    requirements: list[Requirement] = []
    seen: dict[str, Requirement] = {}

    for raw in must_have:
        name = normalize_text(raw)[:60]
        if not name:
            continue
        canonical = taxonomy.canonical_for(name) or name
        if canonical.lower() in seen:
            continue
        req = Requirement(f"r{len(requirements) + 1}", canonical, [canonical], "mandatory", None, "recruiter_must_have",
                          custom=taxonomy.canonical_for(name) is None)
        requirements.append(req)
        seen[canonical.lower()] = req

    carried: str | None = None
    offset = 0
    for line in text.split("\n"):
        line_start = offset
        offset += len(line) + 1
        line_cls, carried = _block_class(line, carried)
        matches = [m for m in taxonomy.find(line) if not is_negated(line, m.start, m.end)]
        groups: list[list] = []
        i = 0
        while i < len(matches):
            # Build a chain of skills separated only by "," / "or" / "/". If any separator is "or" or "/",
            # the whole chain is a list of alternatives ("NLP, Airflow or Kubernetes"); otherwise each item
            # is its own requirement ("Python, SQL and Docker").
            chain = [matches[i]]
            gaps = []
            while i + 1 < len(matches) and _CHAIN_GAP.match(line[chain[-1].end:matches[i + 1].start]):
                gaps.append(line[chain[-1].end:matches[i + 1].start])
                chain.append(matches[i + 1])
                i += 1
            i += 1
            # "/" binds its two neighbours ("AWS/GCP"); a comma list is a set of alternatives only when its
            # last separator is "or" ("NLP, Airflow or Kubernetes"); otherwise items stay separate.
            units: list[list] = [[chain[0]]]
            unit_gaps: list[str] = []
            for gap, item in zip(gaps, chain[1:], strict=True):
                if gap.strip() == "/":
                    units[-1].append(item)
                else:
                    unit_gaps.append(gap)
                    units.append([item])
            if unit_gaps and re.search(r"\bor\b", unit_gaps[-1], re.IGNORECASE):
                groups.append([m for unit in units for m in unit])
            else:
                groups.extend(units)
        for group in groups:
            names = list(dict.fromkeys(m.canonical for m in group))
            cls = line_cls or "general"
            existing = [seen[n.lower()] for n in names if n.lower() in seen]
            if existing:
                # Upgrade the class if this mention is stronger (mandatory > general > preferred).
                rank = {"preferred": 0, "general": 1, "mandatory": 2}
                for req in existing:
                    if rank[cls] > rank[req.cls] and req.origin != "recruiter_must_have":
                        req.cls = cls
                continue
            label = " or ".join(names)
            source = {"text": line.strip(), "start": line_start, "end": line_start + len(line)}
            req = Requirement(f"r{len(requirements) + 1}", label, names, cls, source)
            requirements.append(req)
            for n in names:
                seen[n.lower()] = req
    return requirements


def custom_pattern(req: Requirement):
    return custom_skill_pattern(req.label) if req.custom else None
