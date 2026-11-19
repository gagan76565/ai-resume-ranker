"""Context analysis for a skill mention: clause, negation and evidence strength.

Evidence strength (deterministic, documented in docs/ai/SCORING_METHODOLOGY.md §B2):

| type          | strength | rule |
|---------------|---------:|------|
| demonstrated  | 1.0      | the clause contains an action verb (built, developed, deployed …) and is not in a Skills list |
| listed        | 0.6      | the mention is in a Skills section, or in an Experience/Projects clause without an action verb |
| bare          | 0.3      | anywhere else (no recognised section, summary text without an action) |
| negated       | 0.0      | a negation cue governs the mention in the same clause ("no", "not", "without", "lack of" …) |
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sections import Section, section_at

ACTION_VERBS = re.compile(
    r"\b(?:built|builds|build|developed|develops|develop|implemented|implements|implement|designed|designs|design|"
    r"created|creates|deployed|deploys|deploy|engineered|wrote|writes|led|leads|architected|automated|automates|"
    r"migrated|optimi[sz]ed|integrated|integrates|trained|trains|shipped|launched|delivered|maintained|maintains|"
    r"served|packaged|containeri[sz]ed|analy[sz]ed|analy[sz]es|ran|runs|reduced|improved|increased|refactored|"
    r"tested|configured|administered|managed|modelled|modeled|queried|scaled|monitored|programmed|used|uses)\b",
    re.IGNORECASE,
)
# A negation cue must govern the skill closely: at most three words in between ("no professional
# experience in Java"). "without" only counts when followed by experience-type words ("without any
# Java"), so "migrated without downtime using Kubernetes" is NOT negated. "not just/only" is not negation.
_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:no|not|never|lack(?:ing)?\s+of|lacks?|unfamiliar\s+with)\b(?!\s+(?:only|just|limited)\b)|n't)"
    r"(?:\s+[\w'/-]+){0,3}\s*$"
    r"|\bwithout(?:\s+(?:any|prior|professional|commercial|hands-on|experience|knowledge|exposure|using|with|in|of))*\s*$",
    re.IGNORECASE,
)
_NEGATION_AFTER = re.compile(r"^[^.;\n]{0,25}\b(?:is|are)?\s*not\s+(?:required|necessary|needed|used)\b", re.IGNORECASE)
# Sentence punctuation only counts when followed by whitespace or the end ("Node.js" is not a break).
_CLAUSE_BREAK = re.compile(r"[.!?](?=\s|$)|[;\n]|\bbut\b|\bhowever\b", re.IGNORECASE)

STRENGTH = {"demonstrated": 1.0, "listed": 0.6, "bare": 0.3, "negated": 0.0}


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    clause_start: int
    clause_end: int
    section: str  # skills | experience | projects | <label> | none
    kind: str  # demonstrated | listed | bare | negated
    strength: float


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Smallest [s, e) around the span bounded by sentence punctuation, newlines or 'but/however'."""
    left = 0
    for m in _CLAUSE_BREAK.finditer(text, 0, start):
        left = m.end()
    right_match = _CLAUSE_BREAK.search(text, end)
    right = right_match.start() if right_match else len(text)
    # Keep the clause reasonably short for quoting.
    left = max(left, start - 240)
    right = min(right, end + 240)
    while left < start and text[left].isspace():
        left += 1
    return left, right


def is_negated(text: str, start: int, end: int) -> bool:
    c_start, c_end = clause_bounds(text, start, end)
    before = text[c_start:start]
    after = text[end:c_end]
    return bool(_NEGATION_BEFORE.search(before)) or bool(_NEGATION_AFTER.search(after))


def classify(text: str, start: int, end: int, sections: list[Section]) -> Mention:
    c_start, c_end = clause_bounds(text, start, end)
    section = section_at(start, sections)
    label = (section.label or section.kind) if section else "none"
    clause = text[c_start:c_end]
    if is_negated(text, start, end):
        kind = "negated"
    elif label != "skills" and ACTION_VERBS.search(clause):
        kind = "demonstrated"
    elif label in {"skills", "experience", "projects"}:
        kind = "listed"
    else:
        kind = "bare"
    return Mention(start, end, c_start, c_end, label, kind, STRENGTH[kind])
