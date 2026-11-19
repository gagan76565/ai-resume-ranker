"""Remove non-job-relevant personal data from the text used for *scoring*.

Contact details (name line, email, phone, labelled location) and lines that state protected or
irrelevant characteristics (age, date of birth, gender, marital status, nationality, religion,
ethnicity, disability, photo) are replaced by spaces of the same length, so character offsets — and
therefore every evidence quote taken from the original text — stay valid. Redacted text is only used
for lexical and semantic similarity; skills are never found in these lines in practice.
"""

from __future__ import annotations

import re

_PROTECTED_LINE = re.compile(
    r"^\s*(?:age|d\.?o\.?b\.?|date\s+of\s+birth|born|gender|sex|marital\s+status|married|single|nationality|citizenship|"
    r"religion|ethnicity|race|caste|disability|disabilities|photo(?:graph)?|pronouns|father'?s\s+name|mother'?s\s+name)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://\S+|\bwww\.\S+|\blinkedin\.com/\S+|\bgithub\.com/\S+", re.IGNORECASE)


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def redact_for_scoring(text: str, profile: dict | None = None) -> str:
    spans: list[tuple[int, int]] = [(m.start(), m.end()) for m in _PROTECTED_LINE.finditer(text)]
    spans += [(m.start(), m.end()) for m in _EMAIL.finditer(text)]
    spans += [(m.start(), m.end()) for m in _URL.finditer(text)]
    contact = (profile or {}).get("contact") or {}
    for key in ("name", "phone", "location"):
        value = contact.get(key)
        if value:
            spans.append((value["start"], value["end"]))
    return _blank(text, spans)
