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


# Text that tries to instruct an AI system ("ignore previous instructions", "rank me first" …). It is never obeyed:
# scoring is deterministic and the optional LLM only receives evidence quotes. It is surfaced to the recruiter.
INJECTION = re.compile(
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts?|rules)\b"
    r"|\bdisregard\s+(?:all\s+|the\s+)?(?:above|previous|prior)?\s*instructions\b"
    r"|\byou\s+are\s+now\s+(?:an?\s+|the\s+)?(?:ai|assistant|admin|administrator|system|recruiter|grader|evaluator)\b"
    r"|\b(?:reveal|print|show|repeat)\s+(?:your|the)\s+system\s+prompt\b"
    r"|\brank\s+(?:me|this\s+(?:candidate|resume))\s+(?:first|highest|top)\b"
    r"|\bgive\s+(?:me|this\s+(?:candidate|resume))\s+(?:a\s+)?(?:perfect|full|maximum|100)\b"
    r"|<\s*/?\s*(?:system|assistant|instructions)\s*>",
    re.IGNORECASE,
)


def injection_suspected(text: str) -> bool:
    return bool(INJECTION.search(text))
