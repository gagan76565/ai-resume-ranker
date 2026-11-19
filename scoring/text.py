"""Text normalisation and tokenisation helpers.

Everything here is pure and deterministic: the same input always gives the
same output, and nothing depends on global state or the order of other
documents.
"""

from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Characters that PDF extractors commonly emit for bullets and dashes.
_BULLETS = "•●▪■‣⁃◦∙·"
_DASHES = "‐‑‒–—―−"

_BULLET_RE = re.compile("[" + re.escape(_BULLETS) + "]")
_DASH_RE = re.compile("[" + re.escape(_DASHES) + "]")
_SPACES_RE = re.compile(r"[ \t\f\v ]+")
# Sentence boundary: terminal punctuation followed by whitespace. We do not
# split on a bare "." so that "Node.js" or "e.g." inside words survive.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\s*\|\s*")
# Tokens: start with a letter or digit, may contain + # (C++, C#) and inner
# dots/hyphens between word characters (node.js, front-end, scikit-learn).
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#]*(?:[.\-/][a-z0-9+#]+)*")

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def normalize_text(text: str | None) -> str:
    """Return ``text`` with Unicode compatibility forms, bullets and dashes normalised.

    Line breaks are preserved because section detection and segment-level
    n-grams rely on them. Runs of horizontal whitespace collapse to one space
    and trailing spaces are stripped from every line.
    """
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _BULLET_RE.sub("\n", value)
    value = _DASH_RE.sub("-", value)
    lines = (_SPACES_RE.sub(" ", line).strip() for line in value.split("\n"))
    return "\n".join(line for line in lines if line)


def segments(text: str) -> list[str]:
    """Split normalised text into short segments (lines, then sentences).

    Bigrams are only formed inside a segment, so moving whole lines or
    sections around cannot change a document's term vector.
    """
    out: list[str] = []
    for line in normalize_text(text).split("\n"):
        out.extend(part for part in _SENTENCE_SPLIT_RE.split(line) if part)
    return out


def tokenize(segment: str) -> list[str]:
    """Lower-case word tokens of one segment, without stop words, numbers or 1-char tokens."""
    tokens = []
    for token in _TOKEN_RE.findall(segment.lower()):
        token = token.rstrip(".-/")
        if len(token) < 2 or token in STOP_WORDS or token.replace(".", "").isdigit():
            continue
        tokens.append(token)
    return tokens


def analyzer(text: str) -> list[str]:
    """Unigrams plus within-segment bigrams, for ``TfidfVectorizer(analyzer=...)``."""
    features: list[str] = []
    for segment in segments(text):
        tokens = tokenize(segment)
        features.extend(tokens)
        features.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return features
