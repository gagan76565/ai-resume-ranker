"""Skill taxonomy loading and alias matching.

Matching rules (all tested in ``tests/test_scoring_skills.py``):

* Aliases match case-insensitively, except those listed under
  ``case_sensitive`` in ``skills.json`` (single letters such as ``C``/``R``
  and words like ``Go``/``Swift``/``Excel`` that are also ordinary English).
* Boundaries are explicit: a match may not touch letters, digits, ``_``,
  ``+``, ``#`` or ``&`` on either side, and may not be followed by ``.x``.
  That keeps ``java`` out of ``javascript``, ``c`` out of ``c++``/``c#`` and
  ``node`` out of ``node.js``, while still matching ``C++`` and ``Node.js``
  themselves and allowing a trailing full stop ("... and Node.js.").
* Single-letter aliases additionally may not touch ``-``, ``'``, ``.``, or
  ``&`` (so ``R&D``, ``C-level``, ``C.`` initials do not count).
* When aliases overlap ("SQL Server" vs "SQL"), the longest match wins and
  spans never overlap, so one phrase counts as one skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .text import normalize_text

DATA_PATH = Path(__file__).resolve().parent / "data" / "skills.json"

_WORD = r"A-Za-z0-9_+#&"
_LEFT = rf"(?<![{_WORD}])(?<![A-Za-z0-9]\.)"
_RIGHT = rf"(?![{_WORD}])(?!\.[A-Za-z0-9])"
_LEFT_STRICT = rf"(?<![{_WORD}\-'’.])"
_RIGHT_STRICT = rf"(?![{_WORD}\-'’.])"


@dataclass(frozen=True)
class Skill:
    """One taxonomy entry."""

    canonical: str
    category: str
    aliases: tuple[str, ...] = ()
    case_sensitive: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillMatch:
    """One occurrence of a skill in a text (character offsets into the normalised text)."""

    canonical: str
    start: int
    end: int
    surface: str


@dataclass
class Taxonomy:
    """A compiled skill taxonomy. Build it with :func:`load_taxonomy` or :func:`build_taxonomy`."""

    skills: dict[str, Skill]
    _ci_pattern: re.Pattern | None = field(default=None, repr=False)
    _cs_pattern: re.Pattern | None = field(default=None, repr=False)
    _ci_lookup: dict[str, str] = field(default_factory=dict, repr=False)
    _cs_lookup: dict[str, str] = field(default_factory=dict, repr=False)

    def canonical_for(self, name: str) -> str | None:
        """Return the canonical skill for ``name`` (canonical or alias), or ``None``."""
        key = normalize_text(name).lower()
        if key in self._ci_lookup:
            return self._ci_lookup[key]
        return self._cs_lookup.get(normalize_text(name))

    def find(self, text: str) -> list[SkillMatch]:
        """Return non-overlapping skill matches in ``text`` (which should be normalised)."""
        candidates: list[SkillMatch] = []
        for pattern, lookup, fold in (
            (self._ci_pattern, self._ci_lookup, True),
            (self._cs_pattern, self._cs_lookup, False),
        ):
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                surface = match.group(0)
                key = surface.lower() if fold else surface
                key = re.sub(r"\s+", " ", key)
                canonical = lookup.get(key)
                if canonical:
                    candidates.append(SkillMatch(canonical, match.start(), match.end(), surface))
        return _resolve_overlaps(candidates)

    def found_skills(self, text: str) -> dict[str, list[SkillMatch]]:
        """Group matches by canonical skill name (insertion order = first appearance)."""
        grouped: dict[str, list[SkillMatch]] = {}
        for match in self.find(text):
            grouped.setdefault(match.canonical, []).append(match)
        return grouped


def _resolve_overlaps(matches: list[SkillMatch]) -> list[SkillMatch]:
    """Greedy left-to-right selection preferring longer spans at the same start."""
    matches.sort(key=lambda m: (m.start, -(m.end - m.start), m.canonical))
    chosen: list[SkillMatch] = []
    last_end = -1
    for match in matches:
        if match.start >= last_end:
            chosen.append(match)
            last_end = match.end
    return chosen


def alias_regex(alias: str) -> str:
    """Return a boundary-safe regex fragment for one alias.

    Internal whitespace matches any run of spaces so "machine  learning" and
    "machine learning" are equivalent.
    """
    body = r"\s+".join(re.escape(part) for part in alias.split())
    if len(alias) == 1:
        return f"{_LEFT_STRICT}{body}{_RIGHT_STRICT}"
    return f"{_LEFT}{body}{_RIGHT}"


def _compile(aliases: list[str], flags: int) -> re.Pattern | None:
    if not aliases:
        return None
    # Longest first so the regex alternation prefers "sql server" over "sql".
    ordered = sorted(set(aliases), key=lambda a: (-len(a), a))
    return re.compile("|".join(alias_regex(alias) for alias in ordered), flags)


def build_taxonomy(entries: list[dict]) -> Taxonomy:
    """Compile a taxonomy from raw dict entries (the ``skills`` list of ``skills.json``)."""
    skills: dict[str, Skill] = {}
    ci_lookup: dict[str, str] = {}
    cs_lookup: dict[str, str] = {}
    for entry in entries:
        canonical = entry["canonical"].strip()
        if canonical in skills:
            raise ValueError(f"Duplicate canonical skill: {canonical}")
        case_sensitive = tuple(a.strip() for a in entry.get("case_sensitive", []))
        aliases = tuple(a.strip().lower() for a in entry.get("aliases", []))
        skill = Skill(canonical, entry.get("category", "other"), aliases, case_sensitive)
        skills[canonical] = skill
        ci_names = list(aliases)
        if canonical not in case_sensitive:
            ci_names.append(canonical.lower())
        for alias in ci_names:
            alias = re.sub(r"\s+", " ", alias)
            owner = ci_lookup.setdefault(alias, canonical)
            if owner != canonical:
                raise ValueError(f"Alias {alias!r} is used by both {owner} and {canonical}")
        for alias in case_sensitive:
            cs_lookup[alias] = canonical
    return Taxonomy(
        skills=skills,
        _ci_pattern=_compile(list(ci_lookup), re.IGNORECASE),
        _cs_pattern=_compile(list(cs_lookup), 0),
        _ci_lookup=ci_lookup,
        _cs_lookup=cs_lookup,
    )


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    """Load and compile ``skills.json`` (cached per path)."""
    with open(path or DATA_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    return build_taxonomy(data["skills"])


def custom_skill_pattern(name: str) -> re.Pattern:
    """Boundary-safe, case-insensitive pattern for a free-text must-have skill not in the taxonomy."""
    cleaned = normalize_text(name)
    return re.compile(alias_regex(cleaned.lower()), re.IGNORECASE)
