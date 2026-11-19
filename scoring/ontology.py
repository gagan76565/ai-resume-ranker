"""Skill ontology: top-level groups, related technologies and conservative implications.

Distinctions used throughout v3 (see docs/ai/SCORING_METHODOLOGY.md):

* **exact**     — the resume uses the canonical name ("PostgreSQL");
* **alias**     — the resume uses a recognised alias ("postgres", "k8s");
* **related**   — a *different* technology in the same family (Angular for a React requirement).
                  Reported for the reviewer, never counted as the required skill;
* **inferred**  — implied by another skill (PostgreSQL ⇒ SQL). Reported as "inferred (unverified)" with a
                  low strength, never as demonstrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .skills import Taxonomy, load_taxonomy

RELATIONS_PATH = Path(__file__).resolve().parent / "data" / "relations.json"
TOP_LEVEL_GROUPS = (
    "Programming languages", "Frontend technologies", "Backend technologies", "Databases", "Cloud and DevOps",
    "Machine learning", "Data science", "Testing", "Development tools", "Computer science fundamentals",
)


@dataclass(frozen=True)
class Ontology:
    taxonomy: Taxonomy
    group_of: dict[str, str]
    related_family: dict[str, str]  # skill -> family name
    family_members: dict[str, tuple[str, ...]]
    implied_by: dict[str, tuple[str, ...]]  # implied skill -> skills that imply it

    def group(self, skill: str) -> str:
        return self.group_of.get(skill, "Development tools")

    def related(self, skill: str) -> tuple[str, ...]:
        family = self.related_family.get(skill)
        return tuple(s for s in self.family_members.get(family, ()) if s != skill) if family else ()

    def family(self, skill: str) -> str | None:
        return self.related_family.get(skill)

    def implying(self, skill: str) -> tuple[str, ...]:
        return self.implied_by.get(skill, ())


@lru_cache(maxsize=2)
def load_ontology(path: str | None = None) -> Ontology:
    taxonomy = load_taxonomy()
    data = json.loads(Path(path or RELATIONS_PATH).read_text(encoding="utf-8"))
    known = set(taxonomy.skills)

    def check(name: str) -> str:
        if name not in known:
            raise ValueError(f"relations.json refers to unknown skill {name!r}")
        return name

    group_of = {name: data["groups"].get(skill.category, "Development tools") for name, skill in taxonomy.skills.items()}
    for group, names in data.get("group_overrides", {}).items():
        if group not in TOP_LEVEL_GROUPS:
            raise ValueError(f"Unknown top-level group {group!r}")
        for name in names:
            group_of[check(name)] = group
    for name, group in group_of.items():
        if group not in TOP_LEVEL_GROUPS:
            raise ValueError(f"{name} maps to unknown group {group!r}")

    related_family: dict[str, str] = {}
    family_members: dict[str, tuple[str, ...]] = {}
    for family, names in data.get("related", {}).items():
        family_members[family] = tuple(check(n) for n in names)
        for name in names:
            related_family.setdefault(name, family)  # first family wins if listed twice

    implied_by: dict[str, list[str]] = {}
    for source, targets in data.get("implies", {}).items():
        for target in targets:
            implied_by.setdefault(check(target), []).append(check(source))
    return Ontology(taxonomy, group_of, related_family, family_members, {k: tuple(v) for k, v in implied_by.items()})
