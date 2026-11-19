"""Bundled synthetic sample resumes (fictional people, clearly labelled)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@lru_cache(maxsize=1)
def manifest() -> dict:
    with open(SAMPLES_DIR / "manifest.json", encoding="utf-8") as handle:
        data = json.load(handle)
    data["job_description"] = (SAMPLES_DIR / data["job_description_file"]).read_text(encoding="utf-8").strip()
    return data


def public_manifest() -> dict:
    data = manifest()
    return {
        "label": data["label"],
        "job_title": data["job_title"],
        "job_description": data["job_description"],
        "must_have": data["must_have"],
        "resumes": [{"filename": r["file"], "description": r["description"]} for r in data["resumes"]],
    }


def sample_files() -> list[tuple[str, bytes]]:
    return [(r["file"], (SAMPLES_DIR / r["file"]).read_bytes()) for r in manifest()["resumes"]]
