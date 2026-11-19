"""Generate the bundled synthetic sample resumes in ``samples/``.

Every person here is fictional. Each PDF is stamped "SYNTHETIC DEMO DATA".
Run from the project root:  python scripts/make_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talentlens.minipdf import build_pdf  # noqa: E402

OUT = ROOT / "samples"
STAMP = "SYNTHETIC DEMO DATA - fictional person created for the TalentLens demo. Not a real candidate."

JOB_TITLE = "Junior Machine Learning Engineer (sample)"
JOB_DESCRIPTION = """Junior Machine Learning Engineer - SAMPLE JOB DESCRIPTION (synthetic demo data)

We are a small product team building recommendation features for an online learning platform.

What you will do:
- Build and evaluate machine learning models in Python using scikit-learn and pandas.
- Write SQL against PostgreSQL to prepare training data and analyse experiments.
- Package models as REST APIs with Flask and ship them with Docker.
- Deploy and monitor services on AWS.
- Use Git and code review every day; write unit tests with pytest.

Nice to have: experience with NLP, Apache Airflow or Kubernetes.
"""
MUST_HAVE = ["Python", "SQL"]

RESUMES = [
    {
        "file": "sample-01-aarav-placeholder.pdf",
        "description": "Strong match: clear Skills, Experience and Projects sections.",
        "lines": [
            "# Aarav Placeholder", STAMP, "aarav.placeholder@example.invalid",
            "# Summary",
            "Junior ML engineer who likes turning messy data into small, reliable services.",
            "# Skills",
            "Python, scikit-learn, pandas, SQL, PostgreSQL, Flask, Docker, AWS, Git, pytest",
            "# Experience",
            "Machine Learning Intern, Example Learning Co. (fictional), 2025-2026",
            "Built a course recommendation model in Python with scikit-learn and pandas; improved offline recall.",
            "Wrote SQL queries in PostgreSQL to build training sets and analyse A/B tests.",
            "Served the model as a REST API with Flask, packaged with Docker and deployed on AWS.",
            "# Projects",
            "Quiz difficulty predictor: gradient boosting model with unit tests in pytest; regular code review on GitHub.",
            "# Education",
            "BSc Computer Science, Placeholder University (fictional)",
        ],
    },
    {
        "file": "sample-02-bea-example.pdf",
        "description": "Uses aliases (ml, sklearn, postgres, k8s) - shows synonym matching.",
        "lines": [
            "# Bea Example", STAMP, "bea.example@example.invalid",
            "# Technical Skills",
            "py, ml, sklearn, SQL (postgres), k8s, docker, fastapi, git, nlp",
            "Python 3 for data work; ML model training and evaluation.",
            "# Work Experience",
            "Data Analyst, Sample Retail Ltd (fictional), 2024-2026",
            "Trained ml models (sklearn) for demand forecasting; queried postgres daily.",
            "Containerised a FastAPI scoring service with docker and ran it on k8s.",
            "# Education",
            "MSc Data Science, Example Institute (fictional)",
        ],
    },
    {
        "file": "sample-03-chen-fictional.pdf",
        "description": "Front-end developer: weak match for an ML role.",
        "lines": [
            "# Chen Fictional", STAMP, "chen.fictional@example.invalid",
            "# Skills",
            "JavaScript, TypeScript, React, Next.js, CSS, Figma, Jest, Git",
            "# Experience",
            "Front-end Developer, Demo Studio (fictional), 2023-2026",
            "Built accessible React interfaces in TypeScript; wrote Jest tests; reviewed pull requests on GitHub.",
            "# Projects",
            "Design-system component library with Storybook-style docs.",
            "# Education",
            "BA Interaction Design, Sample College (fictional)",
        ],
    },
    {
        "file": "sample-04-dana-keywords.pdf",
        "description": "Keyword list with no section headings: section signal is 0.",
        "lines": [
            "# Dana Keywords", STAMP,
            "Python SQL PostgreSQL scikit-learn pandas Docker AWS Flask Git pytest machine learning REST APIs",
            "Looking for a junior role in machine learning.",
        ],
    },
    {
        "file": "sample-05-eli-analyst.pdf",
        "description": "Data analyst: partial match (SQL and Python, no deployment skills).",
        "lines": [
            "# Eli Analyst", STAMP, "eli.analyst@example.invalid",
            "# Profile",
            "Analyst who builds dashboards and explains numbers to non-technical teams.",
            "# Skills",
            "SQL, Python, pandas, Excel, Tableau, Power BI, statistics",
            "# Experience",
            "Business Analyst, Fictional Finance Group, 2022-2026",
            "Wrote SQL for weekly KPI reporting; automated Excel reports with Python and pandas.",
            "Ran A/B test analysis and presented results in Tableau dashboards.",
            "# Education",
            "BCom, Example University (fictional)",
        ],
    },
    {
        "file": "sample-06-scanned-simulation.pdf",
        "description": "Simulates a scanned (image-only) PDF: almost no extractable text, so a warning is shown.",
        "lines": ["Fen Scan", "SYNTHETIC DEMO DATA"],
    },
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "job_description.txt").write_text(JOB_DESCRIPTION, encoding="utf-8", newline="\n")
    for resume in RESUMES:
        (OUT / resume["file"]).write_bytes(build_pdf(resume["lines"], title=resume["file"]))
    manifest = {
        "label": "Synthetic demo data: fictional people and a sample job description generated for TalentLens.",
        "job_title": JOB_TITLE,
        "job_description_file": "job_description.txt",
        "must_have": MUST_HAVE,
        "resumes": [{"file": r["file"], "description": r["description"]} for r in RESUMES],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(RESUMES)} sample resumes to {OUT}")


if __name__ == "__main__":
    main()
