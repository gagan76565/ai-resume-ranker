"""Synthetic benchmark dataset: 6 job descriptions × 12 resume archetypes = 72 annotated pairs.

All people, companies and jobs are fictional. Ground truth is assigned *by construction* by the author
of this file (who also wrote the taxonomy) — it is not an independent human annotation, so absolute
numbers are optimistic; the benchmark is most useful for *relative* comparison of engines and for
catching regressions.

Annotations per resume:
* ``relevance`` 0–3 (3 strong, 2 good/partial, 1 weak, 0 irrelevant);
* ``gold_skills``  — canonical skills the resume positively mentions (negated mentions excluded);
* ``gold_sections`` — skill → section where its strongest (demonstrated) evidence was written;
* ``gold_mandatory_met`` — whether every mandatory requirement is evidenced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ROLES = {
    "backend": {
        "title": "Backend Python Engineer",
        "jd": ("Backend Python Engineer\nWe build APIs for an online grocery platform.\nRequirements:\n- Python and SQL\n"
               "- Flask or FastAPI\n- REST APIs and Docker\nResponsibilities:\n- Design and build web services that power checkout and orders\n"
               "- Write efficient queries against relational databases\n- Ship services in containers and monitor them in production\n"
               "Nice to have: Kubernetes, AWS, Redis"),
        "must_have": ["Python"],
        "core": ["Python", "SQL", "Flask", "REST APIs", "Docker"],
        "preferred": ["Kubernetes", "AWS", "Redis"],
        "aliases": {"Python": "python3", "SQL": "structured query language", "REST APIs": "RESTful API", "Docker": "docker compose",
                    "Kubernetes": "k8s", "AWS": "Amazon Web Services"},
        "paraphrase": ["Designed and built web services that power checkout and order flows in Python.",
                       "Wrote efficient queries against relational databases for the finance team.",
                       "Shipped services as container images and monitored them in production."],
        "related": ["Django", "MySQL", "Java"],
        "things": ["the checkout service", "an order tracking endpoint", "the inventory sync job", "a pricing service"],
    },
    "frontend": {
        "title": "Frontend Engineer (React)",
        "jd": ("Frontend Engineer (React)\nRequirements:\n- React and TypeScript\n- HTML and CSS\n- Unit testing with Jest\n"
               "Responsibilities:\n- Build reusable interface components for a single-page web app\n- Write tests for user interface behaviour\n"
               "- Craft responsive layouts that work on phones\nNice to have: Next.js, GraphQL, Figma"),
        "must_have": ["React"],
        "core": ["React", "TypeScript", "HTML", "CSS", "Jest"],
        "preferred": ["Next.js", "GraphQL", "Figma"],
        "aliases": {"React": "React.js", "TypeScript": "TS", "HTML": "HTML5", "CSS": "CSS3", "Next.js": "nextjs"},
        "paraphrase": ["Built reusable interface components for a single-page web app used by clinics.",
                       "Wrote automated checks for user interface behaviour before each release.",
                       "Crafted responsive layouts that work well on phones and tablets."],
        "related": ["Angular", "Vue.js", "JavaScript"],
        "things": ["the booking widget", "a design-system button set", "the account settings page", "a search results view"],
    },
    "analyst": {
        "title": "Data Analyst",
        "jd": ("Data Analyst\nRequirements:\n- SQL and Python\n- Tableau or Power BI\n- Statistics\nResponsibilities:\n"
               "- Turn raw sales numbers into weekly insight reports for managers\n- Explain trends and experiment results to non-technical teams\n"
               "Nice to have: A/B testing, Looker, dbt"),
        "must_have": ["SQL"],
        "core": ["SQL", "Python", "Tableau", "Statistics"],
        "preferred": ["A/B testing", "Looker", "dbt"],
        "aliases": {"Python": "python3", "Statistics": "statistical analysis", "A/B testing": "split testing"},
        "paraphrase": ["Turned raw sales numbers into weekly insight reports for regional managers.",
                       "Explained trends and experiment outcomes to non-technical teams using SQL extracts.",
                       "Cleaned and reconciled monthly revenue spreadsheets for the board pack."],
        "related": ["Looker", "Microsoft Excel"],
        "things": ["the weekly sales report", "a churn analysis", "the pricing experiment review", "a regional revenue model"],
    },
    "ml": {
        "title": "Machine Learning Engineer",
        "jd": ("Machine Learning Engineer\nRequirements:\n- Python, scikit-learn and pandas\n- Machine learning model development\n- SQL\n"
               "Responsibilities:\n- Train and evaluate predictive models on customer data\n- Prepare training datasets and features\n"
               "Nice to have: PyTorch, MLflow, Docker"),
        "must_have": ["Python"],
        "core": ["Python", "scikit-learn", "pandas", "Machine learning", "SQL"],
        "preferred": ["PyTorch", "MLflow", "Docker"],
        "aliases": {"Python": "python3", "scikit-learn": "sklearn", "Machine learning": "ML", "PyTorch": "torch"},
        "paraphrase": ["Trained and evaluated predictive models on customer data in Python.",
                       "Prepared training datasets and engineered input signals for churn prediction.",
                       "Compared candidate models with cross-validation and reported their accuracy."],
        "related": ["TensorFlow", "Keras", "XGBoost"],
        "things": ["a churn prediction model", "the demand forecast", "a fraud classifier", "the recommendation ranker"],
    },
    "devops": {
        "title": "DevOps Engineer",
        "jd": ("DevOps Engineer\nRequirements:\n- Kubernetes and Docker\n- Terraform\n- CI/CD pipelines\n- Linux\n"
               "Responsibilities:\n- Automate infrastructure provisioning and releases\n- Keep production clusters healthy and observable\n"
               "Nice to have: AWS, Prometheus, Ansible"),
        "must_have": ["Kubernetes"],
        "core": ["Kubernetes", "Docker", "Terraform", "CI/CD", "Linux"],
        "preferred": ["AWS", "Prometheus", "Ansible"],
        "aliases": {"Kubernetes": "k8s", "CI/CD": "continuous integration", "Linux": "Ubuntu", "AWS": "Amazon Web Services"},
        "paraphrase": ["Automated infrastructure provisioning and releases with declarative templates.",
                       "Kept production clusters healthy and observable with alerts and runbooks.",
                       "Ran container orchestration for twenty internal services."],
        "related": ["CloudFormation", "Microsoft Azure", "Jenkins"],
        "things": ["the deployment pipeline", "a staging cluster", "the backup automation", "our logging stack"],
    },
    "dataeng": {
        "title": "Data Engineer",
        "jd": ("Data Engineer\nRequirements:\n- SQL and Python\n- Apache Spark\n- Apache Airflow\n- ETL pipelines\n"
               "Responsibilities:\n- Move data from many sources into the warehouse on a schedule\n- Make datasets reliable for analysts\n"
               "Nice to have: Apache Kafka, dbt, Snowflake"),
        "must_have": ["SQL"],
        "core": ["SQL", "Python", "Apache Spark", "Apache Airflow", "ETL"],
        "preferred": ["Apache Kafka", "dbt", "Snowflake"],
        "aliases": {"Python": "python3", "Apache Spark": "PySpark", "Apache Airflow": "Airflow", "Apache Kafka": "Kafka", "ETL": "ELT"},
        "paraphrase": ["Moved data from many sources into the warehouse on a schedule using Python.",
                       "Made datasets reliable for analysts with validation checks and alerts.",
                       "Rebuilt nightly batch jobs so they finish before the morning reports."],
        "related": ["Apache Flink", "BigQuery", "Hadoop"],
        "things": ["the nightly batch", "a customer events table", "the finance data mart", "a clickstream feed"],
    },
}

VERBS = ["Developed", "Built", "Implemented", "Designed", "Deployed", "Maintained"]
FILLER = [
    "Coordinated weekly planning meetings with the product team.",
    "Mentored two new colleagues during their first months.",
    "Presented quarterly updates to stakeholders.",
    "Wrote onboarding notes for new starters.",
    "Took part in hiring interviews for the team.",
]
TRAPS = ["Worked at Express Logistics Ltd on go-to-market planning.", "Supported R&D reporting for C-level stakeholders.",
         "Helped the rest of the team with rota planning."]
ARCHETYPES = [
    ("strong", 3), ("alias", 3), ("multipage", 3), ("paraphrase", 2), ("partial", 2), ("poor_format", 2),
    ("related", 1), ("ambiguous", 1), ("keyword", 1), ("missing_mandatory", 1), ("negated", 0), ("irrelevant", 0),
]


@dataclass
class Resume:
    id: str
    role: str
    archetype: str
    relevance: int
    lines: list[str]  # "# Heading" lines are headings
    gold_skills: set[str] = field(default_factory=set)
    gold_sections: dict[str, str] = field(default_factory=dict)
    gold_mandatory_met: bool = True
    negated_skills: set[str] = field(default_factory=set)

    @property
    def text(self) -> str:
        return "\n".join(line[2:] if line.startswith("# ") else line for line in self.lines)


NAMES = ["Robin Example", "Alex Placeholder", "Sam Fictional", "Jordan Sample", "Casey Demo", "Taylor Mock",
         "Morgan Synthetic", "Riley Example", "Jamie Placeholder", "Avery Fictional", "Quinn Sample", "Drew Demo"]


def _name(index: int) -> str:
    return NAMES[index % len(NAMES)]


def _demonstrated(spec, skills, form, rng_index):
    out = []
    for i in range(0, len(skills), 2):
        pair = skills[i:i + 2]
        thing = spec["things"][(i // 2 + rng_index) % len(spec["things"])]
        verb = VERBS[(i // 2 + rng_index) % len(VERBS)]
        out.append(f"{verb} {thing} using {' and '.join(form(s) for s in pair)}.")
    return out


def build_resume(role: str, archetype: str, relevance: int, index: int) -> Resume:
    spec = ROLES[role]
    r = Resume(f"{role}-{index:02d}-{archetype}", role, archetype, relevance, [])
    core, pref, must = spec["core"], spec["preferred"], spec["must_have"]

    def canon(s):
        return s

    def alias(s):
        return spec["aliases"].get(s, s)

    form = alias if archetype == "alias" else canon

    def add_demo(skills, section="experience"):
        for s in skills:
            r.gold_skills.add(s)
            r.gold_sections.setdefault(s, section)

    lines = [f"# {_name(index)}", "candidate@example.invalid", "Synthetic demo data - fictional person."]
    if archetype in {"strong", "alias", "multipage", "poor_format"}:
        exp = _demonstrated(spec, core, form, index)
        proj = _demonstrated(spec, pref[:2], form, index + 1)
        lines += ["# Summary", f"{spec['title']} with four years of delivery experience.", "# Skills", ", ".join(form(s) for s in core + pref[:2]),
                  "# Experience", f"{spec['title']}, Example Systems (fictional), 2021 - Present", *exp, FILLER[index % len(FILLER)],
                  "# Projects", *[f"Side project {i + 1}: {p}" for i, p in enumerate(proj)], "# Education", "BSc Computer Science, Example University, 2020"]
        add_demo(core)
        add_demo(pref[:2], "projects")
        if archetype == "multipage":
            lines += ["# Additional experience"] + [f"{FILLER[i % len(FILLER)][:-1]} (period {i + 1})." for i in range(110)]
        if archetype == "poor_format":
            body = " ".join(line[2:] if line.startswith("# ") else line for line in lines[3:])
            lines = lines[:3] + [body]
            r.gold_sections = {s: "none" for s in r.gold_sections}
    elif archetype == "paraphrase":
        lines += ["# Summary", f"Engineer who enjoys work like this: {spec['paraphrase'][0]}", "# Experience",
                  "Engineer, Example Works (fictional), 2020 - 2025", *spec["paraphrase"]]
        found = {s for s in core + pref if re.search(rf"\b{re.escape(s)}\b", " ".join(spec["paraphrase"]), re.IGNORECASE)}
        add_demo(sorted(found))
    elif archetype == "partial":
        half = core[: (len(core) + 1) // 2]
        lines += ["# Skills", ", ".join(half), "# Experience", "Junior Engineer, Example Labs (fictional), 2023 - Present",
                  *_demonstrated(spec, half, canon, index), FILLER[(index + 1) % len(FILLER)]]
        add_demo(half)
    elif archetype == "related":
        lines += ["# Skills", ", ".join(spec["related"]), "# Experience", "Engineer, Example Group (fictional), 2019 - 2024",
                  *_demonstrated(spec, spec["related"], canon, index)]
        add_demo(spec["related"])
    elif archetype == "ambiguous":
        lines += ["# Skills", ", ".join(core + pref), "# Experience", "Office coordinator, Example Ltd (fictional), 2019 - 2024",
                  "Answered customer emails and organised the shared calendar.", FILLER[index % len(FILLER)]]
        for s in core + pref:
            r.gold_skills.add(s)
            r.gold_sections[s] = "skills"
    elif archetype == "keyword":
        lines += [" ".join(core + pref + core), "Motivated professional looking for an exciting opportunity."]
        for s in core + pref:
            r.gold_skills.add(s)
            r.gold_sections[s] = "none"
    elif archetype == "missing_mandatory":
        rest = [s for s in core if s not in must]
        lines += ["# Skills", ", ".join(rest + pref[:1]), "# Experience", "Software Engineer, Example Cloud (fictional), 2020 - Present",
                  *_demonstrated(spec, rest + pref[:1], canon, index)]
        add_demo(rest + pref[:1])
        r.gold_mandatory_met = False
    elif archetype == "negated":
        neg = core[:3]
        lines += ["# Experience", "Customer support agent, Example Retail (fictional), 2018 - 2024",
                  f"No experience with {neg[0]}.", f"Never used {neg[1]}.", f"I have no professional experience in {neg[2]}.",
                  "Resolved customer tickets and updated help articles."]
        r.negated_skills = set(neg)
        r.gold_mandatory_met = False
    elif archetype == "irrelevant":
        lines += ["# Experience", "Head chef, Example Bistro (fictional), 2015 - 2024", "Designed seasonal menus and trained kitchen staff.",
                  TRAPS[list(ROLES).index(role) % len(TRAPS)]]
        r.gold_mandatory_met = False
    if archetype not in {"missing_mandatory", "negated", "irrelevant"}:
        r.gold_mandatory_met = all(m in r.gold_skills for m in must)
    r.lines = lines
    return r


def build_dataset() -> list[tuple[str, dict, list[Resume]]]:
    out = []
    for role, spec in ROLES.items():
        resumes = [build_resume(role, archetype, relevance, i) for i, (archetype, relevance) in enumerate(ARCHETYPES)]
        out.append((role, spec, resumes))
    return out


def self_check(dataset, taxonomy) -> list[str]:
    """Report unintended taxonomy matches in each resume (outside gold skills, negated skills and known traps)."""
    problems = []
    trap_text = " ".join(TRAPS)
    for _role, _spec, resumes in dataset:
        for r in resumes:
            text = r.text
            found = {m.canonical for m in taxonomy.find(text)}
            unexpected = found - r.gold_skills - r.negated_skills
            unexpected = {s for s in unexpected if not any(m.canonical == s for m in taxonomy.find(trap_text))}
            if unexpected:
                problems.append(f"{r.id}: unintended skills {sorted(unexpected)}")
    return problems
