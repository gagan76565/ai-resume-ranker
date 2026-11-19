import unittest

from scoring.context import classify, is_negated
from scoring.job import analyze_job
from scoring.ontology import TOP_LEVEL_GROUPS, load_ontology
from scoring.sections import detect_sections
from scoring.skills import load_taxonomy

T = load_taxonomy()


def mentions(text):
    sections = detect_sections(text)
    return {m.canonical: classify(text, m.start, m.end, sections) for m in T.find(text)}


class OntologyTests(unittest.TestCase):
    def test_every_skill_has_a_top_level_group(self):
        ontology = load_ontology()
        for name in T.skills:
            self.assertIn(ontology.group(name), TOP_LEVEL_GROUPS, name)
        self.assertEqual(ontology.group("pytest"), "Testing")
        self.assertEqual(ontology.group("PostgreSQL"), "Databases")

    def test_related_is_not_equivalence(self):
        ontology = load_ontology()
        self.assertIn("Angular", ontology.related("React"))
        self.assertNotIn("React", ontology.related("React"))
        self.assertIsNone(ontology.family("Python"))

    def test_implications_are_one_way(self):
        ontology = load_ontology()
        self.assertIn("PostgreSQL", ontology.implying("SQL"))
        self.assertEqual(ontology.implying("PostgreSQL"), ())

    def test_required_aliases_from_brief(self):
        for alias, canonical in {"JS": "JavaScript", "TS": "TypeScript", "React.js": "React", "Node.js": "Node.js",
                                 "Postgres": "PostgreSQL", "RESTful APIs": "REST APIs", "scikit learn": "scikit-learn"}.items():
            with self.subTest(alias=alias):
                self.assertEqual(T.canonical_for(alias), canonical)


class NegationTests(unittest.TestCase):
    def check(self, text, expected):
        found = {m.canonical: is_negated(text, m.start, m.end) for m in T.find(text)}
        self.assertEqual(found, expected, text)

    def test_negated(self):
        self.check("No experience with Java.", {"Java": True})
        self.check("I have no professional experience in Java", {"Java": True})
        self.check("Never used Docker", {"Docker": True})
        self.check("Deployed without any Docker", {"Docker": True})
        self.check("Java is not required", {"Java": True})

    def test_not_negated_false_positive_guards(self):
        self.check("Migrated services without downtime using Kubernetes", {"Kubernetes": False})
        self.check("Built zero-downtime deploys with Kubernetes", {"Kubernetes": False})
        self.check("Not just Python but also Rust", {"Python": False, "Rust": False})
        self.check("Built services in Node.js and Express", {"Node.js": False, "Express": False})

    def test_negation_is_clause_scoped(self):
        self.check("Strong Python, not familiar with Go yet", {"Python": False, "Go": True})
        self.check("No Java. Built APIs in Kotlin.", {"Java": True, "Kotlin": False})


class ContextStrengthTests(unittest.TestCase):
    def test_demonstrated_beats_listed(self):
        text = "Skills\nPython, Docker\nExperience\nUsed Python to develop a Flask REST API."
        m = {k: v for k, v in mentions(text).items()}
        self.assertEqual(m["Flask"].kind, "demonstrated")
        self.assertEqual(m["Docker"].kind, "listed")
        self.assertEqual(m["Docker"].strength, 0.6)
        self.assertEqual(m["Flask"].strength, 1.0)

    def test_bare_keyword_line_is_weak(self):
        m = mentions("Python SQL Docker AWS\nLooking for a junior role.")
        self.assertTrue(all(v.kind == "bare" for v in m.values()))

    def test_clause_is_quotable(self):
        text = "Experience\nDeveloped REST APIs using Python and Flask. Enjoys hiking."
        m = mentions(text)["Python"]
        self.assertEqual(text[m.clause_start:m.clause_end], "Developed REST APIs using Python and Flask")


class JobAnalysisTests(unittest.TestCase):
    def test_classes_and_alternatives(self):
        jd = ("Requirements:\n- Python and SQL\n- Experience with Flask or FastAPI\n"
              "Nice to have: Kubernetes, AWS/GCP\nJava is not required.")
        reqs = {r.label: r.cls for r in analyze_job(jd)}
        self.assertEqual(reqs, {"Python": "mandatory", "SQL": "mandatory", "Flask or FastAPI": "mandatory",
                                "Kubernetes": "preferred", "AWS or Google Cloud": "preferred"})

    def test_comma_or_list_is_one_alternative_group(self):
        reqs = analyze_job("Nice to have: experience with NLP, Apache Airflow or Kubernetes")
        self.assertEqual([r.label for r in reqs], ["NLP or Apache Airflow or Kubernetes"])

    def test_recruiter_must_haves_are_mandatory_and_first(self):
        reqs = analyze_job("We use Docker daily.", must_have=["k8s", "Fivetran"])
        self.assertEqual([(r.label, r.cls, r.origin) for r in reqs[:2]],
                         [("Kubernetes", "mandatory", "recruiter_must_have"), ("Fivetran", "mandatory", "recruiter_must_have")])
        self.assertTrue(reqs[1].custom)
        self.assertEqual(reqs[2].cls, "general")

    def test_requirement_source_is_a_jd_substring(self):
        jd = "About the role\nYou will build services in Go and PostgreSQL."
        for req in analyze_job(jd):
            self.assertEqual(jd[req.source["start"]:req.source["end"]].strip(), req.source["text"])


if __name__ == "__main__":
    unittest.main()
