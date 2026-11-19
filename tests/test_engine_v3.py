import hashlib
import json
import random
import unittest

import numpy as np

from scoring import semantic
from scoring.engine_v3 import DEFAULT_WEIGHTS_V3, ConfigV3, ResumeInput, rank_candidates_v3
from scoring.redact import redact_for_scoring
from talentlens.documents import parse_document
from talentlens.minipdf import build_pdf
from talentlens.uploads import UploadLimits

JD = """Backend Engineer
Requirements:
- Python and SQL
- Experience with Flask or FastAPI
- Build REST APIs and deploy services with Docker
Nice to have: Kubernetes, AWS
"""
STRONG = """Jordan Sample
jordan@example.invalid
Skills
Python, SQL, Docker
Experience
Backend Developer, Example Co, 2022 - Present
Developed REST APIs using Python and Flask, deployed with Docker on AWS.
Wrote SQL queries against PostgreSQL for reporting.
Projects
Queue service: Built a job queue in FastAPI with Docker.
"""
LISTED = """Sam Lister
Skills
Python, SQL, Flask, Docker, REST APIs
Experience
Office administrator, Example Ltd, 2020 - 2024
Answered phones and organised meetings.
"""
NEGATED = """Experience
Support engineer, 2021 - 2024
No experience with Python. Used Excel daily.
"""


def run(resumes, must_have=(), config=None, encoder=None):
    inputs = [ResumeInput(name, text) for name, text in resumes]
    return rank_candidates_v3(JD, inputs, must_have, config or ConfigV3.from_env({}), encoder=encoder)


class FakeEncoder:
    """Deterministic bag-of-words embedding: lets us test the semantic path without the model."""

    name = "fake"

    def encode(self, texts):
        vectors = []
        for text in texts:
            vec = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                vec[int(hashlib.md5(word.strip(".,").encode()).hexdigest(), 16) % 64] += 1  # noqa: S324
            vectors.append(vec / (np.linalg.norm(vec) or 1))
        return np.stack(vectors)


class ConfigTests(unittest.TestCase):
    def test_defaults_sum_to_one(self):
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS_V3.values()), 1.0)

    def test_env_and_override_weights(self):
        config = ConfigV3.from_env({"V3_WEIGHT_LEXICAL": "0.5", "V3_WEIGHT_SEMANTIC": "abc"}, overrides={"required": 0.6})
        self.assertEqual(config.weights["lexical"], 0.5)
        self.assertEqual(config.weights["required"], 0.6)
        self.assertEqual(config.weights["semantic"], 0.25)
        self.assertTrue(config.problems)

    def test_all_zero_weights_fall_back(self):
        config = ConfigV3.from_env({f"V3_WEIGHT_{k.upper()}": "0" for k in DEFAULT_WEIGHTS_V3})
        self.assertEqual(dict(config.weights), DEFAULT_WEIGHTS_V3)


class ScoringTests(unittest.TestCase):
    def test_component_breakdown_reconstructs_fit(self):
        ranked, _ = run([("strong", STRONG)])
        r = ranked[0]
        self.assertAlmostEqual(sum(c["contribution"] for c in r["components"].values()), r["fit_score"], delta=0.3)
        self.assertAlmostEqual(sum(r["weights"].values()), 1.0, places=3)
        for key, comp in r["components"].items():
            if not comp["available"]:
                self.assertEqual(r["weights"][key], 0.0)

    def test_semantic_unavailable_is_redistributed_and_reported(self):
        r = run([("strong", STRONG)])[0][0]
        self.assertFalse(r["components"]["semantic"]["available"])
        self.assertIn("unavailable", r["components"]["semantic"]["note"])
        self.assertTrue(any("Semantic" in reason for reason in r["confidence"]["reasons"]))

    def test_demonstrated_beats_listed_keywords(self):
        ranked, _ = run([("listed", LISTED), ("strong", STRONG)])
        self.assertEqual([r["name"] for r in ranked], ["strong", "listed"])
        listed = ranked[1]
        statuses = {q["label"]: q["status"] for q in listed["requirements"]}
        self.assertEqual(statuses["Python"], "unclear")

    def test_missing_information_handled_explicitly(self):
        r = run([("listed", LISTED)])[0][0]
        self.assertFalse(r["components"]["projects"]["available"])  # Experience present, no Projects
        self.assertIn("redistributed", r["components"]["projects"]["note"])
        bare = run([("bare", "Python SQL Flask Docker developer looking for work")])[0][0]
        self.assertEqual(bare["components"]["experience"]["value"], 0.0)
        self.assertEqual(bare["components"]["projects"]["value"], 0.0)

    def test_mandatory_flag_policy_does_not_hide_score(self):
        r = run([("strong", STRONG)], must_have=["Kubernetes"])[0][0]
        self.assertFalse(r["mandatory"]["met"])
        self.assertEqual(r["mandatory"]["missing"], ["Kubernetes"])
        self.assertFalse(r["capped"])
        self.assertGreater(r["fit_score"], 40)
        self.assertTrue(any("not demonstrated" in w for w in r["warnings"]))

    def test_mandatory_cap_policy_is_opt_in(self):
        config = ConfigV3.from_env({"MANDATORY_POLICY": "cap", "MUST_HAVE_CAP": "40"})
        r = run([("strong", STRONG)], must_have=["Kubernetes"], config=config)[0][0]
        self.assertTrue(r["capped"])
        self.assertEqual(r["fit_score"], 40.0)
        self.assertGreater(r["evidence_fit"], 40.0)  # the evidence-based score is still reported

    def test_candidates_meeting_mandatory_rank_first(self):
        ranked, _ = run([("strong", STRONG), ("negated", NEGATED)], must_have=["Python"])
        self.assertEqual(ranked[0]["name"], "strong")
        self.assertFalse(ranked[1]["mandatory"]["met"])

    def test_reproducible_and_order_invariant(self):
        resumes = [("a", STRONG), ("b", LISTED), ("c", NEGATED)]
        first = json.dumps(run(resumes)[0], sort_keys=True)
        shuffled = resumes[:]
        random.Random(3).shuffle(shuffled)
        self.assertEqual(first, json.dumps(run(shuffled)[0], sort_keys=True))
        self.assertEqual(first, json.dumps(run(resumes)[0], sort_keys=True))

    def test_irrelevant_resume_scores_low(self):
        r = run([("chef", "Experience\nHead chef, 2015 - 2024\nDesigned seasonal menus and trained kitchen staff.")])[0][0]
        self.assertLess(r["fit_score"], 15)
        self.assertEqual(r["mandatory"]["coverage"], 0.0)


class EvidenceTests(unittest.TestCase):
    def test_quotes_are_exact_substrings_with_correct_section(self):
        r = run([("strong", STRONG)])[0][0]
        for req in r["requirements"]:
            for item in req["evidence"]:
                self.assertEqual(STRONG[item["start"]:item["end"]], item["quote"])
                self.assertEqual(STRONG[item["match_start"]:item["match_end"]], item["surface"])
        flask = next(q for q in r["requirements"] if q["label"] == "Flask or FastAPI")
        self.assertEqual(flask["status"], "supported")
        self.assertEqual(flask["evidence"][0]["section"], "experience")
        self.assertEqual(flask["evidence"][0]["quote"], "Developed REST APIs using Python and Flask, deployed with Docker on AWS")

    def test_missing_ambiguous_negated_inferred_related(self):
        r = run([("negated", NEGATED + "Skills\nPostgreSQL, Angular\n")], must_have=["React"])[0][0]
        by = {q["label"]: q for q in r["requirements"]}
        self.assertEqual(by["Python"]["status"], "negated")
        self.assertEqual(by["SQL"]["status"], "inferred")
        self.assertEqual(by["SQL"]["inferred_from"], ["PostgreSQL"])
        self.assertEqual(by["React"]["status"], "not_found")
        self.assertIn("Angular", by["React"]["related_found"])
        self.assertIn("not demonstrated", by["React"]["explanation"])
        self.assertNotIn("does not have", by["React"]["explanation"])

    def test_page_numbers_for_multi_page_pdf(self):
        lines = ["# Experience"] + [f"Maintained internal tool number {i} for the support team." for i in range(110)]
        lines += ["Developed REST APIs using Python and Flask with Docker."]
        doc = parse_document("long.pdf", build_pdf(lines), UploadLimits())
        inputs = [ResumeInput("long", doc.text, doc.page_at, doc.quality.score, doc.quality.to_dict())]
        r = rank_candidates_v3(JD, inputs, (), ConfigV3.from_env({}), encoder=None)[0][0]
        flask = next(q for q in r["requirements"] if q["label"] == "Flask or FastAPI")
        self.assertGreater(len(doc.pages), 1)
        self.assertEqual(flask["evidence"][0]["page"], doc.pages[-1].number)


class SemanticTests(unittest.TestCase):
    def test_semantic_component_with_encoder(self):
        r = run([("strong", STRONG)], encoder=FakeEncoder())[0][0]
        self.assertTrue(r["components"]["semantic"]["available"])
        self.assertTrue(r["semantic"]["used"])
        best = r["semantic"]["matches"][0]["best"]
        self.assertEqual(STRONG[best["start"]:best["end"]], best["text"])

    def test_budget_skips_semantic_for_whole_batch(self):
        config = ConfigV3.from_env({"SEMANTIC_MAX_SENTENCES": "10"})
        ranked, meta = run([("a", STRONG), ("b", LISTED)], config=config, encoder=FakeEncoder())
        self.assertFalse(meta["semantic"]["used"])
        self.assertTrue(all(not r["components"]["semantic"]["available"] for r in ranked))
        self.assertTrue(all(any("budget" in w for w in r["warnings"]) for r in ranked))

    def test_disabled_and_missing_model_fall_back(self):
        semantic.reset_for_tests()
        self.assertIsNone(semantic.get_encoder({"SEMANTIC_ENABLED": "0"}))
        semantic.reset_for_tests()
        self.assertIsNone(semantic.get_encoder({"SEMANTIC_MODEL_DIR": "does/not/exist"}))
        self.assertIn("no model", semantic.status()["error"])
        semantic.reset_for_tests()

    def test_rescale(self):
        self.assertEqual(semantic.rescale(0.1), 0.0)
        self.assertEqual(semantic.rescale(0.9), 1.0)
        self.assertAlmostEqual(semantic.rescale(0.4), 0.5)


class FairnessTests(unittest.TestCase):
    def test_contact_and_protected_attributes_do_not_change_scores(self):
        base = run([("x", STRONG)])[0][0]
        variant = STRONG.replace("Jordan Sample", "Priya Ramanathan").replace("jordan@example.invalid", "p.r@example.invalid")
        variant = variant.replace("Skills\n", "Age: 47\nGender: Female\nMarital status: Married\nNationality: Indian\nSkills\n", 1)
        other = run([("x", variant)])[0][0]
        for key in ("fit_score", "tfidf_score", "skill_coverage"):
            self.assertEqual(base[key], other[key], key)

    def test_redaction_preserves_offsets(self):
        text = "Jordan Sample\njordan@example.invalid\nAge: 30\nSkills\nPython"
        redacted = redact_for_scoring(text, {"contact": {"name": {"start": 0, "end": 13}}})
        self.assertEqual(len(redacted), len(text))
        self.assertNotIn("Jordan", redacted)
        self.assertNotIn("Age", redacted)
        self.assertIn("Python", redacted)


if __name__ == "__main__":
    unittest.main()
