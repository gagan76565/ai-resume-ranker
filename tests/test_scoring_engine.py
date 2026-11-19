import random
import unittest

from scoring import ScoringConfig, explanation_text, rank_candidates, required_skills, score_resume
from scoring.sections import detect_sections
from scoring.text import analyzer, normalize_text

JD = (
    "Junior data engineer. Required: Python, SQL, Docker, AWS and machine learning basics. "
    "You will build REST APIs and data pipelines with PostgreSQL."
)

STRONG = """Priya Example
Summary
Data engineer who enjoys clean pipelines.
Skills
Python, SQL, PostgreSQL, Docker, AWS, machine learning
Experience
Built data pipelines and REST APIs in Python on AWS, deployed with Docker.
Projects
A small machine learning model served over a REST API backed by PostgreSQL.
Education
BSc Computer Science
"""

WEAK = """Sam Example
Experience
Retail store supervisor. Managed rotas and stock levels.
Education
Diploma in business
"""


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = ScoringConfig.from_env({})
        self.assertEqual(config.normalised_weights(), {"tfidf": 0.40, "skills": 0.45, "sections": 0.15})
        self.assertEqual(config.must_have_cap, 40.0)
        self.assertEqual(config.warnings, ())

    def test_custom_weights_are_normalised(self):
        config = ScoringConfig.from_env({"SCORE_WEIGHT_TFIDF": "1", "SCORE_WEIGHT_SKILLS": "1", "SCORE_WEIGHT_SECTIONS": "2"})
        self.assertEqual(config.normalised_weights(), {"tfidf": 0.25, "skills": 0.25, "sections": 0.5})

    def test_invalid_weights_fall_back_with_warning(self):
        config = ScoringConfig.from_env({"SCORE_WEIGHT_TFIDF": "abc", "SCORE_WEIGHT_SKILLS": "-1", "SCORE_WEIGHT_SECTIONS": "nan"})
        self.assertAlmostEqual(sum(config.normalised_weights().values()), 1.0)
        self.assertEqual(config.weight_tfidf, 0.40)
        self.assertEqual(len(config.warnings), 3)

    def test_all_zero_weights_fall_back_to_defaults(self):
        config = ScoringConfig.from_env({"SCORE_WEIGHT_TFIDF": "0", "SCORE_WEIGHT_SKILLS": "0", "SCORE_WEIGHT_SECTIONS": "0"})
        self.assertEqual(config.normalised_weights()["skills"], 0.45)
        self.assertTrue(config.warnings)

    def test_weights_change_the_score(self):
        text_only = ScoringConfig(weight_tfidf=1, weight_skills=0, weight_sections=0)
        skills_only = ScoringConfig(weight_tfidf=0, weight_skills=1, weight_sections=0)
        a = score_resume(JD, STRONG, config=text_only)
        b = score_resume(JD, STRONG, config=skills_only)
        self.assertEqual(a["final_score"], a["tfidf_score"])
        self.assertEqual(b["final_score"], b["skill_coverage"])


class BreakdownTests(unittest.TestCase):
    def test_breakdown_has_all_documented_keys(self):
        result = score_resume(JD, STRONG)
        for key in ("final_score", "tfidf_score", "skill_coverage", "section_signal", "matched_skills",
                    "missing_skills", "top_terms", "warnings"):
            self.assertIn(key, result)

    def test_formula_matches_components(self):
        result = score_resume(JD, STRONG)
        w = result["weights"]
        expected = w["tfidf"] * result["tfidf_score"] + w["skills"] * result["skill_coverage"] + w["sections"] * result["section_signal"]
        self.assertAlmostEqual(result["final_score"], expected, delta=0.15)

    def test_strong_beats_weak_and_scores_are_bounded(self):
        strong = score_resume(JD, STRONG)
        weak = score_resume(JD, WEAK)
        self.assertGreater(strong["final_score"], weak["final_score"])
        for result in (strong, weak):
            self.assertGreaterEqual(result["final_score"], 0)
            self.assertLessEqual(result["final_score"], 100)
        self.assertEqual(strong["skill_coverage"], 100.0)
        self.assertEqual(strong["section_signal"], 100.0)

    def test_required_skills_come_from_job_description(self):
        required = required_skills(JD)
        for skill in ("Python", "SQL", "Docker", "AWS", "Machine learning", "REST APIs", "Data pipelines", "PostgreSQL"):
            self.assertIn(skill, required.skills)

    def test_section_signal_lower_when_skills_only_in_summary(self):
        no_sections = "Python SQL Docker AWS machine learning REST APIs data pipelines PostgreSQL"
        result = score_resume(JD, no_sections)
        self.assertEqual(result["section_signal"], 0.0)
        self.assertTrue(any("headings" in w for w in result["warnings"]))

    def test_explanation_mentions_formula(self):
        text = explanation_text(score_resume(JD, STRONG))
        self.assertIn("skill coverage", text)
        self.assertIn("Matched skills", text)

    def test_snippets_mark_matched_terms(self):
        result = score_resume(JD, STRONG)
        marked = [seg["text"] for snip in result["snippets"] for seg in snip["segments"] if seg["mark"]]
        self.assertIn("Python", marked)


class MustHaveTests(unittest.TestCase):
    def test_missing_must_have_caps_score(self):
        result = score_resume(JD, STRONG, must_have=["Kubernetes"])
        self.assertTrue(result["capped"])
        self.assertLessEqual(result["final_score"], 40.0)
        self.assertEqual(result["must_have_missing"], ["Kubernetes"])
        self.assertTrue(any("must-have" in w.lower() for w in result["warnings"]))

    def test_present_must_have_alias_does_not_cap(self):
        result = score_resume(JD, STRONG + "\nSkills: k8s", must_have=["kubernetes"])
        self.assertFalse(result["capped"])
        self.assertEqual(result["must_have_missing"], [])

    def test_custom_must_have_outside_taxonomy(self):
        result = score_resume(JD, STRONG + "\nSkills: Fivetran", must_have=["Fivetran"])
        self.assertIn("Fivetran", result["matched_skills"])

    def test_cap_is_configurable(self):
        result = score_resume(JD, STRONG, must_have=["Kubernetes"], config=ScoringConfig(must_have_cap=10))
        self.assertLessEqual(result["final_score"], 10.0)


class EdgeCaseTests(unittest.TestCase):
    def test_empty_resume(self):
        result = score_resume(JD, "")
        self.assertEqual(result["final_score"], 0.0)
        self.assertTrue(result["low_text"])
        self.assertTrue(any("scanned" in w for w in result["warnings"]))

    def test_short_resume_warns_but_still_scores(self):
        result = score_resume(JD, "Python and SQL")
        self.assertTrue(result["low_text"])
        self.assertGreater(result["final_score"], 0)

    def test_job_description_without_known_skills_uses_tfidf_only(self):
        result = score_resume("Friendly receptionist for a busy dental clinic.", "Receptionist at a dental clinic, friendly.")
        self.assertEqual(result["weights"], {"tfidf": 1.0, "skills": 0.0, "sections": 0.0})
        self.assertEqual(result["final_score"], result["tfidf_score"])

    def test_html_in_resume_is_plain_text(self):
        result = score_resume(JD, "<script>alert(1)</script> Python")
        joined = "".join(seg["text"] for snip in result["snippets"] for seg in snip["segments"])
        self.assertIn("<script>", joined)  # returned as data; the UI renders it with textContent


class BiasAndDeterminismTests(unittest.TestCase):
    def test_reordered_sections_score_identically(self):
        blocks = STRONG.strip().split("\n")
        # Rebuild as heading+body pairs and shuffle the pairs.
        pairs = [blocks[i:i + 2] for i in range(1, len(blocks), 2)]
        rng = random.Random(7)
        for _ in range(5):
            rng.shuffle(pairs)
            shuffled = "\n".join([blocks[0]] + [line for pair in pairs for line in pair])
            a = score_resume(JD, STRONG)
            b = score_resume(JD, shuffled)
            for key in ("final_score", "tfidf_score", "skill_coverage", "section_signal"):
                self.assertEqual(a[key], b[key], key)

    def test_reordered_lines_give_same_term_vector(self):
        lines = STRONG.strip().split("\n")
        self.assertEqual(sorted(analyzer("\n".join(lines))), sorted(analyzer("\n".join(reversed(lines)))))

    def test_batch_order_does_not_change_scores_or_ranking(self):
        resumes = [("a.pdf", STRONG), ("b.pdf", WEAK), ("c.pdf", STRONG.replace("Docker", ""))]
        first = rank_candidates(JD, resumes)
        second = rank_candidates(JD, list(reversed(resumes)))
        self.assertEqual([(r["name"], r["final_score"]) for r in first], [(r["name"], r["final_score"]) for r in second])

    def test_identical_content_identical_score(self):
        ranked = rank_candidates(JD, [("x.pdf", STRONG), ("y.pdf", STRONG)])
        self.assertEqual(ranked[0]["final_score"], ranked[1]["final_score"])
        self.assertEqual([r["name"] for r in ranked], ["x.pdf", "y.pdf"])  # ties broken by name

    def test_determinism_over_repeated_runs(self):
        results = {repr(score_resume(JD, STRONG, must_have=["AWS"])) for _ in range(5)}
        self.assertEqual(len(results), 1)

    def test_name_does_not_affect_score(self):
        a = score_resume(JD, STRONG.replace("Priya Example", "Alex Smith"))
        b = score_resume(JD, STRONG.replace("Priya Example", "Wei Zhang"))
        self.assertEqual(a["final_score"], b["final_score"])


class TextTests(unittest.TestCase):
    def test_normalize_handles_bullets_and_dashes(self):
        self.assertEqual(normalize_text("Skills•Python – SQL\r\n\r\n"), "Skills\nPython - SQL")

    def test_section_detection(self):
        kinds = [s.kind for s in detect_sections(normalize_text(STRONG))]
        self.assertEqual(kinds, ["other", "skills", "experience", "projects", "other"])

    def test_inline_heading(self):
        sections = detect_sections("Skills: Python, SQL\nEducation\nBSc")
        self.assertEqual(sections[0].kind, "skills")


if __name__ == "__main__":
    unittest.main()
