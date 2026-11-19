import json
import unittest

from scoring.skills import DATA_PATH, build_taxonomy, load_taxonomy


def found(text):
    return [match.canonical for match in load_taxonomy().find(text)]


class TaxonomyFileTests(unittest.TestCase):
    def test_has_at_least_150_skills_with_required_fields(self):
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["skills"]), 150)
        for entry in data["skills"]:
            self.assertTrue(entry["canonical"])
            self.assertIsInstance(entry["aliases"], list)
            self.assertTrue(entry["category"])

    def test_duplicate_alias_across_skills_is_rejected(self):
        with self.assertRaises(ValueError):
            build_taxonomy([
                {"canonical": "A1", "aliases": ["shared"], "category": "x"},
                {"canonical": "B1", "aliases": ["shared"], "category": "x"},
            ])


class AliasMatchingTests(unittest.TestCase):
    def test_required_alias_pairs(self):
        pairs = {
            "js": "JavaScript", "javascript": "JavaScript", "ml": "Machine learning",
            "machine learning": "Machine learning", "k8s": "Kubernetes", "kubernetes": "Kubernetes",
            "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "node": "Node.js", "node.js": "Node.js",
            "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
        }
        for alias, canonical in pairs.items():
            with self.subTest(alias=alias):
                self.assertEqual(found(f"Experienced with {alias} daily."), [canonical])

    def test_case_insensitive(self):
        self.assertEqual(found("PYTHON, python, PyThOn"), ["Python", "Python", "Python"])

    def test_punctuation_skills(self):
        self.assertEqual(found("C++ and C# and Node.js."), ["C++", "C#", "Node.js"])
        self.assertEqual(found("Built with .NET and ASP.NET"), [".NET", "ASP.NET"])
        self.assertEqual(found("CI/CD pipelines"), ["CI/CD"])

    def test_multiword_aliases_tolerate_extra_spaces(self):
        self.assertEqual(found("machine   learning"), ["Machine learning"])

    def test_canonical_lookup(self):
        taxonomy = load_taxonomy()
        self.assertEqual(taxonomy.canonical_for("K8S"), "Kubernetes")
        self.assertEqual(taxonomy.canonical_for("C"), "C")
        self.assertIsNone(taxonomy.canonical_for("basket weaving"))


class FalsePositiveGuardTests(unittest.TestCase):
    def test_java_does_not_match_javascript(self):
        self.assertEqual(found("JavaScript only"), ["JavaScript"])

    def test_c_and_r_do_not_match_inside_words(self):
        for text in ("cloud computing", "react router", "circuit", "R&D department", "C-level executives",
                     "rust-coloured", "car", "Grade-C"):
            with self.subTest(text=text):
                self.assertNotIn("C", found(text))
                self.assertNotIn("R", found(text))

    def test_c_does_not_match_cpp_or_csharp(self):
        self.assertEqual(found("C++"), ["C++"])
        self.assertEqual(found("C#"), ["C#"])

    def test_single_letter_lowercase_does_not_count(self):
        self.assertEqual(found("plan a, b or c"), [])

    def test_node_alias_does_not_double_count_nodejs(self):
        self.assertEqual(found("Node.js"), ["Node.js"])

    def test_longest_match_wins(self):
        self.assertEqual(found("Microsoft SQL Server"), ["SQL Server"])
        self.assertEqual(found("React Native apps"), ["React Native"])

    def test_common_words_are_case_sensitive(self):
        self.assertEqual(found("I excel at go-getting and swift delivery"), [])
        self.assertEqual(found("Excel and Swift"), ["Microsoft Excel", "Swift"])

    def test_rest_of_is_not_rest_api(self):
        self.assertEqual(found("the rest of the team"), [])


if __name__ == "__main__":
    unittest.main()
