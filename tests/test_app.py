import tempfile
import unittest
from pathlib import Path

import app as application


class ResumeRankerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = application.DATABASE_PATH
        application.DATABASE_PATH = Path(self.temp_dir.name) / "test.db"
        application.init_db()
        self.client = application.app.test_client()

    def tearDown(self):
        application.DATABASE_PATH = self.original_database_path
        self.temp_dir.cleanup()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_stats_start_empty(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["screenings"], 0)
        self.assertEqual(response.get_json()["candidates"], 0)

    def test_rejects_short_job_description(self):
        response = self.client.post(
            "/api/screen",
            data={"job_title": "Developer", "job_description": "too short"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("job description", response.get_json()["error"].lower())

    def test_similarity_is_bounded_and_explained(self):
        score, matched, missing, explanation = application.analyse_resume(
            "Python developer with SQL database and API design experience",
            "Python developer building REST API services and working with SQL databases",
        )
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertTrue(matched)
        self.assertIsInstance(missing, list)
        self.assertIn("similarity", explanation.lower())

    def test_missing_screening_returns_404(self):
        response = self.client.get("/api/screenings/999")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
