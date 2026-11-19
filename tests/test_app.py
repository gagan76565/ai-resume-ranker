"""v1.x compatibility tests: the original routes and response shapes keep working."""

import unittest

from tests.helpers import AppTestCase, pdf_bytes, upload


class LegacyRouteTests(AppTestCase):
    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "service": "AI Resume Ranker"})

    def test_stats_start_empty_for_owner(self):
        self.login()
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["screenings"], 0)
        self.assertEqual(response.get_json()["candidates"], 0)

    def test_stats_require_login(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 401)
        self.assertIsInstance(response.get_json()["error"], str)  # v1.x error shape

    def test_rejects_short_job_description(self):
        response = self.client.post("/api/screen", data={"job_title": "Developer", "job_description": "too short"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("job description", response.get_json()["error"].lower())

    def test_legacy_screen_response_shape(self):
        response = self.client.post("/api/screen", data={
            "job_title": "Dev", "job_description": "Python developer with SQL database and API design experience",
            "resumes": [upload("cv.pdf", pdf_bytes())]}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for key in ("screening_id", "job_title", "created_at", "candidate_count", "candidates", "file_errors"):
            self.assertIn(key, body)
        candidate = body["candidates"][0]
        for key in ("filename", "score", "matched_terms", "missing_terms", "explanation"):
            self.assertIn(key, candidate)
        self.assertGreaterEqual(candidate["score"], 0)
        self.assertLessEqual(candidate["score"], 100)

    def test_legacy_no_readable_resumes_keeps_file_errors_key(self):
        response = self.client.post("/api/screen", data={
            "job_description": "Python developer with SQL database experience",
            "resumes": [upload("x.pdf", b"junk")]}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["file_errors"][0]["filename"], "x.pdf")

    def test_missing_screening_returns_404(self):
        self.login()
        response = self.client.get("/api/screenings/999")
        self.assertEqual(response.status_code, 404)

    def test_legacy_list_is_bare_array_and_export(self):
        self.login()
        saved = self.client.post("/api/screen", data={
            "job_description": "Python developer with SQL database experience",
            "resumes": [upload("cv.pdf", pdf_bytes())]}, headers={"X-CSRF-Token": self.csrf()},
            content_type="multipart/form-data").get_json()
        listing = self.client.get("/api/screenings").get_json()
        self.assertIsInstance(listing, list)
        self.assertEqual(listing[0]["id"], saved["screening_id"])
        csv_response = self.client.get(f"/api/export.csv?screening_id={saved['screening_id']}")
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(self.client.get("/api/export.csv").status_code, 400)


if __name__ == "__main__":
    unittest.main()
