import json
import unittest

from tests.helpers import JD, WEAK_LINES, AppTestCase, pdf_bytes, upload


class HealthTests(AppTestCase):
    def test_healthz_and_readyz(self):
        self.assertEqual(self.client.get("/healthz").get_json(), {"status": "ok"})
        ready = self.client.get("/readyz")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.get_json()["database"], "ok")

    def test_readyz_reports_database_failure(self):
        from unittest import mock

        with mock.patch("talentlens.views.ping", side_effect=RuntimeError("down")):
            response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("down", response.get_data(as_text=True))  # no internals leaked

    def test_v1_health(self):
        body = self.client.get("/api/v1/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["version"], "2.0.0")

    def test_config_is_public_and_has_no_secrets(self):
        body = self.client.get("/api/v1/config").get_json()
        self.assertEqual(body["weights"], {"tfidf": 0.4, "skills": 0.45, "sections": 0.15})
        text = json.dumps(body)
        self.assertNotIn("test-secret-key", text)
        self.assertNotIn("purge-token", text)

    def test_skills_endpoint(self):
        body = self.client.get("/api/v1/skills").get_json()
        self.assertGreaterEqual(body["count"], 150)


class ScreenTests(AppTestCase):
    def test_public_screening_ranks_and_stores_nothing(self):
        response = self.screen(files=[upload("weak.pdf", pdf_bytes(WEAK_LINES)), upload("strong.pdf", pdf_bytes())])
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertFalse(body["saved"])
        self.assertIsNone(body["screening_id"])
        self.assertEqual([c["filename"] for c in body["candidates"]], ["strong.pdf", "weak.pdf"])
        top = body["candidates"][0]
        for key in ("score", "matched_terms", "missing_terms", "explanation", "breakdown"):
            self.assertIn(key, top)
        for key in ("final_score", "tfidf_score", "skill_coverage", "section_signal", "matched_skills",
                    "missing_skills", "top_terms", "warnings"):
            self.assertIn(key, top["breakdown"])
        self.assertEqual(body["summary"]["candidate_count"], 2)
        # nothing persisted
        self.login()
        self.assertEqual(self.client.get("/api/v1/stats").get_json()["screenings"], 0)

    def test_must_have_cap_via_form(self):
        response = self.screen(files=[upload("strong.pdf", pdf_bytes())], must_have=["Kubernetes"])
        candidate = response.get_json()["candidates"][0]
        self.assertLessEqual(candidate["score"], 40)
        self.assertEqual(candidate["breakdown"]["must_have_missing"], ["Kubernetes"])

    def test_validation_error_shape_v1(self):
        response = self.screen(files=[upload("a.pdf", pdf_bytes())], job_description="short")
        self.assertEqual(response.status_code, 400)
        error = response.get_json()["error"]
        self.assertEqual(error["code"], "validation_error")
        self.assertIn("job description", error["message"].lower())
        self.assertEqual(error["details"][0]["field"], "$.job_description")

    def test_too_many_must_haves(self):
        response = self.screen(files=[upload("a.pdf", pdf_bytes())], must_have=[f"skill{i}" for i in range(21)])
        self.assertEqual(response.status_code, 400)

    def test_per_file_errors_are_reported(self):
        response = self.screen(files=[upload("good.pdf", pdf_bytes()), upload("bad.pdf", b"not a pdf")])
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["file_errors"][0]["filename"], "bad.pdf")

    def test_all_files_bad_returns_422(self):
        response = self.screen(files=[upload("bad.pdf", b"nope")])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "no_readable_resumes")

    def test_no_files(self):
        response = self.screen(files=[])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_upload")

    def test_low_text_pdf_warns_instead_of_silent_zero(self):
        response = self.screen(files=[upload("scan.pdf", pdf_bytes(["Scan"]))])
        candidate = response.get_json()["candidates"][0]
        self.assertTrue(candidate["breakdown"]["low_text"])
        self.assertTrue(any("scanned" in w for w in candidate["breakdown"]["warnings"]))

    def test_samples(self):
        info = self.client.get("/api/v1/samples").get_json()
        self.assertIn("Synthetic", info["label"])
        self.assertEqual(len(info["resumes"]), 6)
        response = self.client.post("/api/v1/samples/screen", json={})
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["sample_data"])
        self.assertEqual(body["candidate_count"], 6)
        via_form = self.screen(use_samples="1", job_description=info["job_description"])
        self.assertEqual(via_form.get_json()["candidate_count"], 6)

    def test_rate_limit(self):
        limiter = self.state.limiters["screen"]
        limiter.limit = 2
        for _ in range(2):
            self.assertEqual(self.client.post("/api/v1/score", json={"job_description": JD, "resumes": [{"name": "a", "text": "Python"}]}).status_code, 200)
        blocked = self.client.post("/api/v1/score", json={"job_description": JD, "resumes": [{"name": "a", "text": "Python"}]})
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.get_json()["error"]["code"], "rate_limited")


class ScoreJsonTests(AppTestCase):
    def test_score_endpoint(self):
        body = {"job_description": JD, "must_have": ["Python"],
                "resumes": [{"name": "a", "text": "Skills\nPython SQL Docker AWS"}, {"name": "b", "text": "Cooking"}]}
        response = self.client.post("/api/v1/score", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["candidates"][0]["filename"], "a")

    def test_schema_rejects_unknown_fields_and_bad_types(self):
        response = self.client.post("/api/v1/score", json={"job_description": JD, "resumes": "nope", "extra": 1})
        details = response.get_json()["error"]["details"]
        fields = {d["field"] for d in details}
        self.assertIn("$.extra", fields)
        self.assertIn("$.resumes", fields)

    def test_invalid_json(self):
        response = self.client.post("/api/v1/score", data="{", content_type="application/json")
        self.assertEqual(response.get_json()["error"]["code"], "invalid_json")

    def test_duplicate_names(self):
        body = {"job_description": JD, "resumes": [{"name": "a", "text": "x"}, {"name": "a", "text": "y"}]}
        self.assertEqual(self.client.post("/api/v1/score", json=body).status_code, 400)


class OwnerApiTests(AppTestCase):
    def _saved(self):
        self.login()
        response = self.screen(files=[upload("strong.pdf", pdf_bytes()), upload("weak.pdf", pdf_bytes(WEAK_LINES))], token=self.csrf())
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def test_owner_screening_is_saved_with_ids(self):
        body = self._saved()
        self.assertTrue(body["saved"])
        self.assertTrue(all(c["id"] for c in body["candidates"]))
        listing = self.client.get("/api/v1/screenings").get_json()
        self.assertEqual(listing["count"], 1)
        detail = self.client.get(f"/api/v1/screenings/{body['screening_id']}").get_json()
        self.assertEqual(detail["candidates"][0]["filename"], "strong.pdf")
        self.assertIn("summary", detail)

    def test_owner_can_opt_out_of_saving(self):
        self.login()
        response = self.screen(files=[upload("a.pdf", pdf_bytes())], token=self.csrf(), save="0")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["saved"])

    def test_search(self):
        self._saved()
        self.assertEqual(self.client.get("/api/v1/screenings?q=engineer").get_json()["count"], 1)
        self.assertEqual(self.client.get("/api/v1/screenings?q=zzz%25").get_json()["count"], 0)

    def test_patch_candidate_notes_and_shortlist(self):
        body = self._saved()
        sid, cid = body["screening_id"], body["candidates"][0]["id"]
        response = self.client.patch(f"/api/v1/screenings/{sid}/candidates/{cid}", json={"shortlisted": True, "notes": "Call back"},
                                     headers={"X-CSRF-Token": self.csrf()})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["shortlisted"])
        self.assertEqual(response.get_json()["notes"], "Call back")
        bad = self.client.patch(f"/api/v1/screenings/{sid}/candidates/{cid}", json={"shortlisted": "yes"},
                                headers={"X-CSRF-Token": self.csrf()})
        self.assertEqual(bad.status_code, 400)
        missing = self.client.patch(f"/api/v1/screenings/{sid}/candidates/9999", json={"notes": "x"},
                                    headers={"X-CSRF-Token": self.csrf()})
        self.assertEqual(missing.status_code, 404)

    def test_delete(self):
        body = self._saved()
        sid = body["screening_id"]
        self.assertEqual(self.client.delete(f"/api/v1/screenings/{sid}", headers={"X-CSRF-Token": self.csrf()}).status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/screenings/{sid}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/screenings/{sid}", headers={"X-CSRF-Token": self.csrf()}).status_code, 404)

    def test_csv_export(self):
        body = self._saved()
        response = self.client.get(f"/api/v1/screenings/{body['screening_id']}/export.csv")
        text = response.get_data(as_text=True)
        self.assertEqual(response.headers["Content-Type"], "text/csv; charset=utf-8")
        header = text.lstrip("﻿").splitlines()[0]
        self.assertTrue(header.startswith("Candidate PDF,Text similarity (%),Matched terms,Terms not found,Explanation"))
        self.assertIn("strong.pdf", text)

    def test_saved_results_page(self):
        body = self._saved()
        self.assertEqual(self.client.get(f"/screenings/{body['screening_id']}").status_code, 200)
        self.assertEqual(self.client.get("/history").status_code, 200)

    def test_anonymous_cannot_read_history(self):
        body = self._saved()
        self.client.post("/logout", data={"csrf_token": self.csrf()})
        for path in ("/api/v1/screenings", f"/api/v1/screenings/{body['screening_id']}", "/api/v1/stats",
                     f"/api/v1/screenings/{body['screening_id']}/export.csv"):
            self.assertEqual(self.client.get(path).status_code, 401, path)


class CsvSafetyTests(unittest.TestCase):
    def test_formula_injection_neutralised(self):
        from talentlens.api import csv_safe

        self.assertEqual(csv_safe("=HYPERLINK(1)"), "'=HYPERLINK(1)")
        self.assertEqual(csv_safe("@SUM"), "'@SUM")
        self.assertEqual(csv_safe("Normal"), "Normal")
        self.assertEqual(csv_safe(42.5), "42.5")


class ErrorPageTests(AppTestCase):
    def test_404_html_and_json(self):
        page = self.client.get("/does-not-exist")
        self.assertEqual(page.status_code, 404)
        self.assertIn(b"We could not find that page", page.data)
        api = self.client.get("/api/v1/nope")
        self.assertEqual(api.get_json()["error"]["code"], "not_found")

    def test_500_is_friendly_and_logged(self):
        @self.app.get("/boom")
        def boom():
            raise RuntimeError("secret internals")

        with self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.get("/boom")
        self.assertEqual(response.status_code, 500)
        self.assertIn(b"Something went wrong", response.data)
        self.assertNotIn(b"secret internals", response.data)

    def test_413(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1000
        response = self.screen(files=[upload("big.pdf", b"%PDF-" + b"0" * 5000)])
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"]["code"], "payload_too_large")


if __name__ == "__main__":
    unittest.main()
