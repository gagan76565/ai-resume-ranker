import unittest

from tests.helpers import OWNER, PASSWORD, AppTestCase


class LoginTests(AppTestCase):
    def test_login_page_renders(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="csrf_token"', response.data)

    def test_successful_login_and_logout(self):
        response = self.login()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 200)
        token = self.csrf()
        self.client.post("/logout", data={"csrf_token": token})
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 401)

    def test_wrong_password_is_rejected(self):
        response = self.login(password="wrong")
        self.assertEqual(response.status_code, 401)
        self.assertIn(b"Incorrect username or password", response.data)
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 401)

    def test_wrong_username_is_rejected(self):
        self.assertEqual(self.login(username="admin").status_code, 401)

    def test_login_rate_limited_after_five_failures(self):
        for _ in range(5):
            self.assertEqual(self.login(password="nope").status_code, 401)
        blocked = self.login()  # even the correct password is refused while blocked
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)

    def test_login_requires_csrf(self):
        response = self.client.post("/login", data={"username": OWNER, "password": PASSWORD})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 401)

    def test_open_redirect_is_blocked(self):
        token = self.csrf()
        response = self.client.post("/login", data={"username": OWNER, "password": PASSWORD, "csrf_token": token,
                                                    "next": "https://evil.example/"})
        self.assertEqual(response.headers["Location"], "/")
        self.client.get("/")

    def test_next_redirect_within_site(self):
        token = self.csrf()
        response = self.client.post("/login", data={"username": OWNER, "password": PASSWORD, "csrf_token": token, "next": "/history"})
        self.assertEqual(response.headers["Location"], "/history")

    def test_history_page_redirects_anonymous_to_login(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_session_cookie_flags(self):
        response = self.login()
        cookie = response.headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertNotIn("Secure", cookie)  # testing env; production sets Secure (see ProductionCookieTests)

    def test_password_hash_change_invalidates_session(self):
        self.login()
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 200)
        from dataclasses import replace

        self.state.settings = replace(self.state.settings, admin_password_hash="pbkdf2:sha256:1000$x$y")
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 401)


class ProductionCookieTests(AppTestCase):
    env_overrides = {"APP_ENV": "production"}

    def test_secure_flag_and_hsts_in_production(self):
        response = self.login()
        self.assertIn("Secure", response.headers.get("Set-Cookie", ""))
        self.assertIn("Strict-Transport-Security", self.client.get("/healthz").headers)


class LoginDisabledTests(AppTestCase):
    env_overrides = {"ADMIN_USERNAME": None, "ADMIN_PASSWORD_HASH": None}

    def test_login_disabled_without_credentials(self):
        self.assertIn(b"not configured", self.client.get("/login").data)
        self.assertEqual(self.login().status_code, 403)


class CsrfTests(AppTestCase):
    def test_anonymous_screening_needs_no_token(self):
        from tests.helpers import pdf_bytes, upload

        response = self.screen(files=[upload("a.pdf", pdf_bytes())])
        self.assertEqual(response.status_code, 200)

    def test_owner_screening_requires_token(self):
        from tests.helpers import pdf_bytes, upload

        self.login()
        response = self.screen(files=[upload("a.pdf", pdf_bytes())])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "csrf_failed")
        ok = self.screen(files=[upload("a.pdf", pdf_bytes())], token=self.csrf())
        self.assertEqual(ok.status_code, 201)

    def test_delete_requires_token(self):
        self.login()
        response = self.client.delete("/api/v1/screenings/1")
        self.assertEqual(response.status_code, 400)

    def test_wrong_token_rejected(self):
        self.login()
        response = self.client.delete("/api/v1/screenings/1", headers={"X-CSRF-Token": "forged"})
        self.assertEqual(response.status_code, 400)

    def test_logout_requires_token(self):
        self.login()
        self.assertEqual(self.client.post("/logout").status_code, 400)
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 200)


class SecurityHeaderTests(AppTestCase):
    def test_headers_present(self):
        response = self.client.get("/")
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

    def test_api_not_cached(self):
        self.assertEqual(self.client.get("/api/v1/health").headers["Cache-Control"], "no-store")

    def test_templates_have_no_inline_scripts_or_styles(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("<script>", html)
        self.assertNotIn("style=", html)
        self.assertNotIn("<style", html)


if __name__ == "__main__":
    unittest.main()
