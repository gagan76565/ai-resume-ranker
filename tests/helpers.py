"""Shared test helpers: isolated app instances and PDF fixtures."""

from __future__ import annotations

import io
import os
import re
import tempfile
import unittest

from werkzeug.security import generate_password_hash

from talentlens import create_app
from talentlens.minipdf import build_pdf

OWNER = "owner"
PASSWORD = "correct horse battery staple"
# Fast hash for tests only (production uses the scrypt default of `flask hash-password`).
PASSWORD_HASH = generate_password_hash(PASSWORD, method="pbkdf2:sha256:1000")

JD = "Junior data engineer. Required: Python, SQL, Docker and AWS. You will build REST APIs and data pipelines."

STRONG_LINES = [
    "# Skills", "Python, SQL, Docker, AWS, REST APIs",
    "# Experience", "Built data pipelines and REST APIs in Python on AWS with Docker.",
] + [f"Delivered project {i} with Python and SQL for the analytics team." for i in range(8)]
WEAK_LINES = ["# Experience", "Retail supervisor managing rotas, stock and customer service for a busy store."] * 4


def pdf_bytes(lines=STRONG_LINES) -> bytes:
    return build_pdf(list(lines))


def upload(name: str, data: bytes):
    return (io.BytesIO(data), name, "application/pdf")


def base_env(tmpdir: str, **overrides) -> dict:
    env = {
        "APP_ENV": "testing",
        "SECRET_KEY": "test-secret-key-not-for-production",
        "DATABASE_URL": os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmpdir}/test.db".replace("\\", "/"),
        "ADMIN_USERNAME": OWNER,
        "ADMIN_PASSWORD_HASH": PASSWORD_HASH,
        "PURGE_TOKEN": "purge-token-0123456789abcdefghij",
    }
    env.update({k: v for k, v in overrides.items() if v is not None})
    for key in [k for k, v in overrides.items() if v is None]:
        env.pop(key, None)
    return env


class AppTestCase(unittest.TestCase):
    """Fresh app + database per test."""

    env_overrides: dict = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.app = create_app(base_env(self._tmp.name, **self.env_overrides))
        self.state = self.app.extensions["talentlens"]
        self.client = self.app.test_client()
        if os.environ.get("TEST_DATABASE_URL"):
            from talentlens.db import candidates, screenings

            with self.state.engine.begin() as conn:
                conn.execute(candidates.delete())
                conn.execute(screenings.delete())

    def tearDown(self):
        self.state.engine.dispose()
        self._tmp.cleanup()

    # -- helpers
    def csrf(self) -> str:
        html = self.client.get("/").get_data(as_text=True)
        return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)

    def login(self, username=OWNER, password=PASSWORD):
        token = self.csrf()
        return self.client.post("/login", data={"username": username, "password": password, "csrf_token": token})

    def screen(self, files=None, token=None, **fields):
        data = {"job_title": "Data engineer", "job_description": JD}
        data.update(fields)
        if files is not None:
            data["resumes"] = files
        headers = {"X-CSRF-Token": token} if token else {}
        return self.client.post("/api/v1/screen", data=data, headers=headers, content_type="multipart/form-data")
