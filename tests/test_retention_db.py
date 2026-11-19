import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import sqlalchemy as sa

from talentlens import repository
from talentlens.config import load_settings, normalise_database_url
from talentlens.db import current_revision, metadata
from talentlens.importer import import_legacy_sqlite
from tests.helpers import AppTestCase, pdf_bytes, upload


def _ranked(score=50.0):
    return [{"rank": 1, "filename": "a.pdf", "score": score, "explanation": "x",
             "breakdown": {"label": "Partial match", "tfidf_score": 1.0, "skill_coverage": 1.0, "section_signal": 1.0}}]


class RetentionTests(AppTestCase):
    def _create(self, days_ago: int) -> int:
        return repository.create_screening(
            self.state.engine, job_title=f"old {days_ago}", job_description="jd", must_have=[], settings={},
            scoring_version="2.0", ranked=_ranked(), created_at=repository.utcnow() - timedelta(days=days_ago))

    def test_purge_older_than(self):
        old = self._create(40)
        new = self._create(5)
        self.assertEqual(repository.purge_older_than(self.state.engine, 30, dry_run=True), 1)
        self.assertIsNotNone(repository.get_screening(self.state.engine, old))
        self.assertEqual(repository.purge_older_than(self.state.engine, 30), 1)
        self.assertIsNone(repository.get_screening(self.state.engine, old))
        self.assertIsNotNone(repository.get_screening(self.state.engine, new))
        with self.state.engine.connect() as conn:
            orphans = conn.execute(sa.text("SELECT COUNT(*) FROM candidates WHERE screening_id = :i"), {"i": old}).scalar()
        self.assertEqual(orphans, 0)

    def test_cli_purge_old(self):
        self._create(40)
        result = self.app.test_cli_runner().invoke(args=["purge-old", "--dry-run"])
        self.assertIn("Would delete 1", result.output)
        result = self.app.test_cli_runner().invoke(args=["purge-old"])
        self.assertIn("Deleted 1", result.output)

    def test_purge_endpoint_requires_bearer_token(self):
        self._create(40)
        self.assertEqual(self.client.post("/api/v1/admin/purge").status_code, 401)
        self.assertEqual(self.client.post("/api/v1/admin/purge", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        ok = self.client.post("/api/v1/admin/purge", headers={"Authorization": "Bearer purge-token-0123456789abcdefghij"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.get_json()["deleted"], 1)


class PurgeDisabledTests(AppTestCase):
    env_overrides = {"PURGE_TOKEN": None}

    def test_disabled_without_token(self):
        self.assertEqual(self.client.post("/api/v1/admin/purge", headers={"Authorization": "Bearer x"}).status_code, 404)


class CliTests(AppTestCase):
    def test_hash_password(self):
        from werkzeug.security import check_password_hash

        result = self.app.test_cli_runner().invoke(args=["hash-password", "--password", "a-long-enough-password"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(check_password_hash(result.output.strip(), "a-long-enough-password"))

    def test_hash_password_rejects_short(self):
        result = self.app.test_cli_runner().invoke(args=["hash-password", "--password", "short"])
        self.assertNotEqual(result.exit_code, 0)

    def test_db_upgrade(self):
        result = self.app.test_cli_runner().invoke(args=["db-upgrade"])
        self.assertIn("revision 0001", result.output)


class MigrationTests(AppTestCase):
    def test_migration_matches_metadata(self):
        self.assertEqual(current_revision(self.state.engine), "0001")
        inspector = sa.inspect(self.state.engine)
        for table in metadata.sorted_tables:
            db_columns = {c["name"] for c in inspector.get_columns(table.name)}
            self.assertEqual(db_columns, {c.name for c in table.columns}, table.name)

    def test_upgrade_is_idempotent(self):
        from talentlens.db import upgrade

        upgrade(self.state.engine)
        upgrade(self.state.engine)


def make_legacy_db(path: Path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, job_title TEXT NOT NULL, job_description TEXT NOT NULL,
            candidate_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, screening_id INTEGER NOT NULL, filename TEXT NOT NULL,
            score REAL NOT NULL, matched_terms TEXT NOT NULL DEFAULT '[]', missing_terms TEXT NOT NULL DEFAULT '[]',
            explanation TEXT NOT NULL);
        INSERT INTO screenings VALUES (1, 'Old role', 'Python developer wanted', 2, '2026-10-01T10:00:00+00:00');
        INSERT INTO candidates VALUES (1, 1, 'a.pdf', 52.5, '["python"]', '["docker"]', 'Higher textual similarity.');
        INSERT INTO candidates VALUES (2, 1, 'b.pdf', 12.0, 'not json', '[]', 'Lower textual similarity.');
    """)
    conn.commit()
    conn.close()


class ImportTests(AppTestCase):
    def test_import_legacy_sqlite_and_skip_duplicates(self):
        legacy = Path(self._tmp.name) / "resume_ranker.db"
        make_legacy_db(legacy)
        dry = import_legacy_sqlite(self.state.engine, legacy, dry_run=True)
        self.assertEqual(dry["imported"], 1)
        self.assertEqual(repository.list_screenings(self.state.engine), [])
        summary = import_legacy_sqlite(self.state.engine, legacy)
        self.assertEqual(summary["candidates_imported"], 2)
        again = import_legacy_sqlite(self.state.engine, legacy)
        self.assertEqual(again["skipped_existing"], 1)
        screenings = repository.list_screenings(self.state.engine)
        self.assertEqual(len(screenings), 1)
        detail = repository.get_screening(self.state.engine, screenings[0]["id"])
        self.assertEqual(detail["scoring_version"], "1.x-import")
        self.assertEqual(detail["candidates"][0]["score"], 52.5)
        self.assertEqual(detail["candidates"][0]["matched_terms"], ["python"])
        self.assertEqual(detail["candidates"][1]["matched_terms"], [])
        self.assertTrue(detail["created_at"].startswith("2026-10-01"))

    def test_cli_import(self):
        legacy = Path(self._tmp.name) / "resume_ranker.db"
        make_legacy_db(legacy)
        result = self.app.test_cli_runner().invoke(args=["import-sqlite", str(legacy)])
        self.assertIn("imported 1", result.output)

    def test_imported_screening_renders_in_api(self):
        legacy = Path(self._tmp.name) / "resume_ranker.db"
        make_legacy_db(legacy)
        import_legacy_sqlite(self.state.engine, legacy)
        self.login()
        sid = self.client.get("/api/v1/screenings").get_json()["screenings"][0]["id"]
        detail = self.client.get(f"/api/v1/screenings/{sid}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/screenings/{sid}/export.csv").status_code, 200)

    def test_rejects_non_legacy_file(self):
        bogus = Path(self._tmp.name) / "other.db"
        sqlite3.connect(bogus).close()
        with self.assertRaises(ValueError):
            import_legacy_sqlite(self.state.engine, bogus)


class ConfigTests(unittest.TestCase):
    def test_zero_env_defaults_to_sqlite_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings({"DATA_DIR": tmp})
            self.assertEqual(settings.env, "development")
            self.assertTrue(settings.database_url.startswith("sqlite:///"))
            self.assertTrue(settings.secret_key_generated)
            self.assertFalse(settings.owner_login_enabled)
            self.assertEqual(settings.retention_days, 30)

    def test_render_implies_production_and_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings({"RENDER": "true", "DATA_DIR": tmp})
        self.assertTrue(settings.production)
        self.assertEqual(settings.trust_proxy, 1)

    def test_postgres_url_normalised(self):
        self.assertEqual(normalise_database_url("postgres://u:p@h/db"), "postgresql+psycopg2://u:p@h/db")
        self.assertEqual(normalise_database_url("postgresql://u:p@h/db?sslmode=require"), "postgresql+psycopg2://u:p@h/db?sslmode=require")

    def test_half_configured_owner_disables_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings({"ADMIN_USERNAME": "x", "DATA_DIR": tmp})
        self.assertFalse(settings.owner_login_enabled)
        self.assertTrue(any("ADMIN_PASSWORD_HASH" in p for p in settings.problems))

    def test_bad_numbers_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings({"RETENTION_DAYS": "abc", "MAX_FILE_MB": "999", "DATA_DIR": tmp})
        self.assertEqual(settings.retention_days, 30)
        self.assertEqual(settings.upload_limits.max_file_bytes, 5 * 1024 * 1024)

    def test_short_purge_token_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_settings({"PURGE_TOKEN": "short", "DATA_DIR": tmp}).purge_token)


class ZeroConfigAppTests(unittest.TestCase):
    def test_app_starts_with_no_environment(self):
        from talentlens import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app({"DATA_DIR": tmp})
            client = app.test_client()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/readyz").status_code, 200)
            response = client.post("/api/v1/screen", data={"job_description": "Python developer with SQL skills",
                                                           "resumes": [upload("a.pdf", pdf_bytes())]})
            self.assertEqual(response.status_code, 200)
            app.extensions["talentlens"].engine.dispose()


if __name__ == "__main__":
    unittest.main()
