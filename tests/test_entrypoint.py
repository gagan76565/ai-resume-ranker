import importlib
import os
import tempfile
import unittest
from unittest import mock


class DefaultHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {"DATA_DIR": cls._tmp.name, "SECRET_KEY": "x"}, clear=False):
            cls.entry = importlib.import_module("app")
        cls.entry.app.extensions["talentlens"].engine.dispose()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_localhost_by_default(self):
        self.assertEqual(self.entry.default_host({}), "127.0.0.1")

    def test_public_bind_on_render(self):
        self.assertEqual(self.entry.default_host({"RENDER": "true"}), "0.0.0.0")

    def test_explicit_host_wins(self):
        self.assertEqual(self.entry.default_host({"RENDER": "true", "HOST": "127.0.0.1"}), "127.0.0.1")
        self.assertEqual(self.entry.default_host({"HOST": "0.0.0.0"}), "0.0.0.0")


if __name__ == "__main__":
    unittest.main()
