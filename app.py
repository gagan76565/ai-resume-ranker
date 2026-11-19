"""WSGI entry point: ``gunicorn app:app`` (see Procfile) or ``python app.py`` for local development."""

import os

try:  # optional: load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from talentlens import create_app  # noqa: E402

app = create_app()


def default_host(env=os.environ) -> str:
    """Bind publicly on hosting platforms (Render sets RENDER), localhost elsewhere. HOST always wins."""
    if env.get("HOST"):
        return env["HOST"]
    return "0.0.0.0" if env.get("RENDER") else "127.0.0.1"  # noqa: S104 - public bind is required on Render


if __name__ == "__main__":
    app.run(host=default_host(), port=int(os.environ.get("PORT", "5000")), debug=False)
