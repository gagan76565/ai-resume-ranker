"""Consistent error responses.

``/api/v1/*`` errors look like::

    {"error": {"code": "validation_error", "message": "...", "details": [...]}}

Legacy ``/api/*`` routes keep the v1.x shape ``{"error": "message"}`` and add
``code`` (and ``details`` when present) as extra top-level keys, so old
clients that read ``error`` as a string keep working.
"""

from __future__ import annotations

from flask import jsonify, render_template, request
from werkzeug.exceptions import HTTPException


class ApiError(Exception):
    """An error with a stable machine-readable ``code`` and a user-safe ``message``."""

    def __init__(self, status: int, code: str, message: str, details=None, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers or {}


def wants_json() -> bool:
    return request.path.startswith("/api/") or request.path in {"/healthz", "/readyz"}


def error_payload(code: str, message: str, details=None) -> dict:
    if request.path.startswith("/api/v1/"):
        body = {"code": code, "message": message}
        if details is not None:
            body["details"] = details
        return {"error": body}
    body = {"error": message, "code": code}
    if details is not None:
        body["details"] = details
        if code == "no_readable_resumes":
            body["file_errors"] = details  # v1.x key
    return body


def error_response(status: int, code: str, message: str, details=None, headers: dict | None = None):
    if wants_json():
        response = jsonify(error_payload(code, message, details))
    else:
        response = render_template("error.html", status=status, code=code, message=message)
    return response, status, headers or {}


_HTTP_CODES = {
    400: ("bad_request", "The request could not be understood."),
    401: ("unauthorized", "Please sign in to continue."),
    403: ("forbidden", "You do not have access to this resource."),
    404: ("not_found", "We could not find that page or resource."),
    405: ("method_not_allowed", "That action is not allowed here."),
    413: ("payload_too_large", "The upload is too large."),
    429: ("rate_limited", "Too many requests. Please wait a moment and try again."),
}


def register_error_handlers(app):
    @app.errorhandler(ApiError)
    def _api_error(error: ApiError):
        return error_response(error.status, error.code, error.message, error.details, error.headers)

    @app.errorhandler(HTTPException)
    def _http_error(error: HTTPException):
        code, message = _HTTP_CODES.get(error.code or 500, ("http_error", error.description or "Request failed."))
        if error.code == 413:
            limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
            message = f"The total upload is too large. Keep all files together under about {limit} MB."
        return error_response(error.code or 500, code, message)

    @app.errorhandler(Exception)
    def _unexpected(error: Exception):
        # Log the exception type and traceback but never the request body.
        app.logger.exception("Unhandled error on %s %s", request.method, request.path)
        return error_response(500, "internal_error", "Something went wrong on our side. Please try again.")
