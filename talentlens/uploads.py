"""Upload validation and PDF text extraction.

Uploaded bytes are only ever parsed by ``pypdf`` for text. They are never
written to disk by this module, never executed and never served back to a
browser. Client-supplied file names are reduced with ``secure_filename`` and
are used only as display labels.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import PurePath

from pypdf import PdfReader
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

logging.getLogger("pypdf").setLevel(logging.ERROR)  # pypdf warnings can quote document content

PDF_MAGIC = b"%PDF-"
MB = 1024 * 1024


@dataclass(frozen=True)
class UploadLimits:
    """Size and count limits. All sizes are in bytes."""

    max_file_bytes: int = 5 * MB
    max_files: int = 20
    max_total_bytes: int = 30 * MB
    max_pages: int = 30
    max_chars: int = 100_000


@dataclass
class AcceptedFile:
    """A file that passed validation, held in memory."""

    name: str
    data: bytes


@dataclass
class ValidationResult:
    accepted: list[AcceptedFile] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)  # [{"filename": ..., "error": ...}]
    batch_error: str | None = None


@dataclass
class Extraction:
    text: str
    pages: int
    warnings: list[str] = field(default_factory=list)


class PdfReadError(Exception):
    """A PDF that cannot be read; the message is safe to show to users."""


def display_name(raw: str | None, index: int) -> str:
    """Sanitised file name for display. Never used as a filesystem path.

    ``secure_filename`` strips directories (``../../etc/passwd`` → ``etc_passwd``)
    and non-ASCII characters; the result is truncated to 120 characters.
    """
    base = PurePath((raw or "").replace("\\", "/")).name
    cleaned = secure_filename(base)[:120]
    return cleaned or f"resume-{index}.pdf"


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / MB:.1f} MB"


def validate_uploads(files: list[FileStorage], limits: UploadLimits) -> ValidationResult:
    """Check count, extension, size, total size and the ``%PDF-`` magic bytes.

    Invalid files are reported individually by name and skipped; the rest are
    accepted. A batch-level error (too many files, nothing selected) rejects
    the whole request.
    """
    result = ValidationResult()
    files = [f for f in files if f is not None and (f.filename or "").strip()]
    if not files:
        result.batch_error = "Select at least one PDF resume."
        return result
    if len(files) > limits.max_files:
        result.batch_error = f"You selected {len(files)} files; the limit is {limits.max_files} per screening."
        return result

    total = 0
    used_names: set[str] = set()
    for index, storage in enumerate(files, start=1):
        name = display_name(storage.filename, index)
        stem, dot, suffix = name.rpartition(".")
        candidate, n = name, 2
        while candidate.lower() in used_names:
            candidate = f"{stem}-{n}.{suffix}" if dot else f"{name}-{n}"
            n += 1
        name = candidate
        used_names.add(name.lower())

        if not name.lower().endswith(".pdf"):
            result.errors.append({"filename": name, "error": f"{name} is not a .pdf file. Only PDF resumes are supported."})
            continue
        data = storage.stream.read(limits.max_file_bytes + 1)
        if len(data) > limits.max_file_bytes:
            result.errors.append({
                "filename": name,
                "error": f"{name} is larger than the {_format_mb(limits.max_file_bytes)} per-file limit.",
            })
            continue
        if not data:
            result.errors.append({"filename": name, "error": f"{name} is empty."})
            continue
        if not data.startswith(PDF_MAGIC):
            result.errors.append({
                "filename": name,
                "error": f"{name} does not look like a real PDF (it is missing the %PDF header). Re-export it as PDF.",
            })
            continue
        if total + len(data) > limits.max_total_bytes:
            result.errors.append({
                "filename": name,
                "error": f"{name} was skipped because the batch would exceed the {_format_mb(limits.max_total_bytes)} total limit.",
            })
            continue
        total += len(data)
        result.accepted.append(AcceptedFile(name, data))
    return result


def extract_text(data: bytes, limits: UploadLimits) -> Extraction:
    """Extract text from PDF bytes with page and character caps.

    Raises :class:`PdfReadError` for password-protected or unreadable files.
    """
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - pypdf raises various types
                raise PdfReadError("This PDF is password-protected. Remove the password and upload it again.") from exc
            if not ok:
                raise PdfReadError("This PDF is password-protected. Remove the password and upload it again.")
        page_count = len(reader.pages)
    except PdfReadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PdfReadError("This PDF could not be read. Try exporting it again as a text-based PDF.") from exc

    warnings: list[str] = []
    if page_count > limits.max_pages:
        warnings.append(f"Only the first {limits.max_pages} of {page_count} pages were read.")
    parts: list[str] = []
    length = 0
    for page_index in range(min(page_count, limits.max_pages)):
        try:
            text = reader.pages[page_index].extract_text() or ""
        except Exception:  # noqa: BLE001 - a single bad page should not sink the file
            warnings.append(f"Page {page_index + 1} could not be read and was skipped.")
            continue
        parts.append(text)
        length += len(text)
        if length >= limits.max_chars:
            warnings.append(f"Text was truncated to the first {limits.max_chars:,} characters.")
            break
    text = "\n".join(parts)[: limits.max_chars].strip()
    return Extraction(text=text, pages=page_count, warnings=warnings)
