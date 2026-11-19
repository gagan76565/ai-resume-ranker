"""Upload validation (PDF and DOCX) and a compatibility text-extraction helper.

Uploaded bytes are only parsed for text by ``talentlens.documents`` (PyMuPDF /
python-docx). They are never written to disk, executed or served back to a
browser. The format is identified from magic bytes, never from the client's
content type. Client-supplied file names are reduced with ``secure_filename``
and used only as display labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .documents import DocumentError, parse_document, sniff

PDF_MAGIC = b"%PDF-"
MB = 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx"}
PdfReadError = DocumentError  # backwards-compatible name


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

        extension = "." + suffix.lower() if dot else ""
        expected_kind = ALLOWED_EXTENSIONS.get(extension)
        if expected_kind is None:
            result.errors.append({"filename": name, "error": f"{name} is not a PDF or DOCX file. Only .pdf and .docx resumes are supported."})
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
        if sniff(data) != expected_kind:
            message = (
                f"{name} does not look like a real PDF (it is missing the %PDF header). Re-export it as PDF."
                if expected_kind == "pdf" else
                f"{name} does not look like a real Word .docx file. Save it again from Word or export it as PDF."
            )
            result.errors.append({"filename": name, "error": message})
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
    """Compatibility helper: flat text of a PDF/DOCX (see :func:`talentlens.documents.parse_document`).

    Raises :class:`PdfReadError` (= ``DocumentError``) for password-protected or unreadable files.
    """
    doc = parse_document("upload", data, limits, use_ocr=False)
    pages = doc.quality.pages
    return Extraction(text=doc.text[: limits.max_chars].strip(), pages=pages, warnings=list(doc.warnings))
