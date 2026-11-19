"""Document processing pipeline: identify → extract (per page) → normalise → grade quality.

    doc = parse_document("cv.pdf", data, limits)
    doc.text, doc.pages, doc.quality.grade, doc.page_at(offset)

Raises :class:`DocumentError` for unreadable files. A scanned PDF without OCR is *not*
an error: it is returned with ``quality.grade == "needs_ocr"`` so the caller can report it
instead of silently scoring it.
"""

from __future__ import annotations

import os

from . import ocr
from .model import DocumentError, Page, ParsedDocument, Quality
from .parsers import extract_docx, extract_pdf, sniff
from .quality import grade

__all__ = ["DocumentError", "Page", "ParsedDocument", "Quality", "ocr_enabled", "parse_document", "sniff"]


def ocr_enabled(env=os.environ) -> bool:
    return str(env.get("OCR_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"} and ocr.available()


def _assemble(pages: list[Page]) -> str:
    parts: list[str] = []
    offset = 0
    for page in pages:
        page.start = offset
        parts.append(page.text)
        offset += len(page.text)
        page.end = offset
        offset += 1  # the joining newline
    return "\n".join(parts)


def parse_document(name: str, data: bytes, limits, low_text_threshold: int = 200, use_ocr: bool | None = None) -> ParsedDocument:
    """Parse validated upload bytes into a :class:`ParsedDocument`."""
    kind = sniff(data)
    if kind is None:
        raise DocumentError("This file is neither a PDF nor a Word (.docx) document.")
    use_ocr = ocr_enabled() if use_ocr is None else use_ocr
    if kind == "pdf":
        pages, warnings, ocr_used = extract_pdf(data, limits.max_pages, limits.max_chars, use_ocr)
    else:
        pages, warnings = extract_docx(data, limits.max_chars)
        ocr_used = False
    text = _assemble(pages)
    quality = grade(pages, low_text_threshold, ocr_used=ocr_used, ocr_available=use_ocr)
    return ParsedDocument(name=name, kind=kind, text=text, pages=pages, quality=quality, warnings=warnings)
