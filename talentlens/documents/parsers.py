"""PDF (PyMuPDF) and DOCX (python-docx) text extraction, page by page.

Uploaded bytes are parsed in memory only: never written to disk, executed or rendered
back to the browser. PyMuPDF is AGPL-3.0 licensed (see LICENSE / README).
"""

from __future__ import annotations

import io
import logging
import zipfile

from scoring.text import normalize_text

from . import ocr
from .model import DocumentError, Page

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
DOCX_MAX_UNCOMPRESSED = 40 * 1024 * 1024  # zip-bomb guard
DOCX_MAX_ENTRIES = 2000


def sniff(data: bytes) -> str | None:
    """Identify the format from magic bytes (never from the client's content type)."""
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(ZIP_MAGIC):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return None
        if "word/document.xml" in names and "[Content_Types].xml" in names:
            return "docx"
    return None


def extract_pdf(data: bytes, max_pages: int, max_chars: int, ocr_enabled: bool) -> tuple[list[Page], list[str], bool]:
    """Return (pages, warnings, ocr_used)."""
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several types
        raise DocumentError("This PDF could not be read. Try exporting it again as a text-based PDF.") from exc
    warnings: list[str] = []
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise DocumentError("This PDF is password-protected. Remove the password and upload it again.")
        total = len(doc)
        if total == 0:
            raise DocumentError("This PDF has no pages.")
        if total > max_pages:
            warnings.append(f"Only the first {max_pages} of {total} pages were read.")
        pages: list[Page] = []
        length = 0
        for index in range(min(total, max_pages)):
            try:
                raw = doc[index].get_text("text", sort=True) or ""
            except Exception:  # noqa: BLE001 - one bad page should not sink the file
                warnings.append(f"Page {index + 1} could not be read and was skipped.")
                raw = ""
            text = normalize_text(raw)
            pages.append(Page(index + 1, text))
            length += len(text)
            if length >= max_chars:
                warnings.append(f"Text was truncated to the first {max_chars:,} characters.")
                break
        ocr_used = False
        if sum(len(p.text) for p in pages) < 50 and ocr_enabled and ocr.available():
            ocr_texts = ocr.ocr_pdf_pages(doc)
            if sum(len(t.strip()) for t in ocr_texts) >= 50:
                pages = [Page(i + 1, normalize_text(t), ocr=True) for i, t in enumerate(ocr_texts)]
                ocr_used = True
                warnings.append("This PDF has no text layer; its text was recovered with OCR.")
        return pages, warnings, ocr_used
    finally:
        doc.close()


def extract_docx(data: bytes, max_chars: int) -> tuple[list[Page], list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > DOCX_MAX_ENTRIES or sum(i.file_size for i in infos) > DOCX_MAX_UNCOMPRESSED:
                raise DocumentError("This DOCX file is unusually large when unpacked and was rejected.")
        import docx

        document = docx.Document(io.BytesIO(data))
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentError("This DOCX file could not be read. Save it again from Word or export it as PDF.") from exc

    lines: list[str] = []
    body = document.element.body
    # Walk paragraphs and tables in document order.
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append("".join(node.text or "" for node in child.iter() if node.tag.endswith("}t")))
        elif tag == "tbl":
            for row in child.iter():
                if row.tag.endswith("}tr"):
                    cells = []
                    for cell in row.iter():
                        if cell.tag.endswith("}tc"):
                            cells.append("".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t")))
                    lines.append(" | ".join(c for c in cells if c.strip()))
    text = normalize_text("\n".join(lines))
    warnings: list[str] = []
    if len(text) > max_chars:
        text = text[:max_chars]
        warnings.append(f"Text was truncated to the first {max_chars:,} characters.")
    return [Page(None, text)], warnings
