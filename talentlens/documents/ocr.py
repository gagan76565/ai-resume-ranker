"""Optional OCR fallback with Tesseract (local only).

Used only when ``OCR_ENABLED`` is not ``0`` *and* the ``pytesseract`` package and the
system ``tesseract`` binary are both present. Render's free tier cannot install the
binary, so there OCR is simply unavailable and scanned PDFs are flagged ``needs_ocr``.
"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def ocr_pdf_pages(doc, max_pages: int = 5, dpi: int = 200, timeout: int = 20) -> list[str]:
    """OCR the first ``max_pages`` pages of an open PyMuPDF document. Returns one string per page."""
    import io

    import pytesseract
    from PIL import Image

    texts: list[str] = []
    for index in range(min(len(doc), max_pages)):
        pix = doc[index].get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            texts.append(pytesseract.image_to_string(image, timeout=timeout) or "")
        except RuntimeError:  # pytesseract raises RuntimeError on timeout
            log.warning("OCR timed out on a page; skipping it")
            texts.append("")
    return texts
