"""Document-quality grading.

Grades (documented in docs/ai/SCORING_METHODOLOGY.md):

* ``excellent`` — plenty of clean text on every page;
* ``partial``   — usable text but some pages empty or a noticeable share of garbage glyphs;
* ``poor``      — little text (under ``LOW_TEXT_THRESHOLD`` characters) or heavy garbage; scored, low confidence;
* ``needs_ocr`` — essentially no extractable text (image-only / scanned) and OCR not available or not successful;
* ``failed``    — the file could not be parsed at all (raised as an error, not graded here).
"""

from __future__ import annotations

import re

from .model import Page, Quality

MIN_EVALUABLE_CHARS = 50  # below this there is nothing meaningful to score
_GARBAGE = re.compile(r"\(cid:\d+\)|�|[-]")
_WORD = re.compile(r"[A-Za-z]{2,}")


def grade(pages: list[Page], low_text_threshold: int = 200, ocr_used: bool = False, ocr_available: bool = False) -> Quality:
    text = "\n".join(p.text for p in pages)
    chars = len(text.strip())
    with_text = sum(1 for p in pages if len(p.text.strip()) >= 20)
    garbage = len(_GARBAGE.findall(text))
    words = _WORD.findall(text)
    letters = sum(len(w) for w in words)
    alpha_ratio = letters / chars if chars else 0.0
    reasons: list[str] = []

    if chars < MIN_EVALUABLE_CHARS:
        reasons.append(f"Only {chars} characters of text could be extracted; the file is probably scanned or image-only.")
        if not ocr_available:
            reasons.append("OCR is not available on this server, so the resume cannot be read.")
        return Quality("needs_ocr", 0.0, reasons, len(pages), with_text, chars, ocr_used, ocr_available)

    score = 1.0
    if chars < low_text_threshold:
        score -= 0.6
        reasons.append(f"Very little text ({chars} characters).")
    empty_pages = len(pages) - with_text
    if pages and empty_pages:
        score -= 0.3 * empty_pages / len(pages)
        reasons.append(f"{empty_pages} of {len(pages)} page(s) contain little or no extractable text.")
    garbage_ratio = garbage / max(1, len(words))
    if garbage_ratio > 0.02:
        score -= min(0.4, garbage_ratio * 4)
        reasons.append("Some characters could not be decoded (unusual fonts or encoding).")
    if alpha_ratio < 0.5:
        score -= 0.2
        reasons.append("A large share of the text is not words (tables, symbols or layout artefacts).")
    if ocr_used:
        score -= 0.2
        reasons.append("Text was recovered with OCR, which can misread characters.")
    score = max(0.0, min(1.0, score))
    label = "excellent" if score >= 0.85 else "partial" if score >= 0.5 else "poor"
    if not reasons:
        reasons.append("Clean, complete text extraction.")
    return Quality(label, round(score, 2), reasons, len(pages), with_text, chars, ocr_used, ocr_available)
