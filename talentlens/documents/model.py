"""Data model for parsed documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Page:
    """Normalised text of one page and its [start, end) offsets in ``ParsedDocument.text``."""

    number: int | None  # 1-based PDF page; None for formats without pages (DOCX)
    text: str
    start: int = 0
    end: int = 0
    ocr: bool = False


@dataclass
class Quality:
    """Extraction diagnostics. ``grade`` is one of excellent | partial | poor | needs_ocr | failed."""

    grade: str
    score: float  # 0..1, used as one input to evaluation confidence
    reasons: list[str] = field(default_factory=list)
    pages: int = 0
    pages_with_text: int = 0
    chars: int = 0
    ocr_used: bool = False
    ocr_available: bool = False

    @property
    def evaluable(self) -> bool:
        """Only documents with usable text are scored; needs_ocr / failed are reported, never ranked."""
        return self.grade in {"excellent", "partial", "poor"}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["evaluable"] = self.evaluable
        return data


@dataclass
class ParsedDocument:
    name: str
    kind: str  # "pdf" | "docx"
    text: str
    pages: list[Page]
    quality: Quality
    warnings: list[str] = field(default_factory=list)

    def page_at(self, offset: int) -> int | None:
        """1-based page number containing character ``offset`` of ``text`` (None if unknown)."""
        for page in self.pages:
            if page.start <= offset < page.end:
                return page.number
        return None

    def page_spans(self) -> list[tuple[int | None, int, int]]:
        return [(p.number, p.start, p.end) for p in self.pages]


class DocumentError(Exception):
    """A document that cannot be read; the message is safe to show to users."""
