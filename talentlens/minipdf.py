"""A tiny, dependency-free writer for text-only PDFs.

Used to generate the bundled synthetic sample resumes and PDFs in tests. It
supports Helvetica text in the WinAnsi (Latin-1) range, simple word wrapping
and automatic page breaks. It is not a general PDF library.
"""

from __future__ import annotations

import textwrap

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN = 56
LEADING = 14
WRAP = 92


def _escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _layout(lines: list[str]) -> list[list[tuple[str, bool]]]:
    """Wrap lines and split them into pages. Lines starting with '# ' are bold headings."""
    per_page = (PAGE_HEIGHT - 2 * MARGIN) // LEADING
    flat: list[tuple[str, bool]] = []
    for line in lines:
        bold = line.startswith("# ")
        content = line[2:] if bold else line
        wrapped = textwrap.wrap(content, WRAP) or [""]
        flat.extend((part, bold) for part in wrapped)
    pages = [flat[i:i + per_page] for i in range(0, len(flat), per_page)]
    return pages or [[]]


def build_pdf(lines: list[str], title: str = "Document") -> bytes:
    """Return the bytes of a PDF containing ``lines`` of text."""
    pages = _layout(lines)
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_id = add(b"")  # placeholder, filled below
    pages_id = add(b"")
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    bold_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    page_ids = []
    for page in pages:
        ops = ["BT", f"{LEADING} TL", f"{MARGIN} {PAGE_HEIGHT - MARGIN} Td"]
        current = None
        for text, bold in page:
            font = "/F2 11 Tf" if bold else "/F1 10 Tf"
            if font != current:
                ops.append(font)
                current = font
            ops.append(f"({_escape(text)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_id = add(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R /F2 {bold_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        page_ids.append(page_id)
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    info_id = add(f"<< /Title ({_escape(title)}) /Producer (TalentLens minipdf) >>".encode("latin-1"))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
