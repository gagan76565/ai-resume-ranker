import io
import unittest
import zipfile
from unittest import mock

from talentlens.documents import DocumentError, parse_document, sniff
from talentlens.documents import ocr as ocr_module
from talentlens.minipdf import build_pdf
from talentlens.uploads import UploadLimits

LIMITS = UploadLimits()
RESUME_LINES = [
    "# Jordan Sample", "jordan.sample@example.invalid | +44 7700 900123", "Location: Leeds, UK",
    "# Summary", "Backend engineer focused on reliable Python services.",
    "# Skills", "Python, SQL, Docker, AWS",
    "# Experience",
    "Backend Developer, Example Co, Jan 2022 - Present",
    "Developed REST APIs with Python and Flask, deployed with Docker on AWS.",
    "Reduced API latency by 30% by adding Redis caching.",
    "# Projects",
    "Quiz service: Built a quiz API in FastAPI backed by PostgreSQL.",
    "# Education",
    "BSc in Computer Science, Example University, 2021",
    "# Certifications",
    "AWS Certified Cloud Practitioner",
]


def docx_bytes(paragraphs, table=None):
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def blank_pdf(pages=1):
    import pymupdf

    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    return doc.tobytes()


class SniffTests(unittest.TestCase):
    def test_identifies_formats_by_magic(self):
        self.assertEqual(sniff(build_pdf(["x"])), "pdf")
        self.assertEqual(sniff(docx_bytes(["hello"])), "docx")
        self.assertIsNone(sniff(b"plain text"))

    def test_zip_that_is_not_docx(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("readme.txt", "hi")
        self.assertIsNone(sniff(buffer.getvalue()))


class PdfTests(unittest.TestCase):
    def test_valid_pdf_is_excellent(self):
        doc = parse_document("cv.pdf", build_pdf(RESUME_LINES), LIMITS)
        self.assertEqual(doc.kind, "pdf")
        self.assertEqual(doc.quality.grade, "excellent")
        self.assertTrue(doc.quality.evaluable)
        self.assertIn("Developed REST APIs", doc.text)

    def test_multi_page_pdf_keeps_page_numbers(self):
        lines = [f"Line {i} describing Python work in detail." for i in range(120)]
        doc = parse_document("long.pdf", build_pdf(lines), LIMITS)
        self.assertGreater(len(doc.pages), 1)
        last_line = doc.text.rfind("Line 119")
        self.assertEqual(doc.page_at(last_line), doc.pages[-1].number)
        self.assertEqual(doc.page_at(0), 1)
        for page in doc.pages:  # spans index the joined text exactly
            self.assertEqual(doc.text[page.start:page.end], page.text)

    def test_empty_pdf_needs_ocr_and_is_not_evaluable(self):
        with mock.patch.object(ocr_module, "available", return_value=False):
            doc = parse_document("scan.pdf", blank_pdf(2), LIMITS, use_ocr=False)
        self.assertEqual(doc.quality.grade, "needs_ocr")
        self.assertFalse(doc.quality.evaluable)
        self.assertTrue(any("OCR" in r for r in doc.quality.reasons))

    def test_scanned_pdf_uses_ocr_when_available(self):
        recovered = "Jordan Sample\nSkills\nPython, SQL and Docker for backend services and data pipelines."
        with mock.patch.object(ocr_module, "available", return_value=True), \
                mock.patch.object(ocr_module, "ocr_pdf_pages", return_value=[recovered]):
            doc = parse_document("scan.pdf", blank_pdf(1), LIMITS, use_ocr=True)
        self.assertTrue(doc.quality.ocr_used)
        self.assertTrue(doc.quality.evaluable)
        self.assertIn("Python", doc.text)
        self.assertLess(doc.quality.score, 1.0)  # OCR lowers confidence

    def test_low_text_pdf_is_poor_but_evaluable(self):
        doc = parse_document("short.pdf", build_pdf(["Python developer with SQL and Docker skills, open to roles."]), LIMITS)
        self.assertEqual(doc.quality.grade, "poor")
        self.assertTrue(doc.quality.evaluable)

    def test_malformed_pdf_raises_friendly_error(self):
        with self.assertRaises(DocumentError) as ctx:
            parse_document("bad.pdf", b"%PDF-1.4\n garbage without objects", LIMITS)
        self.assertIn("could not be read", str(ctx.exception))

    def test_page_cap(self):
        doc = parse_document("many.pdf", build_pdf([f"line {i}" for i in range(400)]), UploadLimits(max_pages=2))
        self.assertEqual(len(doc.pages), 2)
        self.assertTrue(any("first 2" in w for w in doc.warnings))

    def test_unknown_format_rejected(self):
        with self.assertRaises(DocumentError):
            parse_document("x.pdf", b"not a document", LIMITS)


class DocxTests(unittest.TestCase):
    def test_docx_paragraphs_and_tables(self):
        data = docx_bytes(["Jordan Sample", "Skills", "Python, SQL"], table=[["Language", "Years"], ["Python", "4"]])
        doc = parse_document("cv.docx", data, LIMITS)
        self.assertEqual(doc.kind, "docx")
        self.assertIn("Python, SQL", doc.text)
        self.assertIn("Language | Years", doc.text)
        self.assertIsNone(doc.pages[0].number)  # DOCX has no pages

    def test_malformed_docx(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr("word/document.xml", "<not-valid-xml")
        with self.assertRaises(DocumentError):
            parse_document("broken.docx", buffer.getvalue(), LIMITS)

    def test_zip_bomb_guard(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr("word/document.xml", "<w/>")
            z.writestr("word/media/huge.bin", b"\0" * (41 * 1024 * 1024))
        with self.assertRaises(DocumentError) as ctx:
            parse_document("bomb.docx", buffer.getvalue(), LIMITS)
        self.assertIn("unusually large", str(ctx.exception))


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scoring.profile import extract_profile

        cls.doc = parse_document("cv.pdf", build_pdf(RESUME_LINES), LIMITS)
        cls.profile = extract_profile(cls.doc.text)

    def test_contact_fields_are_verbatim_and_not_scored(self):
        contact = self.profile["contact"]
        self.assertEqual(contact["name"]["text"], "Jordan Sample")
        self.assertEqual(contact["email"]["text"], "jordan.sample@example.invalid")
        self.assertIn("7700 900123", contact["phone"]["text"])
        self.assertEqual(contact["location"]["text"], "Leeds, UK")
        self.assertFalse(contact["used_for_scoring"])
        for key in ("name", "email", "phone", "location"):
            span = contact[key]
            self.assertEqual(self.doc.text[span["start"]:span["end"]], span["text"])

    def test_experience_entry(self):
        exp = self.profile["experience"]
        self.assertEqual(exp["status"], "found")
        entry = exp["items"][0]
        self.assertEqual(entry["title"], "Backend Developer")
        self.assertEqual(entry["company"], "Example Co")
        self.assertEqual(entry["dates"]["end"].lower(), "present")
        self.assertIn("Flask", entry["technologies"])
        self.assertTrue(entry["has_implementation_detail"])
        self.assertTrue(any("30%" in a["text"] for a in entry["achievements"]))

    def test_education_and_projects(self):
        edu = self.profile["education"]["items"][0]
        self.assertEqual(edu["degree"], "BSc")
        self.assertEqual(edu["graduation_year"], "2021")
        self.assertIn("Example University", edu["institution"])
        project = self.profile["projects"]["items"][0]
        self.assertEqual(project["title"], "Quiz service")
        self.assertTrue(project["has_implementation_detail"])
        self.assertEqual(self.profile["certifications"]["status"], "found")

    def test_missing_sections_are_reported_not_invented(self):
        from scoring.profile import extract_profile

        profile = extract_profile("Python developer\nI like building APIs.")
        self.assertEqual(profile["experience"]["status"], "section_missing")
        self.assertEqual(profile["education"]["status"], "section_missing")
        self.assertIsNone(profile["contact"]["email"])
        self.assertEqual(profile["publications"]["items"], [])


if __name__ == "__main__":
    unittest.main()
