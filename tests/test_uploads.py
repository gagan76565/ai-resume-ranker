import io
import unittest

from werkzeug.datastructures import FileStorage

from talentlens.minipdf import build_pdf
from talentlens.uploads import PdfReadError, UploadLimits, display_name, extract_text, validate_uploads

TEXT_LINES = ["# Skills", "Python, SQL, Docker and AWS"] + [f"Line {i} of a long, text-based resume." for i in range(20)]


def storage(name, data):
    return FileStorage(stream=io.BytesIO(data), filename=name, content_type="application/pdf")


def pdf(lines=TEXT_LINES):
    return build_pdf(lines, title="test")


class ValidationTests(unittest.TestCase):
    limits = UploadLimits(max_file_bytes=50_000, max_files=3, max_total_bytes=120_000)

    def test_accepts_valid_pdf(self):
        result = validate_uploads([storage("cv.pdf", pdf())], self.limits)
        self.assertIsNone(result.batch_error)
        self.assertEqual([f.name for f in result.accepted], ["cv.pdf"])

    def test_rejects_wrong_extension_with_name(self):
        result = validate_uploads([storage("notes.docx", pdf())], self.limits)
        self.assertEqual(result.accepted, [])
        self.assertIn("notes.docx", result.errors[0]["error"])

    def test_rejects_missing_magic_bytes(self):
        result = validate_uploads([storage("fake.pdf", b"<html>not a pdf</html>")], self.limits)
        self.assertIn("%PDF", result.errors[0]["error"])
        self.assertIn("fake.pdf", result.errors[0]["error"])

    def test_rejects_oversized_file(self):
        big = pdf() + b"0" * 60_000
        result = validate_uploads([storage("big.pdf", big)], self.limits)
        self.assertIn("big.pdf", result.errors[0]["error"])
        self.assertIn("per-file limit", result.errors[0]["error"])

    def test_rejects_too_many_files(self):
        files = [storage(f"{i}.pdf", pdf()) for i in range(4)]
        result = validate_uploads(files, self.limits)
        self.assertIn("limit is 3", result.batch_error)

    def test_total_size_cap_names_the_file(self):
        chunk = pdf() + b"%" * (45_000 - len(pdf()))
        files = [storage(f"f{i}.pdf", chunk) for i in range(3)]
        result = validate_uploads(files, self.limits)
        self.assertEqual(len(result.accepted), 2)
        self.assertIn("f2.pdf", result.errors[0]["error"])
        self.assertIn("total", result.errors[0]["error"])

    def test_empty_selection(self):
        self.assertIsNotNone(validate_uploads([], self.limits).batch_error)

    def test_empty_file(self):
        result = validate_uploads([storage("empty.pdf", b"")], self.limits)
        self.assertIn("empty", result.errors[0]["error"])

    def test_duplicate_names_are_disambiguated(self):
        result = validate_uploads([storage("cv.pdf", pdf()), storage("cv.pdf", pdf())], self.limits)
        self.assertEqual([f.name for f in result.accepted], ["cv.pdf", "cv-2.pdf"])


class FilenameTests(unittest.TestCase):
    def test_path_traversal_is_stripped(self):
        self.assertEqual(display_name("../../etc/passwd.pdf", 1), "passwd.pdf")
        self.assertEqual(display_name("C:\\Users\\x\\cv.pdf", 1), "cv.pdf")

    def test_html_in_name_is_neutralised(self):
        name = display_name('<img src=x onerror="alert(1)">.pdf', 1)
        self.assertNotIn("<", name)
        self.assertNotIn('"', name)

    def test_blank_name_gets_placeholder(self):
        self.assertEqual(display_name("...", 4), "resume-4.pdf")


class ExtractionTests(unittest.TestCase):
    def test_extracts_text(self):
        extraction = extract_text(pdf(), UploadLimits())
        self.assertIn("Python", extraction.text)
        self.assertEqual(extraction.pages, 1)

    def test_page_cap(self):
        many = build_pdf([f"line {i}" for i in range(400)])
        extraction = extract_text(many, UploadLimits(max_pages=2))
        self.assertTrue(any("first 2" in w for w in extraction.warnings))

    def test_char_cap(self):
        extraction = extract_text(pdf(), UploadLimits(max_chars=50))
        self.assertLessEqual(len(extraction.text), 50)
        self.assertTrue(any("truncated" in w for w in extraction.warnings))

    def test_corrupt_pdf_raises_friendly_error(self):
        with self.assertRaises(PdfReadError) as ctx:
            extract_text(b"%PDF-1.4\n garbage without objects", UploadLimits())
        self.assertIn("could not be read", str(ctx.exception))

    def test_low_text_pdf_extracts_little(self):
        extraction = extract_text(build_pdf(["Scan"]), UploadLimits())
        self.assertLess(len(extraction.text), 200)


if __name__ == "__main__":
    unittest.main()
