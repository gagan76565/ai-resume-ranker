import csv
import io
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import PyPDF2
from flask import Flask, jsonify, make_response, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "instance")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "resume_ranker.db")))
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_CONTENT_LENGTH = 25 * 1024 * 1024
MAX_RESUME_SIZE = 8 * 1024 * 1024
MAX_RESUMES = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@contextmanager
def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT NOT NULL,
                job_description TEXT NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screening_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                score REAL NOT NULL,
                matched_terms TEXT NOT NULL DEFAULT '[]',
                missing_terms TEXT NOT NULL DEFAULT '[]',
                explanation TEXT NOT NULL,
                FOREIGN KEY (screening_id) REFERENCES screenings(id) ON DELETE CASCADE
            )
        """)


def extract_pdf_text(file_storage):
    reader = PyPDF2.PdfReader(file_storage.stream)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                return ""
        except Exception:
            return ""
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def keywords(text):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{1,}", text.lower())
    return {token for token in tokens if token not in ENGLISH_STOP_WORDS and len(token) > 2}


def analyse_resume(job_description, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform([job_description, resume_text])
    score = round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 1)

    jd_terms = keywords(job_description)
    resume_terms = keywords(resume_text)
    matched = sorted(jd_terms & resume_terms)[:8]
    missing = sorted(jd_terms - resume_terms)[:8]

    if score >= 45:
        label = "Higher textual similarity"
    elif score >= 20:
        label = "Moderate textual similarity"
    else:
        label = "Lower textual similarity"

    if matched:
        detail = "Matched terms: " + ", ".join(matched) + "."
    else:
        detail = "No clear keyword overlap was found."
    if missing:
        detail += " Job-description terms not found in extracted resume text: " + ", ".join(missing) + "."

    return score, matched, missing, label + ". " + detail


def row_to_screening(row):
    return {
        "id": row["id"],
        "job_title": row["job_title"],
        "candidate_count": row["candidate_count"],
        "created_at": row["created_at"],
    }


def row_to_candidate(row):
    return {
        "id": row["id"],
        "screening_id": row["screening_id"],
        "filename": row["filename"],
        "score": row["score"],
        "matched_terms": json.loads(row["matched_terms"]),
        "missing_terms": json.loads(row["missing_terms"]),
        "explanation": row["explanation"],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "AI Resume Ranker"})


@app.get("/api/stats")
def stats():
    with get_db() as connection:
        screenings = connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
        candidates = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        average = connection.execute("SELECT AVG(score) FROM candidates").fetchone()[0]
        high_matches = connection.execute("SELECT COUNT(*) FROM candidates WHERE score >= 45").fetchone()[0]
    return jsonify({
        "screenings": screenings,
        "candidates": candidates,
        "average_score": round(average or 0, 1),
        "high_matches": high_matches,
    })


@app.get("/api/screenings")
def list_screenings():
    with get_db() as connection:
        rows = connection.execute("SELECT * FROM screenings ORDER BY id DESC LIMIT 100").fetchall()
    return jsonify([row_to_screening(row) for row in rows])


@app.get("/api/screenings/<int:screening_id>")
def screening_detail(screening_id):
    with get_db() as connection:
        screening = connection.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,)).fetchone()
        if screening is None:
            return jsonify({"error": "Screening not found."}), 404
        candidates = connection.execute(
            "SELECT * FROM candidates WHERE screening_id = ? ORDER BY score DESC, filename ASC",
            (screening_id,),
        ).fetchall()
    result = row_to_screening(screening)
    result["job_description"] = screening["job_description"]
    result["candidates"] = [row_to_candidate(row) for row in candidates]
    return jsonify(result)


@app.post("/api/screen")
def screen_resumes():
    job_title = (request.form.get("job_title") or "").strip()[:120] or "Untitled role"
    job_description = (request.form.get("job_description") or "").strip()
    files = [file for file in request.files.getlist("resumes") if file and file.filename]

    if len(job_description) < 20:
        return jsonify({"error": "Enter a job description with at least 20 characters."}), 400
    if not files:
        return jsonify({"error": "Upload at least one PDF resume."}), 400
    if len(files) > MAX_RESUMES:
        return jsonify({"error": f"Upload no more than {MAX_RESUMES} resumes per screening."}), 400

    processed = []
    errors = []
    for file in files:
        filename = secure_filename(file.filename) or "resume.pdf"
        if Path(filename).suffix.lower() != ".pdf":
            errors.append({"filename": filename, "error": "Only PDF files are supported."})
            continue
        file.stream.seek(0, io.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_RESUME_SIZE:
            errors.append({"filename": filename, "error": "File exceeds the 8 MB per-resume limit."})
            continue
        try:
            text = extract_pdf_text(file)
            if len(text) < 20:
                errors.append({"filename": filename, "error": "No usable text found. Scanned/image-only PDFs may need OCR."})
                continue
            score, matched, missing, explanation = analyse_resume(job_description, text)
            processed.append({
                "filename": filename,
                "score": score,
                "matched_terms": matched,
                "missing_terms": missing,
                "explanation": explanation,
            })
        except Exception:
            errors.append({"filename": filename, "error": "This PDF could not be read. Try exporting it again as a text-based PDF."})

    if not processed:
        return jsonify({"error": "No resumes could be analysed.", "file_errors": errors}), 422

    processed.sort(key=lambda candidate: candidate["score"], reverse=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_db() as connection:
        cursor = connection.execute(
            "INSERT INTO screenings (job_title, job_description, candidate_count, created_at) VALUES (?, ?, ?, ?)",
            (job_title, job_description, len(processed), created_at),
        )
        screening_id = cursor.lastrowid
        for candidate in processed:
            connection.execute(
                """INSERT INTO candidates
                   (screening_id, filename, score, matched_terms, missing_terms, explanation)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    screening_id, candidate["filename"], candidate["score"],
                    json.dumps(candidate["matched_terms"]), json.dumps(candidate["missing_terms"]),
                    candidate["explanation"],
                ),
            )

    return jsonify({
        "screening_id": screening_id,
        "job_title": job_title,
        "created_at": created_at,
        "candidate_count": len(processed),
        "candidates": processed,
        "file_errors": errors,
    }), 201


@app.get("/api/export.csv")
def export_csv():
    screening_id = request.args.get("screening_id", type=int)
    if not screening_id:
        return jsonify({"error": "Provide a screening_id query parameter."}), 400
    with get_db() as connection:
        screening = connection.execute("SELECT job_title FROM screenings WHERE id = ?", (screening_id,)).fetchone()
        if screening is None:
            return jsonify({"error": "Screening not found."}), 404
        rows = connection.execute(
            "SELECT filename, score, matched_terms, missing_terms, explanation FROM candidates WHERE screening_id = ? ORDER BY score DESC",
            (screening_id,),
        ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Candidate PDF", "Text similarity (%)", "Matched terms", "Terms not found", "Explanation"])
    for row in rows:
        writer.writerow([
            row["filename"], row["score"], ", ".join(json.loads(row["matched_terms"])),
            ", ".join(json.loads(row["missing_terms"])), row["explanation"],
        ])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="screening-{screening_id}.csv"'
    return response


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The total upload is too large. Keep all uploads under 25 MB."}), 413


@app.errorhandler(500)
def internal_error(_error):
    app.logger.exception("Unexpected application error")
    return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
