import csv
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import PyPDF2
from flask import Flask, jsonify, make_response, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB per request

INSTANCE_DIR = Path(os.environ.get("INSTANCE_DIR", app.instance_path))
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", INSTANCE_DIR / "resume_ranker.db"))
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    "html", "css", "react", "angular", "vue", "node.js", "flask", "django", "fastapi",
    "rest api", "graphql", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "generative ai", "llm", "rag", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "data analysis", "data visualization", "power bi", "tableau",
    "aws", "azure", "google cloud", "docker", "kubernetes", "linux", "git", "github",
    "ci/cd", "pytest", "unit testing", "agile", "communication", "leadership",
    "project management", "excel", "statistics", "problem solving", "tailwind css"
]


def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS screenings (
                id TEXT PRIMARY KEY,
                job_title TEXT NOT NULL,
                job_description TEXT NOT NULL,
                candidate_count INTEGER NOT NULL,
                top_score REAL NOT NULL,
                results_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


init_db()


def extract_pdf_text(file_storage):
    reader = PyPDF2.PdfReader(file_storage.stream)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                return ""
        except Exception:
            return ""
    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return "\n".join(parts).strip()


def contains_term(text, term):
    escaped = re.escape(term)
    pattern = r"(?<![a-zA-Z0-9+#.])" + escaped + r"(?![a-zA-Z0-9+#.])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def find_skills(text):
    return sorted([skill for skill in SKILLS if contains_term(text, skill)])


def analyze_candidate(job_description, resume_text, filename):
    jd = job_description.strip()
    resume = resume_text.strip()
    if not resume:
        return {
            "id": uuid4().hex[:10],
            "name": filename,
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": find_skills(jd),
            "reason": "No readable text was extracted. This may be a scanned PDF; try a text-based PDF.",
            "status": "Could not parse",
            "word_count": 0,
        }

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform([jd, resume])
        score = round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 1)
    except ValueError:
        score = 0.0

    jd_skills = find_skills(jd)
    resume_skills = set(find_skills(resume))
    matched = [skill for skill in jd_skills if skill in resume_skills]
    missing = [skill for skill in jd_skills if skill not in resume_skills]
    coverage = round((len(matched) / len(jd_skills)) * 100, 1) if jd_skills else None

    if score >= 55:
        status = "High text match"
    elif score >= 30:
        status = "Moderate text match"
    else:
        status = "Low text match"

    if jd_skills:
        reason = str(len(matched)) + " of " + str(len(jd_skills)) + " detected job-description skills also appear in this resume."
    else:
        reason = "No skills from the built-in skill dictionary were detected in the job description; score is based on TF-IDF text similarity."
    return {
        "id": uuid4().hex[:10],
        "name": filename,
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage": coverage,
        "reason": reason,
        "status": status,
        "word_count": len(resume.split()),
    }


def serialize_screening(row):
    results = json.loads(row["results_json"])
    return {
        "id": row["id"],
        "job_title": row["job_title"],
        "candidate_count": row["candidate_count"],
        "top_score": row["top_score"],
        "created_at": row["created_at"],
        "results": results,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "AI Resume Ranker"})


@app.post("/api/screen")
def screen_resumes():
    job_description = (request.form.get("job_description") or "").strip()
    files = request.files.getlist("resumes")
    if len(job_description) < 20:
        return jsonify({"error": "Add a job description with at least 20 characters."}), 400
    files = [file for file in files if file and file.filename]
    if not files:
        return jsonify({"error": "Upload at least one PDF resume."}), 400
    if len(files) > 30:
        return jsonify({"error": "Upload no more than 30 resumes per screening."}), 400

    results = []
    for file in files:
        original_name = secure_filename(file.filename) or "resume.pdf"
        if Path(original_name).suffix.lower() != ".pdf":
            results.append({
                "id": uuid4().hex[:10], "name": original_name, "score": 0.0,
                "matched_skills": [], "missing_skills": find_skills(job_description),
                "reason": "Unsupported file type. Upload a PDF resume.",
                "status": "Unsupported file", "word_count": 0,
            })
            continue
        try:
            text = extract_pdf_text(file)
            results.append(analyze_candidate(job_description, text, original_name))
        except Exception:
            results.append({
                "id": uuid4().hex[:10], "name": original_name, "score": 0.0,
                "matched_skills": [], "missing_skills": find_skills(job_description),
                "reason": "The file could not be processed. Check that it is a valid, readable PDF.",
                "status": "Processing error", "word_count": 0,
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    job_title = next((line.strip() for line in job_description.splitlines() if line.strip()), "Untitled role")
    job_title = job_title[:90]
    screening_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    top_score = max((item["score"] for item in results), default=0.0)

    with get_db() as connection:
        connection.execute(
            """INSERT INTO screenings
               (id, job_title, job_description, candidate_count, top_score, results_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (screening_id, job_title, job_description, len(results), top_score, json.dumps(results), created_at),
        )

    return jsonify({
        "id": screening_id,
        "job_title": job_title,
        "candidate_count": len(results),
        "top_score": top_score,
        "created_at": created_at,
        "results": results,
        "notice": "Scores measure text similarity and keyword overlap. They should support—not replace—human review.",
    })


@app.get("/api/history")
def history():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT id, job_title, candidate_count, top_score, created_at FROM screenings ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/history/<screening_id>")
def screening_detail(screening_id):
    with get_db() as connection:
        row = connection.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Screening not found."}), 404
    return jsonify(serialize_screening(row))


@app.get("/api/export/<screening_id>")
def export_screening(screening_id):
    with get_db() as connection:
        row = connection.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Screening not found."}), 404

    results = json.loads(row["results_json"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Resume", "Text Similarity (%)", "Status", "Matched Skills", "Missing Skills", "Skill Coverage (%)", "Notes"])
    for rank, item in enumerate(results, start=1):
        writer.writerow([
            rank, item.get("name", ""), item.get("score", 0), item.get("status", ""),
            ", ".join(item.get("matched_skills", [])), ", ".join(item.get("missing_skills", [])),
            item.get("skill_coverage", ""), item.get("reason", ""),
        ])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = 'attachment; filename="resume-screening-' + screening_id[:8] + '.csv"'
    return response


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The upload is too large. Keep the total upload below 25 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
