from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import os

print("Starting Flask App...")

app = Flask(__name__)

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# -------------------------------
# Generate explanation
# -------------------------------
def generate_reason(jd, resume_text):
    jd_words = set(jd.lower().split())
    resume_words = set(resume_text.lower().split())

    matched = jd_words.intersection(resume_words)

    common_skills = list(matched)[:5]
    missing_skills = list(jd_words - resume_words)[:5]

    if len(matched) > 20:
        return f"Strong match. Skills: {', '.join(common_skills)}. Missing: {', '.join(missing_skills)}"
    elif len(matched) > 10:
        return f"Moderate match. Skills: {', '.join(common_skills)}. Missing: {', '.join(missing_skills)}"
    else:
        return f"Low match. Missing important skills like: {', '.join(missing_skills)}"

# -------------------------------
# Main route
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd = request.form['job_description']
        resumes = request.files.getlist('resumes')

        results = []

        for file in resumes:
            text = extract_text(file)

            # Handle empty text case
            if not text.strip():
                score = 0
                reason = "Could not extract text from resume."
            else:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([jd, text])

                score = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:2]
                )[0][0]

                reason = generate_reason(jd, text)

            results.append({
                "name": file.filename,
                "score": round(score, 3),
                "reason": reason
            })

        results = sorted(results, key=lambda x: x['score'], reverse=True)

        return render_template('result.html', results=results)

    return render_template('index.html')

# -------------------------------
# Run app (Render compatible)
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
