from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

print("Starting Flask App...")

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd = request.form['job_description']
        resumes = request.files.getlist('resumes')

        jd_embedding = model.encode(jd)

        results = []

        for file in resumes:
            text = extract_text(file)
            res_embedding = model.encode(text)

            score = cosine_similarity([jd_embedding], [res_embedding])[0][0]

            reason = generate_reason(jd, text)

            results.append({
                "name": file.filename,
                "score": round(score, 3),
                "reason": reason
            })

        results = sorted(results, key=lambda x: x['score'], reverse=True)

        return render_template('result.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)