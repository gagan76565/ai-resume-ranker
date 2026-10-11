# TalentLens — AI Resume Ranker

A responsive resume-screening dashboard built with Flask, scikit-learn, and SQLite. Compare text extracted from PDF resumes with a job description, review ranked candidates, revisit saved screening sessions, and export results to CSV.

**Live app:** https://ai-resume-ranker-lowa.onrender.com  
**Repository:** https://github.com/gagan76565/ai-resume-ranker

## Features

- Dashboard overview with screening, resume, and similarity statistics
- Multiple PDF upload for a single job description (up to 20 resumes per screening)
- Lightweight TF-IDF vectorization and cosine-similarity scoring
- Candidate ranking, filename search, score filters, and term-overlap explanations
- Saved screening sessions backed by SQLite
- Candidate detail view and CSV export
- Responsive dashboard layout for desktop and mobile
- JSON API endpoints and a health-check endpoint
- Backend smoke tests run through GitHub Actions
- Gunicorn production entry point for Render

## Tech stack

| Layer | Tools |
| --- | --- |
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Text relevance | scikit-learn, TF-IDF, cosine similarity |
| PDF extraction | PyPDF2 |
| Local persistence | SQLite |
| Production server | Gunicorn |
| Deployment | Render |
| Continuous integration | GitHub Actions |

## How it works

1. Enter a role title and job description.
2. Upload one or more text-based PDF resumes.
3. The backend extracts available PDF text.
4. TF-IDF vectors and cosine similarity calculate a text-relevance score.
5. Candidates are sorted by score and the UI displays keyword overlap.
6. The screening session and candidate results are saved to SQLite for later review.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Dashboard |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/screenings` | Recent screening sessions |
| GET | `/api/screenings/<id>` | One session and its candidates |
| POST | `/api/screen` | Analyse uploaded PDF resumes |
| GET | `/api/export.csv?screening_id=<id>` | Export one session to CSV |

## Run locally

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/gagan76565/ai-resume-ranker.git
cd ai-resume-ranker

python -m venv venv
```

Activate the virtual environment:

**Windows PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The same test command runs in GitHub Actions when code is pushed to `main` or a `redesign/**` branch, and on pull requests targeting `main`.

## Deploy on Render

Configure a Python web service connected to this repository:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120`
- **Health check path:** `/api/health`

The included `Procfile` declares the same Gunicorn command. Render provides the `PORT` environment variable; the production WSGI server binds to it.

### Data persistence note

SQLite stores screening history in the application's data directory. A typical Render instance has an ephemeral filesystem, so data can be lost when the service is replaced or restarted. For durable production history, attach a persistent disk and set `DATA_DIR` to its mounted path, or migrate to managed PostgreSQL. Do not treat the current SQLite setup as durable storage unless persistent storage is configured.

## Important limitations

- The similarity percentage measures text overlap; it is **not** a probability of job success, a validated ATS score, or a hiring recommendation.
- Keyword matches are literal terms from extracted text and can miss synonyms, context, transferable skills, and qualifications.
- Image-only or scanned PDFs may require OCR; this app does not currently perform OCR.
- Uploaded resume contents are processed by the application. Do not upload personal data unless you have appropriate permission and understand the deployment's storage and privacy configuration.
- The dashboard has no authentication. Restrict access before using it with real candidate information.

## Project structure

```text
ai-resume-ranker/
├── .github/
│   └── workflows/
│       └── tests.yml
├── templates/
│   ├── index.html
│   └── result.html
├── tests/
│   └── test_app.py
├── app.py
├── Procfile
├── requirements.txt
└── README.md
```

## Author

**Gagan**  
GitHub: https://github.com/gagan76565
