AI Resume Ranker
#####
AI-powered Resume Ranking System built using Flask, NLP, and Machine Learning techniques to automate candidate shortlisting.
#####
Live Demo:
https://ai-resume-ranker-lowa.onrender.com
#####
Features:-
>AI-powered resume ranking
>Job description matching
>Matched skills detection
>Missing skills analysis
>Explainable candidate insights
>Multiple PDF resume upload
>Modern responsive UI
>Cloud deployment
#####

Tech Stack
Backend
Flask
Python
AI / NLP
Scikit-learn
TF-IDF Vectorization
Cosine Similarity
Frontend
Tailwind CSS
HTML
PDF Processing
PyPDF2
Deployment
Render
GitHub
#####

AI Workflow
User uploads job description and resumes
PDF text extraction engine processes resumes
TF-IDF vectorization converts text into vectors
Cosine similarity compares resumes with job description
Candidates are ranked based on relevance
System highlights matched and missing skills
#####

Project Structure
ai-resume-ranker/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
└── README.md
#####

Installation & Clone Repository
git clone https://github.com/gagan76565/ai-resume-ranker.git
Create Virtual Environment:
python -m venv venv
Activate Environment

Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

Install Dependencies
pip install -r requirements.txt
Run Application
python app.py

#####
Future Improvements
Advanced resume parsing
Database integration
User authentication
ATS compatibility scoring
LLM-powered explanations
Export reports as PDF
Dashboard analytics
Resume keyword optimization
#####

Deployment
Deployed on Render cloud platform.
#####

Challenges Faced
Memory limitations during deployment
Optimization of AI pipeline for free-tier hosting
PDF text extraction inconsistencies
Solution
#####

Transformer-based embeddings were replaced with:
TF-IDF Vectorization
Cosine Similarity
This reduced memory usage significantly while maintaining efficient resume ranking.
#####

Author
I.Gagan

GitHub:
https://github.com/gagan76565
