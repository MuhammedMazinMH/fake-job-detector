# 🛡️ Fake Job Detector

ML-powered tool to detect fake job postings and protect job seekers from scams.

## Problem

Every year, thousands of job seekers in India fall victim to fake job scams:

- "Data Entry" work-from-home jobs with unrealistic pay
- Upfront registration/training fees
- Vague requirements and no company details

## Solution

A machine learning model that analyzes job posting text and classifies it as:

- ✅ **Real** — Legitimate job posting
- ⚡ **Uncertain** — Some suspicious elements, verify before applying
- ⚠️ **Fake** — Likely a scam

## Live Demo

🔗 **Try it now:** [Hugging Face Space](https://huggingface.co/spaces/MuhammedMazin/fake-job-detector)

Paste any job posting to instantly check if it's real or fake.

**Test Examples:**
- **Fake:** "Earn $500 daily. No experience needed. Send $50 registration fee."
- **Real:** "Bachelor's degree required. 3+ years Python experience. Microservices architecture."

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | **0.810** |
| Recall | 0.82 |
| Precision | 0.80 |

- Dataset: 17,880 job postings (4.84% fraudulent)
- Algorithm: XGBoost with SMOTE for class balancing
- Features: 2000 TF-IDF (unigrams + bigrams) + 8 numeric features

## Tech Stack

| Component | Tool |
|-----------|------|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| ML | XGBoost, Scikit-learn |
| Text Processing | TF-IDF with bigrams |
| Local UI | Streamlit |
| Public Deploy | Gradio + Hugging Face Spaces |
| Class Balancing | SMOTE |

## Quick Start

### Option 1: Use Online (No Setup)
Visit the live demo: [Hugging Face Space](https://huggingface.co/spaces/MuhammedMazin/fake-job-detector)

### Option 2: Run Locally
```bash
git clone https://github.com/MuhammedMazinMH/fake-job-detector.git
cd fake-job-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run local Streamlit app
cd src/app
streamlit run app.py

# View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db

Project Structure

fake-job-detector/
├── data/
│   ├── fake_job_postings.csv          # DVC tracked
│   └── processed/
│       └── fake_jobs_clean.csv
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering_v2.ipynb # TF-IDF + feature extraction
│   ├── 03_model_training.ipynb        # Baseline models
│   └── 04_model_training_v2.ipynb     # XGBoost + SMOTE
├── src/
│   ├── app/
│   │   ├── app.py                     # Streamlit UI (local)
│   │   └── gradio_app.py              # Gradio UI (Hugging Face)
│   ├── features/
│   │   ├── tfidf_vectorizer_v2.pkl
│   │   └── X_y_v2.pkl
│   └── models/
│       ├── best_model_v2.pkl          # XGBoost (F1=0.810)
│       ├── rf_model_v2.pkl            # Random Forest (Hugging Face)
│       └── vectorizer_v2.pkl
├── .gitignore
├── requirements.txt
└── README.md

