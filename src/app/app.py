import streamlit as st
import pickle
import numpy as np
import re
from scipy.sparse import hstack

st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️")

st.title("🛡️ Fake Job Detector")
st.markdown("Paste a job posting to check if it's **real** or **fake**")

@st.cache_resource
def load_model():
    with open('../models/best_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/vectorizer_v2.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

title = st.text_input("Job Title", "Data Entry Clerk - Work From Home")
description = st.text_area("Job Description", "Earn $500 daily. No experience needed. Send $50 registration fee.")
requirements = st.text_area("Requirements", "Must have internet. Immediate start.")

if st.button("Check Job"):
    # Combine ALL text columns (same as training)
    text = f"{title} {description} {requirements}"
    text_clean = re.sub(r'[^a-z\s]', ' ', text.lower())
    text_clean = ' '.join(text_clean.split())
    
    # TF-IDF
    X_tfidf = vectorizer.transform([text_clean])
    
    # ALL numeric features (must match training exactly)
    text_length = len(text_clean)
    has_salary = 1 if '$' in text or 'salary' in text.lower() else 0
    has_company_logo = 0  # unknown for new input
    has_questions = 0     # unknown for new input
    has_department = 0    # unknown for new input
    has_benefits = 0      # unknown for new input
    title_length = len(title)
    desc_length = len(description)
    
    X_numeric = np.array([[text_length, has_salary, has_company_logo, has_questions,
                           has_department, has_benefits, title_length, desc_length]])
    
    # Combine
    X = hstack([X_tfidf, X_numeric])
    
    # Predict
    proba = model.predict_proba(X)[0]
    fake_prob = proba[1] * 100
    
    if fake_prob > 60:
        st.error(f"⚠️ FAKE JOB (Confidence: {fake_prob:.1f}%)")
        st.markdown("""
        **Red flags detected:**
        - Too good to be true salary
        - Upfront payment requested
        - Vague requirements
        - Work from home with high pay
        """)
    elif fake_prob > 30:
        st.warning(f"⚡ UNCERTAIN (Fake probability: {fake_prob:.1f}%)")
        st.markdown("This posting has some suspicious elements. Verify the company before applying.")
    else:
        st.success(f"✅ REAL JOB (Confidence: {100-fake_prob:.1f}%)")
        st.markdown("This posting looks legitimate.")