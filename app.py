import streamlit as st
import joblib
import pdfplumber
import docx
import numpy as np
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    layout="wide"
)

st.title("AI Resume Screening System")


# -----------------------------
# LOAD MODELS
# -----------------------------

@st.cache_resource
def load_models():

    classifier = joblib.load("resume_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return classifier, label_encoder, embedding_model


classifier, label_encoder, embedding_model = load_models()


# -----------------------------
# GEMINI CONFIG
# -----------------------------

GEMINI_API_KEY = "GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# -----------------------------
# FILE READING FUNCTIONS
# -----------------------------

def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


def extract_docx(file):

    doc = docx.Document(file)

    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return "\n".join(text)


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

job_description = st.text_area(
    "Enter Job Description",
    height=200
)


# -----------------------------
# ANALYSIS BUTTON
# -----------------------------

if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume.")

    elif not job_description.strip():
        st.warning("Please enter job description.")

    else:

        with st.spinner("Analyzing resume..."):

            # -----------------------------
            # Extract resume text
            # -----------------------------

            if uploaded_file.type == "application/pdf":
                resume_text = extract_pdf(uploaded_file)

            else:
                resume_text = extract_docx(uploaded_file)

            resume_text = clean_text(resume_text)
            job_description_clean = clean_text(job_description)

            # -----------------------------
            # Create embeddings
            # -----------------------------

            resume_vec = embedding_model.encode([resume_text])
            job_vec = embedding_model.encode([job_description_clean])

            # -----------------------------
            # Similarity Score
            # -----------------------------

            similarity = cosine_similarity(resume_vec, job_vec)[0][0]

            # -----------------------------
            # Predict Resume Category
            # -----------------------------

            prediction = classifier.predict(resume_vec)
            category = label_encoder.inverse_transform(prediction)[0]

            # -----------------------------
            # Display Results
            # -----------------------------

            st.subheader("AI Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Resume Category",
                    value=category
                )

            with col2:
                st.metric(
                    label="Job Match Score",
                    value=f"{round(similarity*100,2)} %"
                )

            # -----------------------------
            # Gemini Explanation
            # -----------------------------

            prompt = f"""

You are an AI hiring assistant.

Job Description:
{job_description}

Resume:
{resume_text[:2000]}

Explain:

1. Why this resume matches or doesn't match the job.
2. Strengths of the candidate.
3. Missing skills.
4. Final hiring recommendation.

Keep explanation professional and structured.
"""

            response = gemini_model.generate_content(prompt)

            st.subheader("AI Hiring Explanation")

            st.write(response.text)
