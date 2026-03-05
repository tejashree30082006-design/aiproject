import streamlit as st
import joblib
import PyPDF2
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text


# ---------------------------------
# Page Setup
# ---------------------------------

st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("AI Resume Screening System")


# ---------------------------------
# Load AI Models
# ---------------------------------

@st.cache_resource
def load_models():

    classifier = joblib.load("resume_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return classifier, label_encoder, embedding_model


classifier, label_encoder, embedding_model = load_models()


# ---------------------------------
# File Reading Functions
# ---------------------------------

def read_pdf(file):

    pdf_reader = PyPDF2.PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def read_docx(file):

    doc = docx.Document(file)

    text = ""

    for para in doc.paragraphs:
        text += para.text

    return text


# ---------------------------------
# Upload Resume
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

job_desc = st.text_area("Enter Job Description", height=200)


# ---------------------------------
# Analyze Button
# ---------------------------------

if st.button("Analyze Resume"):

    if uploaded_file is None:

        st.warning("Please upload a resume")

    elif not job_desc.strip():

        st.warning("Please enter job description")

    else:

        # Extract text

        if uploaded_file.type == "application/pdf":

            resume_text = read_pdf(uploaded_file)

        else:

            resume_text = read_docx(uploaded_file)

        resume_text = clean_text(resume_text)
        job_desc = clean_text(job_desc)

        # -----------------------------
        # 1️⃣ Category Prediction
        # -----------------------------

        resume_vector = embedding_model.encode([resume_text])

        predicted = classifier.predict(resume_vector)

        category = label_encoder.inverse_transform(predicted)[0]

        # -----------------------------
        # 2️⃣ Similarity Matching
        # -----------------------------

        job_vector = embedding_model.encode([job_desc])

        score = cosine_similarity(resume_vector, job_vector)[0][0]

        score = round(score * 100, 2)

        # -----------------------------
        # Output
        # -----------------------------

        st.subheader("Results")

        st.write("Predicted Resume Category:", category)

        st.write("Match Score:", f"{score}%")

        if score > 70:

            st.success("Strong Match")

        elif score > 40:

            st.warning("Moderate Match")

        else:

            st.error("Weak Match")

        st.subheader("Extracted Resume Text")

        st.text_area("", resume_text, height=300)