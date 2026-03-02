import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocessing import clean_text
from utils.matching import compute_similarity

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("AI Resume Screening System")

# ---------------------------------
# Load Dataset
# ---------------------------------
try:
    df = pd.read_csv("data/resumes.csv", low_memory=False)
except FileNotFoundError:
    st.error("resumes.csv not found inside data folder.")
    st.stop()

# Validate column
if 'resume' not in df.columns:
    st.error("CSV must contain a column named 'resume'")
    st.stop()

# Clean resumes
df['resume'] = df['resume'].fillna("").astype(str)
df['resume'] = df['resume'].apply(clean_text)

st.write(f"Total resumes loaded: {len(df)}")

# ---------------------------------
# Job Description Input
# ---------------------------------
job_desc = st.text_area("Enter Job Description", height=150)

# ---------------------------------
# Analyze Button
# ---------------------------------
if st.button("Analyze Resumes"):

    if not job_desc.strip():
        st.warning("Please enter a job description.")
        st.stop()

    with st.spinner("Analyzing resumes..."):

        # Clean job description
        job_desc_clean = clean_text(job_desc)

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        resume_vectors = vectorizer.fit_transform(df['resume'])
        job_vector = vectorizer.transform([job_desc_clean])

        # Compute similarity
        df['Score'] = compute_similarity(resume_vectors, job_vector)

        # Sort by score
        ranked = df.sort_values(by='Score', ascending=False)

        # Take top 10
        top_results = ranked.head(10).reset_index()

        st.success("Analysis Complete!")
        st.subheader("Top Matching Candidates")

        # ---------------------------------
        # Clickable Table
        # ---------------------------------
        if 'category' in df.columns:
            display_df = top_results[['index', 'category', 'Score']]
        else:
            display_df = top_results[['index', 'Score']]

        selected = st.dataframe(
            display_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        # ---------------------------------
        # Show Selected Resume
        # ---------------------------------
        if selected.selection.rows:
            selected_row = selected.selection.rows[0]
            actual_index = top_results.loc[selected_row, 'index']

            st.subheader("Selected Resume Details")

            st.text_area(
                "Resume Content",
                df.loc[actual_index, 'resume'],
                height=300
            )

            st.write("Match Score:", round(df.loc[actual_index, 'Score'], 4))