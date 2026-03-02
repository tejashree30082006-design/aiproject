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
@st.cache_data
def load_data():
    df = pd.read_csv("data/resumes.csv", low_memory=False)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("resumes.csv not found inside data folder.")
    st.stop()

if "resume" not in df.columns:
    st.error("CSV must contain a column named 'resume'")
    st.stop()

# Clean resumes
df["resume"] = df["resume"].fillna("").astype(str)
df["resume"] = df["resume"].apply(clean_text)

st.write(f"Total resumes loaded: {len(df)}")


# ---------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------
if "ranked_results" not in st.session_state:
    st.session_state.ranked_results = None

if "selected_row" not in st.session_state:
    st.session_state.selected_row = None


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
    else:
        with st.spinner("Analyzing resumes..."):

            job_desc_clean = clean_text(job_desc)

            vectorizer = TfidfVectorizer(stop_words="english")

            resume_vectors = vectorizer.fit_transform(df["resume"])
            job_vector = vectorizer.transform([job_desc_clean])

            df["Score"] = compute_similarity(resume_vectors, job_vector)

            ranked = df.sort_values(by="Score", ascending=False)

            # Store top 10 in session
            st.session_state.ranked_results = ranked.head(10).reset_index()
            st.session_state.selected_row = None

        st.success("Analysis Complete!")


# ---------------------------------
# DISPLAY RESULTS (Persistent)
# ---------------------------------
if st.session_state.ranked_results is not None:

    st.subheader("Top Matching Candidates")

    results = st.session_state.ranked_results

    # Columns to display
    if "category" in results.columns:
        display_df = results[["index", "category", "Score"]]
    else:
        display_df = results[["index", "Score"]]

    selected = st.dataframe(
        display_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Save selected full row (not just index)
    if selected.selection.rows:
        row_id = selected.selection.rows[0]
        st.session_state.selected_row = results.loc[row_id]


# ---------------------------------
# SHOW SELECTED RESUME
# ---------------------------------
if st.session_state.selected_row is not None:

    row = st.session_state.selected_row
    resume_index = row["index"]

    st.subheader("Selected Resume Details")

    st.text_area(
        "Resume Content",
        df.loc[resume_index, "resume"],
        height=300
    )

    st.write("Match Score:", round(row["Score"], 4))