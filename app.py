import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("AI Resume Screening System")

# Load dataset with FULL PATH
df = pd.read_csv(r"C:\Users\TEJASHREE\Desktop\resume_project\resumes.csv", low_memory=False)

# FIX: Handle NaN values and ensure all resumes are strings
df['resume'] = df['resume'].fillna('')  # Replace NaN with empty string
df['resume'] = df['resume'].astype(str)  # Convert everything to string

# Show dataset preview
st.write("Dataset Preview")
st.write(df.head())

# Show basic info (optional - can remove if you don't want to see it)
st.write(f"Total resumes: {len(df)}")
st.write(f"Missing values after cleaning: {df['resume'].isna().sum()}")

# Job description input
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze Resumes"):
    
    # Check if job description is empty
    if not job_desc.strip():
        st.warning("Please enter a job description")
    else:
        with st.spinner("Analyzing resumes..."):
            try:
                # Convert text to numbers
                vectorizer = TfidfVectorizer(stop_words='english')
                
                # Fit and transform resumes
                resume_vectors = vectorizer.fit_transform(df['resume'])
                
                # Transform job description
                job_vector = vectorizer.transform([job_desc])
                
                # Calculate similarity
                similarity = cosine_similarity(resume_vectors, job_vector)
                
                # Add scores to dataframe
                df['Score'] = similarity
                
                # Rank resumes
                ranked = df.sort_values(by='Score', ascending=False)
                
                st.success("Analysis Complete!")
                st.write("Top Matching Candidates")
                
                # Show results - check if 'category' column exists
                if 'category' in df.columns:
                    st.write(ranked[['category', 'Score']].head(10))
                else:
                    # If no category column, just show scores
                    st.write(ranked[['Score']].head(10))
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")