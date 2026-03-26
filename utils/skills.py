import re

# Master skills list — extend this as needed
skills_list = [
    "python", "java", "c++", "c#", "r", "scala", "go", "rust", "kotlin", "swift",
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "data analysis", "data science",
    "statistics", "data visualization",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "opencv", "hugging face", "transformers", "langchain", "spacy", "nltk",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "linux", "bash", "git", "github", "ci/cd", "devops", "terraform",
    "javascript", "typescript", "react", "angular", "vue", "node", "nodejs",
    "html", "css", "flask", "django", "fastapi", "rest api", "graphql",
    "communication", "leadership", "management", "teamwork", "problem solving",
    "excel", "power bi", "tableau", "looker", "jupyter",
    "hadoop", "spark", "airflow", "kafka", "etl",
    "llm", "generative ai", "prompt engineering", "fine-tuning"
]


def extract_skills(text):
    """Extract skills from text using the master skills_list."""
    text = text.lower()
    found = []
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.append(skill)
    return list(set(found))


def match_typed_skills(resume_text, typed_skills_input):
    """
    Match user-typed skills (comma-separated) against resume text.

    Returns:
        matched      — skills found in the resume
        missing      — skills NOT found in the resume
        match_percent — % of typed skills matched
    """
    resume_lower = resume_text.lower()

    typed_skills = [
        s.strip().lower()
        for s in typed_skills_input.split(",")
        if s.strip()
    ]

    matched = []
    missing = []

    for skill in typed_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, resume_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    total = len(typed_skills)
    match_percent = round((len(matched) / total) * 100, 1) if total > 0 else 0.0

    return matched, missing, match_percent


def skill_match_analysis(resume_text, job_desc):
    """Auto-compare resume skills vs job description skills using master list."""
    resume_skills = set(extract_skills(resume_text))
    job_skills    = set(extract_skills(job_desc))
    matched = resume_skills.intersection(job_skills)
    missing = job_skills - resume_skills
    return list(matched), list(missing)