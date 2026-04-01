"""
utils/skills.py
═══════════════════════════════════════════════════════════════════════════════
Skill extraction and analysis utilities.

All functions operate on plain strings — no ML models required here.
The ML model (ResumeJobMatcher) handles semantic scoring; these functions
provide symbolic skill-level analysis on top.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Master skills vocabulary
# ─────────────────────────────────────────────────────────────────────────────

SKILLS_VOCAB: list[str] = [
    # Languages
    "python", "java", "c++", "c#", "r", "scala", "go", "rust",
    "kotlin", "swift", "php", "ruby", "matlab", "typescript", "javascript",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle",
    "redis", "cassandra", "elasticsearch", "dynamodb", "neo4j",
    # ML / AI
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "data analysis", "data science",
    "statistics", "data visualization", "feature engineering", "model deployment",
    "transfer learning", "fine-tuning", "prompt engineering", "llm",
    "generative ai", "rag", "vector database", "embeddings",
    # ML Libraries
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "opencv", "hugging face", "transformers", "langchain", "spacy", "nltk",
    "xgboost", "lightgbm", "catboost", "mlflow", "wandb", "dvc",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "linux", "bash", "git", "github", "ci/cd", "devops", "terraform",
    "jenkins", "airflow", "ansible", "helm",
    # Web
    "react", "angular", "vue", "node", "nodejs", "html", "css",
    "flask", "django", "fastapi", "rest api", "graphql", "spring boot",
    # Data Engineering
    "hadoop", "spark", "kafka", "etl", "dbt", "bigquery", "snowflake",
    "databricks", "flink",
    # BI / Analytics
    "excel", "power bi", "tableau", "looker", "jupyter",
    # Soft Skills
    "communication", "leadership", "management", "teamwork", "problem solving",
    "agile", "scrum", "project management",
]

# Pre-compile patterns for speed
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (skill, re.compile(r"\b" + re.escape(skill) + r"\b", re.I))
    for skill in SKILLS_VOCAB
]


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """Return all skills found in text (de-duplicated, sorted)."""
    found = {skill for skill, pat in _PATTERNS if pat.search(text)}
    return sorted(found)


@dataclass
class SkillMatchResult:
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    match_pct: float = 0.0
    total_required: int = 0


def match_typed_skills(resume_text: str, required_skills_csv: str) -> SkillMatchResult:
    """
    Match comma-separated required skills against resume text.

    Parameters
    ----------
    resume_text          : cleaned resume string
    required_skills_csv  : e.g. "python, machine learning, docker"

    Returns
    -------
    SkillMatchResult with matched, missing, match_pct
    """
    required = [
        s.strip().lower()
        for s in required_skills_csv.split(",")
        if s.strip()
    ]
    if not required:
        return SkillMatchResult()

    matched, missing = [], []
    for skill in required:
        pat = re.compile(r"\b" + re.escape(skill) + r"\b", re.I)
        (matched if pat.search(resume_text) else missing).append(skill)

    pct = round(len(matched) / len(required) * 100, 1)
    return SkillMatchResult(
        matched=matched,
        missing=missing,
        match_pct=pct,
        total_required=len(required),
    )


def skill_gap_analysis(resume_text: str, job_text: str) -> tuple[list[str], list[str]]:
    """
    Auto-detect skill gap between resume and job description.

    Returns
    -------
    (matched_skills, missing_skills)
    """
    resume_skills = set(extract_skills(resume_text))
    job_skills    = set(extract_skills(job_text))
    matched       = sorted(resume_skills & job_skills)
    missing       = sorted(job_skills - resume_skills)
    return matched, missing
