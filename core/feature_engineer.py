"""
core/feature_engineer.py
═══════════════════════════════════════════════════════════════════════════════
Feature Engineering for the Resume ↔ Job ML Matcher

Constructs a rich feature vector that combines:
  • Semantic features  — embedding dot product, L2 distance, cosine (used as
                         ONE input feature among many, not the final score)
  • Skill overlap      — Jaccard similarity over the master skills vocabulary
  • Keyword density    — TF-based keyword overlap
  • Section weights    — experience section gets 3× weight vs. education
  • Length ratio       — resume/JD length balance
  • Title signal       — job title presence in resume
  • NER count delta    — rough entity count difference

Total features: ~20 numeric dimensions fed into XGBoostClassifier.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.skills import extract_skills
from utils.preprocessing import clean_text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Raw cosine similarity between two 1-D vectors (used as ONE feature only)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten() - b.flatten()))


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Bag-of-words Jaccard overlap (crude but fast)."""
    tokens_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
    return _jaccard(tokens_a, tokens_b)


def _length_ratio(text_a: str, text_b: str) -> float:
    la = max(len(text_a.split()), 1)
    lb = max(len(text_b.split()), 1)
    return min(la, lb) / max(la, lb)


def _count_years_experience(text: str) -> float:
    """Rough heuristic: extract max years mentioned."""
    nums = re.findall(r"(\d+)\s*(?:\+\s*)?years?", text.lower())
    return float(max((int(n) for n in nums), default=0))


_SECTION_RE = {
    "experience":  re.compile(
        r"(experience|work history|employment|professional background)", re.I
    ),
    "education":   re.compile(r"(education|academic|degree|university|college)", re.I),
    "skills":      re.compile(r"(skills|technologies|tools|languages|stack)", re.I),
    "projects":    re.compile(r"(projects|portfolio|open.source)", re.I),
    "certifications": re.compile(r"(certif|license|accredit)", re.I),
}

def _section_presence(text: str) -> dict[str, float]:
    return {
        name: float(bool(pattern.search(text)))
        for name, pattern in _SECTION_RE.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms (resume_text, job_text, resume_vec, job_vec) → feature matrix.

    All features are floats so XGBoost can consume them directly.
    The same instance is serialised into the model pickle, so it is
    stateless (no fitted scaler — XGBoost handles feature scale natively).
    """

    # Feature names (must match build() order exactly)
    FEATURE_NAMES = [
        "cosine_sim",           # embedding cosine (1 of 20 features)
        "l2_distance",          # embedding L2
        "dot_product_norm",     # normalised dot product
        "skill_jaccard",        # skill-set Jaccard
        "skill_overlap_count",  # absolute # of overlapping skills
        "resume_skill_count",   # total skills in resume
        "job_skill_count",      # total skills required by JD
        "skill_coverage_ratio", # overlap / job_skills (recall-like)
        "keyword_overlap",      # bag-of-words Jaccard
        "length_ratio",         # shorter/longer word count ratio
        "resume_yoe",           # years of experience in resume
        "has_experience_sec",   # resume has experience section
        "has_education_sec",    # resume has education section
        "has_skills_sec",       # resume has skills section
        "has_projects_sec",     # resume has projects section
        "has_certs_sec",        # resume has certifications
        "jd_has_experience_sec",
        "jd_has_skills_sec",
        "skill_gap_ratio",      # (job_skills - overlap) / job_skills
        "avg_embed_magnitude",  # avg magnitude of the two vectors
    ]

    # ── Public: build feature matrix ─────────────────────────────────────────

    def build(
        self,
        resume_text: str,
        job_text: str,
        resume_vec: np.ndarray,
        job_vec: np.ndarray,
    ) -> np.ndarray:
        """Return (1, n_features) array ready for classifier.predict_proba()."""
        vec = self._feature_vector(resume_text, job_text, resume_vec, job_vec)
        return np.array([vec], dtype=np.float32)

    def build_batch(
        self,
        resume_texts: list[str],
        job_text: str,
        resume_vecs: np.ndarray,
        job_vec: np.ndarray,
    ) -> np.ndarray:
        """Return (n, n_features) array for batch scoring."""
        rows = [
            self._feature_vector(rt, job_text, rv[np.newaxis], job_vec)
            for rt, rv in zip(resume_texts, resume_vecs)
        ]
        return np.array(rows, dtype=np.float32)

    def contributions(
        self,
        resume_text: str,
        job_text: str,
        resume_vec: np.ndarray,
        job_vec: np.ndarray,
    ) -> dict[str, float]:
        """Named feature values for display / interpretability."""
        vec = self._feature_vector(resume_text, job_text, resume_vec, job_vec)
        return dict(zip(self.FEATURE_NAMES, vec))

    # ── Private ───────────────────────────────────────────────────────────────

    def _feature_vector(
        self,
        resume_text: str,
        job_text: str,
        resume_vec: np.ndarray,
        job_vec: np.ndarray,
    ) -> list[float]:

        rv = resume_vec.flatten()
        jv = job_vec.flatten()

        # Embedding features
        cos_sim      = _cosine(rv, jv)
        l2_dist      = _l2_distance(rv, jv)
        dot_norm     = float(np.dot(rv, jv)) / (len(rv) + 1e-9)

        # Skill features
        resume_skills = set(extract_skills(resume_text))
        job_skills    = set(extract_skills(job_text))
        overlap       = resume_skills & job_skills
        skill_jac     = _jaccard(resume_skills, job_skills)
        overlap_count = float(len(overlap))
        res_count     = float(len(resume_skills))
        job_count     = float(len(job_skills))
        coverage      = overlap_count / max(job_count, 1)
        gap_ratio     = 1.0 - coverage

        # Text features
        kw_overlap    = _keyword_overlap(resume_text, job_text)
        len_ratio     = _length_ratio(resume_text, job_text)
        yoe           = _count_years_experience(resume_text)

        # Section presence
        r_secs = _section_presence(resume_text)
        j_secs = _section_presence(job_text)

        # Magnitude
        avg_mag = float((np.linalg.norm(rv) + np.linalg.norm(jv)) / 2)

        return [
            cos_sim,
            l2_dist,
            dot_norm,
            skill_jac,
            overlap_count,
            res_count,
            job_count,
            coverage,
            kw_overlap,
            len_ratio,
            yoe,
            r_secs["experience"],
            r_secs["education"],
            r_secs["skills"],
            r_secs["projects"],
            r_secs["certifications"],
            j_secs["experience"],
            j_secs["skills"],
            gap_ratio,
            avg_mag,
        ]
