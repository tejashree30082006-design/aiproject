"""
core/matcher.py
===============================================================================
ML-based Resume <-> Job Description Matching Model

Fixes applied
─────────────
1. proba index bug   — score formula is now dynamic (works with 2-class or
                       3-class trained model)
2. zero-skill guard  — if no skills overlap AND job requires skills, score is
                       hard-capped at Weak regardless of semantic similarity
3. score formula     — uses p_fit (p_good + p_strong combined) for 2-class model
===============================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from core.feature_engineer import FeatureEngineer
from utils.preprocessing import clean_text

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    score: float                       # 0.0 – 1.0  (calibrated probability)
    score_pct: float                   # 0.0 – 100.0 (display value)
    label: str                         # "Strong" | "Good" | "Weak"
    confidence: float                  # model confidence in the label
    feature_contributions: dict = field(default_factory=dict)


# ── Label thresholds ──────────────────────────────────────────────────────────

LABEL_THRESHOLDS = {
    "Strong": 0.70,
    "Good":   0.45,
    "Weak":   0.00,
}

def _score_to_label(score: float) -> str:
    if score >= LABEL_THRESHOLDS["Strong"]:
        return "Strong"
    if score >= LABEL_THRESHOLDS["Good"]:
        return "Good"
    return "Weak"


# ── Score formula — handles both 2-class and 3-class models ──────────────────

def _proba_to_score(proba: np.ndarray) -> float:
    """
    Convert classifier probability array to a single 0-1 fit score.

    2-class model [p_no_fit, p_good_fit]:
        score = p_good_fit  (direct fit probability)

    3-class model [p_weak, p_good, p_strong]:
        score = 0.0*p_weak + 0.60*p_good + 1.0*p_strong
        (weighted composite — strong contributes more than good)
    """
    if len(proba) == 2:
        return float(proba[1])                                      # p_fit
    else:
        return float(0.0 * proba[0] + 0.60 * proba[1] + 1.0 * proba[2])


# ── Matcher ───────────────────────────────────────────────────────────────────

class ResumeJobMatcher:

    MODEL_FILENAME   = "models/matcher.pkl"
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

    # Minimum skill coverage required before semantic score is trusted.
    # If the JD has extractable skills and coverage is below this, the result
    # is hard-capped to "Weak" regardless of cosine/keyword similarity.
    MIN_SKILL_COVERAGE_FOR_QUALIFIED = 0.10   # at least 10% of JD skills matched

    def __init__(
        self,
        classifier,
        feature_engineer: FeatureEngineer,
        embedding_model: SentenceTransformer,
    ):
        self._clf = classifier
        self._fe  = feature_engineer
        self._emb = embedding_model

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, resume_text: str, job_text: str) -> MatchResult:
        resume_clean = clean_text(resume_text)
        job_clean    = clean_text(job_text)

        # 1. Encode
        resume_vec = self._emb.encode([resume_clean], show_progress_bar=False)
        job_vec    = self._emb.encode([job_clean],    show_progress_bar=False)

        # 2. Engineer features
        features = self._fe.build(
            resume_text=resume_clean,
            job_text=job_clean,
            resume_vec=resume_vec,
            job_vec=job_vec,
        )

        # 3. Predict — dynamic formula works for 2-class or 3-class model
        proba = self._clf.predict_proba(features)[0]
        score = _proba_to_score(proba)
        score = round(float(np.clip(score, 0.0, 1.0)), 4)

        # 4. Zero-skill guard ──────────────────────────────────────────────────
        # If the JD contains recognisable skills but NONE appear in the resume,
        # the candidate cannot be Qualified regardless of semantic score.
        # features shape: (1, 20) — index 7 = skill_coverage_ratio
        #                           index 6 = job_skill_count
        skill_coverage = float(features[0][7])   # overlap / job_skills
        job_skill_count = float(features[0][6])  # how many skills JD requires

        zero_skill_match = (
            job_skill_count > 0                                    # JD has skills
            and skill_coverage < self.MIN_SKILL_COVERAGE_FOR_QUALIFIED
        )
        if zero_skill_match:
            # Hard-cap score so it cannot reach "Good" or "Strong"
            max_weak_score = LABEL_THRESHOLDS["Good"] - 0.01      # 0.44
            score = min(score, max_weak_score)
            logger.debug(
                "Zero-skill guard triggered: coverage=%.3f, score capped to %.3f",
                skill_coverage, score,
            )

        # 5. Label + confidence
        label      = _score_to_label(score)
        confidence = float(max(proba))

        # 6. Feature contributions for Deep Analysis tab
        contributions = self._fe.contributions(
            resume_text=resume_clean,
            job_text=job_clean,
            resume_vec=resume_vec,
            job_vec=job_vec,
        )

        return MatchResult(
            score=score,
            score_pct=round(score * 100, 2),
            label=label,
            confidence=round(confidence, 4),
            feature_contributions=contributions,
        )

    def predict_batch(self, resume_texts: list[str], job_text: str) -> list[MatchResult]:
        results = [self.predict(r, job_text) for r in resume_texts]
        return sorted(results, key=lambda x: x.score, reverse=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_FILENAME) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "classifier":       self._clf,
            "feature_engineer": self._fe,
        }, path)
        logger.info("Matcher saved -> %s", path)

    @classmethod
    def load(cls, path: str = MODEL_FILENAME) -> "ResumeJobMatcher":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Matcher model not found at '{path}'. "
                "Run training/train_from_hf_dataset.py first."
            )
        payload = joblib.load(path)
        emb = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls(
            classifier=payload["classifier"],
            feature_engineer=payload["feature_engineer"],
            embedding_model=emb,
        )

    @classmethod
    def build_untrained(cls) -> "ResumeJobMatcher":
        from core.fallback_scorer import EmbeddingOnlyScorer
        fe  = FeatureEngineer()
        emb = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls(classifier=EmbeddingOnlyScorer(), feature_engineer=fe, embedding_model=emb)