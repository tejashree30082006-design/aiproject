"""
core/matcher.py
═══════════════════════════════════════════════════════════════════════════════
ML-based Resume ↔ Job Description Matching Model

Replaces cosine_similarity() entirely.

Architecture:
  1. SentenceTransformer encodes both resume & JD into dense vectors
  2. A trained XGBClassifier (or LogisticRegression fallback) predicts:
       - match_score  : float 0..1  (calibrated probability)
       - match_label  : "Strong" | "Good" | "Weak"
  3. Feature engineering adds skill-overlap, section-weights, and
     keyword density on top of raw embeddings — giving the model richer
     signal than raw dot-product similarity.

Training:  see training/train_matcher.py
Inference: ResumeJobMatcher.predict(resume_text, job_text) → MatchResult
═══════════════════════════════════════════════════════════════════════════════
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

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Structured output from the ML matcher."""
    score: float                       # 0.0 – 1.0  (calibrated probability)
    score_pct: float                   # 0.0 – 100.0 (display value)
    label: str                         # "Strong" | "Good" | "Weak"
    confidence: float                  # model confidence in the label
    feature_contributions: dict = field(default_factory=dict)  # interpretability


# ─────────────────────────────────────────────────────────────────────────────
# Label thresholds
# ─────────────────────────────────────────────────────────────────────────────

LABEL_THRESHOLDS = {
    "Strong": 0.70,   # ≥70% → Strong match
    "Good":   0.45,   # ≥45% → Good match
    "Weak":   0.00,   # <45% → Weak
}

def _score_to_label(score: float) -> str:
    if score >= LABEL_THRESHOLDS["Strong"]:
        return "Strong"
    if score >= LABEL_THRESHOLDS["Good"]:
        return "Good"
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# Matcher class
# ─────────────────────────────────────────────────────────────────────────────

class ResumeJobMatcher:
    """
    ML-powered resume ↔ job description matching model.

    Usage
    -----
    matcher = ResumeJobMatcher.load("models/matcher.pkl")
    result  = matcher.predict(resume_text, job_text)
    print(result.score_pct, result.label)
    """

    MODEL_FILENAME = "models/matcher.pkl"
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

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
        """
        Run the full ML pipeline and return a MatchResult.

        Parameters
        ----------
        resume_text : raw or lightly cleaned resume string
        job_text    : raw or lightly cleaned job description string

        Returns
        -------
        MatchResult with score, label, and feature contributions.
        """
        resume_clean = clean_text(resume_text)
        job_clean    = clean_text(job_text)

        # 1. Encode to embeddings
        resume_vec = self._emb.encode([resume_clean], show_progress_bar=False)
        job_vec    = self._emb.encode([job_clean],    show_progress_bar=False)

        # 2. Engineer features (skill overlap, keyword density, etc.)
        features = self._fe.build(
            resume_text=resume_clean,
            job_text=job_clean,
            resume_vec=resume_vec,
            job_vec=job_vec,
        )

        # 3. Predict probability from ML model
        proba = self._clf.predict_proba(features)[0]

        # Model is trained with classes [0=Weak, 1=Good, 2=Strong]
        # We derive a composite score: 0*p_weak + 0.6*p_good + 1.0*p_strong
        score = float(0.0 * proba[0] + 0.60 * proba[1] + 1.0 * proba[2])
        score = round(min(max(score, 0.0), 1.0), 4)

        label      = _score_to_label(score)
        confidence = float(max(proba))

        # 4. Feature contributions for interpretability
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

    def predict_batch(
        self,
        resume_texts: list[str],
        job_text: str,
    ) -> list[MatchResult]:
        """
        Score multiple resumes against a single job description.
        Returns results sorted by score descending.
        """
        results = [self.predict(r, job_text) for r in resume_texts]
        return sorted(results, key=lambda x: x.score, reverse=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_FILENAME) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "classifier":        self._clf,
            "feature_engineer":  self._fe,
        }, path)
        logger.info("Matcher saved → %s", path)

    @classmethod
    def load(cls, path: str = MODEL_FILENAME) -> "ResumeJobMatcher":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Matcher model not found at '{path}'. "
                "Run training/train_matcher.py first."
            )
        payload = joblib.load(path)
        emb     = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls(
            classifier=payload["classifier"],
            feature_engineer=payload["feature_engineer"],
            embedding_model=emb,
        )

    @classmethod
    def build_untrained(cls) -> "ResumeJobMatcher":
        """
        Construct a matcher that uses only embedding-based scoring
        (no trained classifier required).  Used as a fallback when
        the trained model file is absent.
        """
        from core.fallback_scorer import EmbeddingOnlyScorer
        fe  = FeatureEngineer()
        emb = SentenceTransformer(cls.EMBED_MODEL_NAME)
        return cls(classifier=EmbeddingOnlyScorer(), feature_engineer=fe, embedding_model=emb)
