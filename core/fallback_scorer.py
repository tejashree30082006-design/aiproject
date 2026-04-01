"""
core/fallback_scorer.py
═══════════════════════════════════════════════════════════════════════════════
EmbeddingOnlyScorer — used when matcher.pkl is not yet trained.

Implements the same sklearn predict_proba() interface as XGBoostClassifier
so the rest of the pipeline is unaffected.  Probabilities are derived from
a weighted combination of feature values (skill coverage + cosine), giving
a reasonable zero-shot baseline without any training data.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import numpy as np


class EmbeddingOnlyScorer:
    """
    Sklearn-compatible pseudo-classifier that computes match probabilities
    from raw feature values without any training.

    Feature layout mirrors FeatureEngineer.FEATURE_NAMES:
      [0] cosine_sim
      [7] skill_coverage_ratio
      [3] skill_jaccard
      [8] keyword_overlap
    """

    # Weights for the composite score (must sum to 1.0)
    _W_COSINE   = 0.35
    _W_COVERAGE = 0.35
    _W_JACCARD  = 0.15
    _W_KEYWORD  = 0.15

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n, 20) feature matrix
        Returns (n, 3) probability matrix [p_weak, p_good, p_strong]
        """
        results = []
        for row in X:
            cosine   = float(np.clip(row[0],  0.0, 1.0))
            coverage = float(np.clip(row[7],  0.0, 1.0))
            jaccard  = float(np.clip(row[3],  0.0, 1.0))
            keyword  = float(np.clip(row[8],  0.0, 1.0))

            score = (
                self._W_COSINE   * cosine +
                self._W_COVERAGE * coverage +
                self._W_JACCARD  * jaccard +
                self._W_KEYWORD  * keyword
            )
            score = float(np.clip(score, 0.0, 1.0))

            # Map scalar score → 3-class probability
            if score >= 0.70:
                p = [0.05, 0.15, 0.80]
            elif score >= 0.45:
                p = [0.10, 0.75, 0.15]
            else:
                p = [0.75, 0.20, 0.05]

            results.append(p)
        return np.array(results, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
