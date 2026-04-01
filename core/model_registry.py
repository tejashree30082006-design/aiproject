"""
core/model_registry.py
===============================================================================
Centralised model loading with path fallbacks, caching, and health checks.

All Streamlit @st.cache_resource calls in app.py delegate here so
model-loading logic is never scattered across UI code.
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
from sentence_transformers import SentenceTransformer

from core.config import config
from core.matcher import ResumeJobMatcher

log = logging.getLogger(__name__)


# ── Health status ─────────────────────────────────────────────────────────────

@dataclass
class ModelHealth:
    matcher_trained:       bool = False
    job_classifier_loaded: bool = False
    resume_clf_loaded:     bool = False
    skill_index_loaded:    bool = False
    warnings:              list = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return (
            self.matcher_trained
            and self.job_classifier_loaded
            and self.resume_clf_loaded
            and self.skill_index_loaded
        )

    def sidebar_items(self):
        """Returns list of (emoji, message) for sidebar display."""
        return [
            ("✅" if self.matcher_trained       else "⚠️",
             "ML Matcher" + ("" if self.matcher_trained
                             else " — fallback mode (run train_matcher.py)")),
            ("✅" if self.job_classifier_loaded  else "❌",
             "Job Classifier" + ("" if self.job_classifier_loaded else " not found")),
            ("✅" if self.resume_clf_loaded      else "❌",
             "Resume Classifier" + ("" if self.resume_clf_loaded else " not found")),
            ("✅" if self.skill_index_loaded     else "❌",
             "Skill Index" + ("" if self.skill_index_loaded
                              else " not found (run train_job_classifier.py)")),
        ]


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Load and cache all models. One instance is held by Streamlit's
    @st.cache_resource so models are loaded once per server process.
    """

    def __init__(self):
        self.health        = ModelHealth()
        self._matcher      = None
        self._job_clf      = None
        self._job_enc      = None
        self._resume_clf   = None
        self._resume_enc   = None
        self._embed_model  = None
        self._skill_index  = {}
        self._load_all()

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def matcher(self) -> ResumeJobMatcher:
        return self._matcher

    @property
    def job_classifier(self):
        return self._job_clf

    @property
    def job_label_encoder(self):
        return self._job_enc

    @property
    def resume_classifier(self):
        return self._resume_clf

    @property
    def resume_label_encoder(self):
        return self._resume_enc

    @property
    def embed_model(self):
        return self._embed_model

    @property
    def skill_index(self) -> dict:
        return self._skill_index

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        self._load_matcher()
        self._load_job_classifier()
        self._load_resume_classifier()
        self._load_skill_index()

    def _load_matcher(self):
        path = config.paths.matcher_model
        if os.path.exists(path):
            try:
                self._matcher = ResumeJobMatcher.load(path)
                self.health.matcher_trained = True
                log.info("ML Matcher loaded from %s", path)
                return
            except Exception as e:
                log.warning("Matcher load failed: %s", e)
                self.health.warnings.append(f"Matcher load error: {e}")

        self.health.warnings.append(
            "Matcher not trained. Run: python training/train_matcher.py"
        )
        self._matcher = ResumeJobMatcher.build_untrained()

    def _load_job_classifier(self):
        pairs = [
            (config.paths.job_classifier,  config.paths.job_label_encoder),
            (config.paths.legacy_job_clf,  config.paths.legacy_job_enc),
        ]
        for clf_path, enc_path in pairs:
            if os.path.exists(clf_path) and os.path.exists(enc_path):
                try:
                    self._job_clf = joblib.load(clf_path)
                    self._job_enc = joblib.load(enc_path)
                    self.health.job_classifier_loaded = True
                    log.info("Job classifier loaded from %s", clf_path)
                    return
                except Exception as e:
                    log.warning("Job classifier load error: %s", e)
        self.health.warnings.append(
            "Job classifier missing. Run: python training/train_job_classifier.py"
        )

    def _load_resume_classifier(self):
        pairs = [
            (config.paths.resume_classifier,  config.paths.resume_label_encoder),
            (config.paths.legacy_res_clf,     config.paths.legacy_res_enc),
        ]
        for clf_path, enc_path in pairs:
            if os.path.exists(clf_path) and os.path.exists(enc_path):
                try:
                    self._resume_clf  = joblib.load(clf_path)
                    self._resume_enc  = joblib.load(enc_path)
                    self._embed_model = SentenceTransformer(config.models.embedding_model)
                    self.health.resume_clf_loaded = True
                    log.info("Resume classifier loaded from %s", clf_path)
                    return
                except Exception as e:
                    log.warning("Resume classifier load error: %s", e)
        self.health.warnings.append("Resume classifier missing.")

    def _load_skill_index(self):
        for path in [config.paths.skill_index, config.paths.legacy_skill_idx]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        self._skill_index = json.load(f)
                    self.health.skill_index_loaded = True
                    log.info("Skill index loaded: %d titles", len(self._skill_index))
                    return
                except Exception as e:
                    log.warning("Skill index load error: %s", e)
        self.health.warnings.append(
            "skill_index.json missing. Run: python training/train_job_classifier.py"
        )

    # ── Prediction helpers ────────────────────────────────────────────────────

    def predict_resume_category(self, resume_text: str) -> str:
        """Return predicted resume category label, or '—' if unavailable."""
        if self._resume_clf is None or self._embed_model is None:
            return "—"
        try:
            vec  = self._embed_model.encode([resume_text])
            pred = self._resume_clf.predict(vec)
            return self._resume_enc.inverse_transform(pred)[0]
        except Exception as e:
            log.warning("Category prediction failed: %s", e)
            return "—"

    def predict_job_title(self, job_text: str) -> Optional[str]:
        """Predict the closest job title from free-text, or None if unavailable."""
        if self._job_clf is None or self._embed_model is None:
            return None
        try:
            vec  = self._embed_model.encode([job_text])
            pred = self._job_clf.predict(vec)
            return self._job_enc.inverse_transform(pred)[0]
        except Exception as e:
            log.warning("Job title prediction failed: %s", e)
            return None
