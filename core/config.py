"""
core/config.py
═══════════════════════════════════════════════════════════════════════════════
Centralised configuration for the AI Resume Screening system.

All magic numbers, paths, and tuneable constants live here.
Import from any module:  from core.config import Config
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Paths:
    # Training data
    resume_csv:          str = "data/resume_dataset.csv"
    job_csv:             str = "data/job_dataset.csv"

    # Trained model artifacts
    matcher_model:       str = "models/matcher.pkl"
    job_classifier:      str = "models/job_classifier.pkl"
    job_label_encoder:   str = "models/job_label_encoder.pkl"
    resume_classifier:   str = "models/resume_classifier.pkl"
    resume_label_encoder:str = "models/label_encoder.pkl"
    skill_index:         str = "models/skill_index.json"

    # Legacy fallbacks (original layout, root dir)
    legacy_job_clf:      str = "job_classifier.pkl"
    legacy_job_enc:      str = "job_label_encoder.pkl"
    legacy_res_clf:      str = "resume_classifier.pkl"
    legacy_res_enc:      str = "label_encoder.pkl"
    legacy_skill_idx:    str = "skill_index.json"


@dataclass(frozen=True)
class _Models:
    embedding_model:     str = "all-MiniLM-L6-v2"
    gemini_model:        str = "gemini-2.5-flash"


@dataclass(frozen=True)
class _Matcher:
    # Label thresholds (score 0–1)
    strong_threshold:    float = 0.70
    good_threshold:      float = 0.45

    # Composite score weights (must sum to 1.0)
    weight_strong_class: float = 1.00   # p_strong weight
    weight_good_class:   float = 0.60   # p_good weight
    weight_weak_class:   float = 0.00   # p_weak weight


@dataclass(frozen=True)
class _Training:
    n_pairs:             int   = 5000
    test_size:           float = 0.20
    random_seed:         int   = 42
    min_job_samples:     int   = 3
    batch_size:          int   = 64

    # Pair generation ratios
    strong_ratio:        float = 0.40
    partial_ratio:       float = 0.20
    weak_ratio:          float = 0.40

    # XGBoost
    xgb_n_estimators:    int   = 300
    xgb_max_depth:       int   = 6
    xgb_learning_rate:   float = 0.05
    xgb_subsample:       float = 0.80
    xgb_colsample:       float = 0.80


@dataclass(frozen=True)
class _UI:
    default_threshold:   int  = 50    # minimum ML score % slider default
    max_resume_words:    int  = 800   # truncation for LLM prompts
    max_jd_chars:        int  = 1000  # truncation for LLM prompts
    cards_per_row:       int  = 2     # result cards per row


@dataclass(frozen=True)
class Config:
    paths:    _Paths    = field(default_factory=_Paths)
    models:   _Models   = field(default_factory=_Models)
    matcher:  _Matcher  = field(default_factory=_Matcher)
    training: _Training = field(default_factory=_Training)
    ui:       _UI       = field(default_factory=_UI)


# Singleton — import this everywhere
config = Config()
