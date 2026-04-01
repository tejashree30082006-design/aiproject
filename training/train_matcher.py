"""
training/train_matcher.py
═══════════════════════════════════════════════════════════════════════════════
Train the Resume ↔ Job ML Matcher

Workflow
────────
  1. Load resume_dataset.csv  (resume_text, category) +
         job_dataset.csv      (Title, Skills, Keywords, Responsibilities)
  2. Synthetic pair generation:
       Positive pairs  (same category): label = 2 (Strong)
       Partial pairs   (related):        label = 1 (Good)
       Negative pairs  (different):      label = 0 (Weak)
  3. Feature engineering via FeatureEngineer
  4. Train XGBClassifier (with LR fallback if xgboost absent)
  5. Calibrate with CalibratedClassifierCV
  6. Evaluate on held-out test set
  7. Save to models/matcher.pkl

Run
───
  python training/train_matcher.py

Requirements
────────────
  sentence-transformers, scikit-learn, xgboost (optional), pandas, joblib
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.feature_engineer import FeatureEngineer
from core.matcher import ResumeJobMatcher
from utils.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RESUME_CSV      = "data/resume_dataset.csv"
JOB_CSV         = "data/job_dataset.csv"
OUTPUT_PATH     = "models/matcher.pkl"
EMBED_MODEL     = "all-MiniLM-L6-v2"
N_PAIRS         = 5000      # synthetic training pairs to generate
RANDOM_SEED     = 42
TEST_SIZE       = 0.2
RELATED_PAIRS   = 0.20      # fraction of "partial match" pairs
NEG_PAIRS       = 0.40      # fraction of "no match" pairs
POS_PAIRS       = 0.40      # fraction of "strong match" pairs

# Category → related categories (same domain, partial match)
RELATED_MAP: dict[str, list[str]] = {
    "Data Science":         ["Machine Learning", "Statistics", "Business Analyst"],
    "Machine Learning":     ["Data Science", "Deep Learning", "AI Research"],
    "Software Development": ["Web Development", "DevOps", "Backend Development"],
    "Web Development":      ["Software Development", "Frontend Development", "UI/UX"],
    "DevOps":               ["Cloud Computing", "Software Development", "SRE"],
    "Cybersecurity":        ["Network Engineering", "IT Support"],
    "Business Analyst":     ["Data Science", "Project Management"],
    "Project Management":   ["Business Analyst", "Operations"],
}


def _sep(title: str) -> None:
    log.info("\n" + "═" * 60)
    log.info("  %s", title)
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "resume" in lower or "text" in lower or "content" in lower:
            col_map[col] = "resume_text"
        elif "categ" in lower or "label" in lower or "class" in lower or "title" in lower:
            col_map[col] = "category"
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["resume_text", "category"])
    df["resume_text"] = df["resume_text"].astype(str).apply(clean_text)
    log.info("Resumes loaded: %d rows, %d categories",
             len(df), df["category"].nunique())
    return df


def load_jobs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Title"])
    df["Skills"]           = df["Skills"].fillna("")
    df["Keywords"]         = df["Keywords"].fillna("")
    df["Responsibilities"] = df["Responsibilities"].fillna("")
    df["job_text"] = (
        df["Skills"].str.replace(";", " ") + " " +
        df["Keywords"].str.replace(";", " ") + " " +
        df["Responsibilities"]
    )
    df["job_text"] = df["job_text"].apply(clean_text)
    log.info("Jobs loaded: %d rows, %d titles", len(df), df["Title"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Synthetic pair generation
# ─────────────────────────────────────────────────────────────────────────────

def _category_to_job_title(category: str) -> Optional[str]:
    """Fuzzy map resume category → job dataset title."""
    return category  # direct match; extend with a lookup table if needed


def generate_pairs(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    n_pairs: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build synthetic (resume_text, job_text, label) pairs.

    Label mapping
    ─────────────
    2 → Strong  (same category / title)
    1 → Good    (related category)
    0 → Weak    (different category)
    """
    rng = random.Random(seed)
    pairs: list[dict] = []

    resume_by_cat: dict[str, list[str]] = {}
    for _, row in resumes.iterrows():
        resume_by_cat.setdefault(row["category"], []).append(row["resume_text"])

    job_by_title: dict[str, list[str]] = {}
    for _, row in jobs.iterrows():
        job_by_title.setdefault(row["Title"], []).append(row["job_text"])

    all_categories = list(resume_by_cat.keys())
    all_job_titles = list(job_by_title.keys())

    n_strong  = int(n_pairs * POS_PAIRS)
    n_partial = int(n_pairs * RELATED_PAIRS)
    n_neg     = n_pairs - n_strong - n_partial

    # ── Strong (same category → matching job title) ───────────────────────────
    for _ in range(n_strong):
        cat = rng.choice(all_categories)
        resume = rng.choice(resume_by_cat[cat])
        # Find jobs that match the category name
        matched_titles = [t for t in all_job_titles if cat.lower() in t.lower()]
        if not matched_titles:
            matched_titles = all_job_titles
        title = rng.choice(matched_titles)
        job   = rng.choice(job_by_title[title])
        pairs.append({"resume_text": resume, "job_text": job, "label": 2})

    # ── Partial (related categories) ─────────────────────────────────────────
    for _ in range(n_partial):
        cat = rng.choice(all_categories)
        resume = rng.choice(resume_by_cat[cat])
        related = RELATED_MAP.get(cat, [])
        if related:
            rel_cat = rng.choice(related)
            related_titles = [t for t in all_job_titles if rel_cat.lower() in t.lower()]
            if not related_titles:
                related_titles = all_job_titles
            title = rng.choice(related_titles)
        else:
            title = rng.choice(all_job_titles)
        job = rng.choice(job_by_title[title])
        pairs.append({"resume_text": resume, "job_text": job, "label": 1})

    # ── Weak (different, unrelated) ───────────────────────────────────────────
    for _ in range(n_neg):
        cat    = rng.choice(all_categories)
        resume = rng.choice(resume_by_cat[cat])
        # Pick a job title from a clearly different domain
        diff_titles = [
            t for t in all_job_titles
            if cat.lower() not in t.lower()
            and not any(r.lower() in t.lower() for r in RELATED_MAP.get(cat, []))
        ]
        title = rng.choice(diff_titles or all_job_titles)
        job   = rng.choice(job_by_title[title])
        pairs.append({"resume_text": resume, "job_text": job, "label": 0})

    df = pd.DataFrame(pairs).sample(frac=1, random_state=seed).reset_index(drop=True)
    label_counts = df["label"].value_counts().sort_index()
    log.info(
        "Pairs generated: %d  →  Weak=%d | Good=%d | Strong=%d",
        len(df), label_counts.get(0, 0), label_counts.get(1, 0), label_counts.get(2, 0)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Encode + engineer features
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    pairs: pd.DataFrame,
    embed_model: SentenceTransformer,
    fe: FeatureEngineer,
) -> tuple[np.ndarray, np.ndarray]:
    log.info("Encoding %d resume texts…", len(pairs))
    resume_vecs = embed_model.encode(
        pairs["resume_text"].tolist(), show_progress_bar=True, batch_size=64
    )
    log.info("Encoding %d job texts…", len(pairs))
    job_vecs = embed_model.encode(
        pairs["job_text"].tolist(), show_progress_bar=True, batch_size=64
    )

    log.info("Engineering features…")
    X = fe.build_batch(
        resume_texts=pairs["resume_text"].tolist(),
        job_text="",          # ignored in batch mode
        resume_vecs=resume_vecs,
        job_vec=job_vecs[0],  # placeholder
    )

    # Recompute properly per-pair in batch
    rows = []
    for i, (rv, jv, rt, jt) in enumerate(zip(
        resume_vecs, job_vecs,
        pairs["resume_text"], pairs["job_text"]
    )):
        row = fe._feature_vector(rt, jt, rv[np.newaxis], jv[np.newaxis])
        rows.append(row)

    X = np.array(rows, dtype=np.float32)
    y = pairs["label"].values
    log.info("Feature matrix shape: %s", X.shape)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train classifier
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier():
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
        )
        log.info("Using XGBClassifier")
    except ImportError:
        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        log.info("xgboost not found → falling back to LogisticRegression")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _sep("STEP 1 — Loading data")
    if not os.path.exists(RESUME_CSV):
        raise FileNotFoundError(f"Resume CSV not found: {RESUME_CSV}")
    if not os.path.exists(JOB_CSV):
        raise FileNotFoundError(f"Job CSV not found: {JOB_CSV}")

    resumes = load_resumes(RESUME_CSV)
    jobs    = load_jobs(JOB_CSV)

    _sep("STEP 2 — Generating synthetic training pairs")
    pairs = generate_pairs(resumes, jobs, n_pairs=N_PAIRS)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        np.arange(len(pairs)), pairs["label"].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=pairs["label"]
    )
    train_pairs = pairs.iloc[X_train_all].reset_index(drop=True)
    test_pairs  = pairs.iloc[X_test].reset_index(drop=True)

    _sep("STEP 3 — Loading embedding model")
    embed_model = SentenceTransformer(EMBED_MODEL)
    fe = FeatureEngineer()

    _sep("STEP 4 — Building feature matrices")
    X_train, y_train = build_features(train_pairs, embed_model, fe)
    X_test_f, y_test_f = build_features(test_pairs, embed_model, fe)

    _sep("STEP 5 — Training classifier")
    base_clf = build_classifier()
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    calibrated_clf.fit(X_train, y_train)

    _sep("STEP 6 — Evaluation")
    y_pred = calibrated_clf.predict(X_test_f)
    acc = accuracy_score(y_test_f, y_pred)
    log.info("Accuracy: %.4f  (%.2f%%)", acc, acc * 100)
    log.info("\n%s", classification_report(
        y_test_f, y_pred,
        target_names=["Weak", "Good", "Strong"],
        zero_division=0,
    ))

    _sep("STEP 7 — Saving model")
    os.makedirs("models", exist_ok=True)
    matcher = ResumeJobMatcher(
        classifier=calibrated_clf,
        feature_engineer=fe,
        embedding_model=embed_model,
    )
    matcher.save(OUTPUT_PATH)
    log.info("✅  Matcher saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    from typing import Optional
    main()
