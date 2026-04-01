"""
training/train_job_classifier.py
═══════════════════════════════════════════════════════════════════════════════
Trains the Job Title Classifier from data/job_dataset.csv.

What this script produces (all saved to models/)
─────────────────────────────────────────────────
  models/job_classifier.pkl       — LogisticRegression on MiniLM embeddings
                                    predicts a job title from free-text JD
  models/job_label_encoder.pkl    — LabelEncoder to decode predictions back
                                    to human-readable titles
  models/skill_index.json         — {Title: [skill1, skill2, …]} lookup
                                    powers the "Select Job Title" dropdown
                                    and skill badge display in the UI

Required input file
───────────────────
  data/job_dataset.csv   (place it at aiproject/data/job_dataset.csv)

  Mandatory columns:
    Title            — job title string  (e.g. "Data Scientist")
    Skills           — semicolon-separated skills  (e.g. "Python;SQL;ML")
    Keywords         — semicolon-separated keywords
    Responsibilities — free-text description of the role

Run (from ANY directory — paths always resolve to the project root)
────
  python training/train_job_classifier.py
  -- OR --
  cd training && python train_job_classifier.py
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Anchor all paths to the project root (aiproject/) ────────────────────────
# Works whether you run from aiproject/ OR from aiproject/training/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Config — all paths absolute, rooted at PROJECT_ROOT ─────────────────────

DATA_DIR           = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR         = os.path.join(PROJECT_ROOT, "models")
JOB_DATASET_PATH   = os.path.join(DATA_DIR, "job_dataset.csv")
JOB_CLASSIFIER_OUT = os.path.join(OUTPUT_DIR, "job_classifier.pkl")
JOB_ENCODER_OUT    = os.path.join(OUTPUT_DIR, "job_label_encoder.pkl")
SKILL_INDEX_OUT    = os.path.join(OUTPUT_DIR, "skill_index.json")
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
MIN_SAMPLES        = 3
RANDOM_SEED        = 42


def _sep(title: str) -> None:
    log.info("\n" + "=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _ensure_dirs() -> None:
    """Create data/ and models/ folders under project root if missing."""
    for folder in [DATA_DIR, OUTPUT_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            log.info("Created directory: %s", folder)
        else:
            log.info("Directory exists:  %s", folder)


def _check_dataset() -> None:
    """Verify data/job_dataset.csv exists. Print a clear error if not."""
    if os.path.exists(JOB_DATASET_PATH):
        return

    log.error("MISSING DATASET — expected: %s", JOB_DATASET_PATH)
    log.error("Place your job_dataset.csv at that path and re-run.")
    raise FileNotFoundError(f"Dataset not found: {JOB_DATASET_PATH}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common alternate column names to the expected standard names."""
    rename_map = {}
    col_lower = {c.lower().strip(): c for c in df.columns}

    for candidate in ["job title", "jobtitle", "job_title", "position", "role"]:
        if candidate in col_lower and "title" not in col_lower:
            rename_map[col_lower[candidate]] = "Title"
            break

    for candidate in ["required skills", "key skills", "skill", "skills required"]:
        if candidate in col_lower and "skills" not in col_lower:
            rename_map[col_lower[candidate]] = "Skills"
            break

    for candidate in ["key skills", "tags", "keyword", "tech stack"]:
        if candidate in col_lower and "keywords" not in col_lower:
            rename_map[col_lower[candidate]] = "Keywords"
            break

    for candidate in ["job description", "description", "duties", "responsibilities required"]:
        if candidate in col_lower and "responsibilities" not in col_lower:
            rename_map[col_lower[candidate]] = "Responsibilities"
            break

    if rename_map:
        log.info("Column renames applied: %s", rename_map)
        df = df.rename(columns=rename_map)

    return df


def parse_skills(raw: str) -> list[str]:
    """Parse a semicolon-separated skills string into a clean lowercase list."""
    skills = []
    for token in re.split(r";", raw):
        token = token.strip()
        token = re.sub(
            r"\s*(basics|fundamentals|concepts|knowledge)$", "", token, flags=re.I
        ).strip()
        if token:
            skills.append(token.lower())
    return skills


def main() -> None:

    _sep("STEP 0 — Ensuring directories exist")
    _ensure_dirs()
    log.info("Project root: %s", PROJECT_ROOT)

    _sep("STEP 1 — Checking dataset")
    _check_dataset()

    log.info("Loading: %s", JOB_DATASET_PATH)
    df = pd.read_csv(JOB_DATASET_PATH)
    log.info("Raw shape: %s | Columns: %s", df.shape, list(df.columns))

    df = _normalise_columns(df)

    required = {"Title", "Skills", "Keywords", "Responsibilities"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}\n"
            "Rename your CSV columns to match, or add the missing ones."
        )

    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    for col in ["Skills", "Keywords", "Responsibilities"]:
        df[col] = df[col].fillna("")
    log.info("Rows after dropna: %d | Unique titles: %d", len(df), df["Title"].nunique())

    _sep("STEP 2 — Build skill_index.json")
    title_skills: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        skills = parse_skills(row["Skills"]) + parse_skills(row["Keywords"])
        title_skills[row["Title"].strip()].update(skills)

    skill_index = {t: sorted(s) for t, s in title_skills.items()}
    with open(SKILL_INDEX_OUT, "w") as f:
        json.dump(skill_index, f, indent=2)
    log.info("Titles indexed: %d  ->  %s", len(skill_index), SKILL_INDEX_OUT)

    _sep("STEP 3 — Build training text")

    def build_text(row) -> str:
        s = re.sub(r";", " ", row["Skills"])
        k = re.sub(r";", " ", row["Keywords"])
        r = row["Responsibilities"]
        return f"{s} {k} {s} {k} {r}"   # skills repeated for weight

    df["combined_text"] = df.apply(build_text, axis=1)

    counts   = df["Title"].value_counts()
    df_train = df[df["Title"].isin(counts[counts >= MIN_SAMPLES].index)].copy()
    log.info(
        "Training rows: %d / %d  (titles with >=  %d samples: %d / %d)",
        len(df_train), len(df),
        MIN_SAMPLES,
        df_train["Title"].nunique(), df["Title"].nunique(),
    )

    if len(df_train) == 0:
        raise ValueError(
            f"No job titles have >= {MIN_SAMPLES} rows. "
            f"Lower MIN_SAMPLES (currently {MIN_SAMPLES}) at the top of this script."
        )

    _sep("STEP 4 — Encode labels")
    le = LabelEncoder()
    y  = le.fit_transform(df_train["Title"])
    log.info("Classes: %d", len(le.classes_))

    _sep("STEP 5 — Generate embeddings")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    X = model.encode(
        df_train["combined_text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    log.info("Embedding shape: %s", X.shape)

    _sep("STEP 6 — Train / evaluate")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    log.info("Accuracy: %.4f  (%.2f%%)", acc, acc * 100)
    log.info(
        "\n%s",
        classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0),
    )

    _sep("STEP 7 — Save models")
    joblib.dump(clf, JOB_CLASSIFIER_OUT)
    joblib.dump(le,  JOB_ENCODER_OUT)
    log.info(
        "Saved:\n  %s\n  %s\n  %s",
        JOB_CLASSIFIER_OUT, JOB_ENCODER_OUT, SKILL_INDEX_OUT,
    )
    log.info("Next step: streamlit run app.py")


if __name__ == "__main__":
    main()