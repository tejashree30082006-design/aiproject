"""
Trainmodel.py
═══════════════════════════════════════════════════════════════════════════════
Trains TWO models from job_dataset.csv:

  MODEL 1 — Job Title Classifier
      Input  : combined text (Skills + Keywords + Responsibilities)
      Output : predicted Job Title  →  saved as  job_classifier.pkl
      Also saves job_label_encoder.pkl

  MODEL 2 — Skill Vocabulary & Lookup Index
      Builds a clean mapping of  Job Title → [required skills]
      saved as  skill_index.json
      This lets the app auto-populate skills when HR picks a job title.

  Additionally re-saves the existing resume classifier if resume_dataset.csv
  is present (optional — skip if you only have job_dataset.csv).

Steps:
  1. Load & clean job_dataset.csv
  2. Build skill_index.json  (title → skills lookup)
  3. Build combined training text per row
  4. Generate sentence embeddings  (all-MiniLM-L6-v2)
  5. Train Logistic Regression job title classifier
  6. Evaluate & save models
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import re
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
JOB_DATASET_PATH      = "data/job_dataset.csv"       # put the CSV here
JOB_CLASSIFIER_OUT    = "job_classifier.pkl"
JOB_LABEL_ENCODER_OUT = "job_label_encoder.pkl"
SKILL_INDEX_OUT       = "skill_index.json"           # title → skills map
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"

# Minimum samples a title needs to be included in classifier training
# Titles with fewer samples are still added to the skill index
MIN_SAMPLES_FOR_TRAINING = 3


def separator(title):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & clean dataset
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 1 — Loading job_dataset.csv")

df = pd.read_csv(JOB_DATASET_PATH)

required_cols = {"Title", "Skills", "Keywords", "Responsibilities"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Drop the one null title row
df = df.dropna(subset=["Title"]).reset_index(drop=True)
df["Skills"]           = df["Skills"].fillna("")
df["Keywords"]         = df["Keywords"].fillna("")
df["Responsibilities"] = df["Responsibilities"].fillna("")
df["ExperienceLevel"]  = df["ExperienceLevel"].fillna("Unknown")

print(f"  Rows loaded        : {len(df)}")
print(f"  Unique job titles  : {df['Title'].nunique()}")
print(f"  Experience levels  : {sorted(df['ExperienceLevel'].unique())}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build skill_index.json  (title → skills)
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 2 — Building skill index (title → skills)")

def parse_skills(raw: str) -> list:
    """Split a semicolon-separated skills string into a clean list."""
    skills = []
    for token in raw.split(";"):
        token = token.strip()
        # Remove trailing qualifier words like 'basics', 'fundamentals'
        token = re.sub(r"\s*(basics|fundamentals|concepts|knowledge)$", "", token, flags=re.I).strip()
        if token:
            skills.append(token.lower())
    return skills


# Aggregate skills per title (union across all experience levels)
title_skills_map = defaultdict(set)

for _, row in df.iterrows():
    title  = row["Title"].strip()
    skills = parse_skills(row["Skills"]) + parse_skills(row["Keywords"])
    title_skills_map[title].update(skills)

# Convert sets to sorted lists
skill_index = {title: sorted(skills) for title, skills in title_skills_map.items()}

with open(SKILL_INDEX_OUT, "w") as f:
    json.dump(skill_index, f, indent=2)

print(f"  Titles indexed     : {len(skill_index)}")
print(f"  Saved → {SKILL_INDEX_OUT}")

# Preview a few entries
print("\n  Sample entries:")
for title in list(skill_index.keys())[:3]:
    print(f"    [{title}] → {skill_index[title][:5]} …")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build combined training text
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 3 — Building training text")

def build_text(row) -> str:
    """
    Combine Skills, Keywords, and Responsibilities into one text block.
    Skills and Keywords are repeated to give them more weight over
    the generic Responsibilities prose.
    """
    skills_text  = row["Skills"].replace(";", " ")
    kw_text      = row["Keywords"].replace(";", " ")
    resp_text    = row["Responsibilities"]
    # Repeat skills + keywords twice so the model focuses on them
    return f"{skills_text} {kw_text} {skills_text} {kw_text} {resp_text}"

df["combined_text"] = df.apply(build_text, axis=1)

# Filter out titles with too few samples for reliable classification
title_counts = df["Title"].value_counts()
valid_titles  = title_counts[title_counts >= MIN_SAMPLES_FOR_TRAINING].index
df_train      = df[df["Title"].isin(valid_titles)].copy()

print(f"  Total rows         : {len(df)}")
print(f"  Rows for training  : {len(df_train)}  (titles with ≥{MIN_SAMPLES_FOR_TRAINING} samples)")
print(f"  Titles in training : {df_train['Title'].nunique()}")
print(f"  Titles skipped     : {df['Title'].nunique() - df_train['Title'].nunique()}  (still in skill index)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Encode labels
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 4 — Encoding labels")

label_encoder = LabelEncoder()
labels        = label_encoder.fit_transform(df_train["Title"])

print(f"  Classes : {len(label_encoder.classes_)}")
print(f"  Sample  : {list(label_encoder.classes_[:5])}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Generate sentence embeddings
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 5 — Generating sentence embeddings")

print(f"  Loading model : {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

texts = df_train["combined_text"].tolist()

print(f"  Encoding {len(texts)} rows…")
features = embedding_model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

print(f"  Embedding shape : {features.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Train / test split
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 6 — Train / test split")

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Train classifier
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 7 — Training Logistic Regression classifier")

classifier = LogisticRegression(
    max_iter=2000,
    C=1.0,
    solver="lbfgs"
)

classifier.fit(X_train, y_train)
print("  Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 8 — Evaluation")

predictions = classifier.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"\n  Accuracy : {accuracy:.4f}  ({round(accuracy*100, 2)}%)")
print("\n  Classification Report:")
print(classification_report(
    y_test, predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save models
# ─────────────────────────────────────────────────────────────────────────────
separator("STEP 9 — Saving models")

joblib.dump(classifier,    JOB_CLASSIFIER_OUT)
joblib.dump(label_encoder, JOB_LABEL_ENCODER_OUT)

print(f"  Saved → {JOB_CLASSIFIER_OUT}")
print(f"  Saved → {JOB_LABEL_ENCODER_OUT}")
print(f"  Saved → {SKILL_INDEX_OUT}  (already saved in Step 2)")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
separator("DONE ✅")
print(f"""
  Files generated:
    ├── {JOB_CLASSIFIER_OUT:<30}  Job title classifier
    ├── {JOB_LABEL_ENCODER_OUT:<30}  Label encoder for job titles
    └── {SKILL_INDEX_OUT:<30}  Title → skills lookup (used by app.py)

  What the app can now do:
    • HR selects a job title from a dropdown
    • Skills are auto-filled from skill_index.json
    • Resumes are screened against those skills instantly
    • The classifier can also predict the closest job title
      from any free-text job description
""")