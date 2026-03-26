"""
Trainmodel.py
─────────────────────────────────────────────────────────────────────────────
Trains the resume classifier AND builds a skill-frequency vocabulary
from the dataset so the skills list stays data-driven.

Steps performed:
  1. Load & clean resume_dataset.csv
  2. Extract skill frequencies across all resumes  →  saves skill_vocab.json
  3. Generate sentence embeddings (all-MiniLM-L6-v2)
  4. Train a Logistic Regression classifier
  5. Evaluate and save resume_classifier.pkl + label_encoder.pkl
─────────────────────────────────────────────────────────────────────────────
"""

import json
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak these if needed
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH      = "data/resume_dataset.csv"
CLASSIFIER_OUT    = "resume_classifier.pkl"
LABEL_ENCODER_OUT = "label_encoder.pkl"
SKILL_VOCAB_OUT   = "skill_vocab.json"   # NEW — used by skills.py at runtime

# Skills that are always included regardless of frequency in the dataset
SEED_SKILLS = [
    "python", "java", "c++", "c#", "r", "scala", "go", "rust", "kotlin", "swift",
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "data analysis", "data science",
    "statistics", "data visualization",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "opencv", "hugging face", "transformers", "langchain", "spacy", "nltk",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "linux", "bash", "git", "github", "ci/cd", "devops", "terraform",
    "javascript", "typescript", "react", "angular", "vue", "node", "nodejs",
    "html", "css", "flask", "django", "fastapi", "rest api", "graphql",
    "communication", "leadership", "management", "teamwork", "problem solving",
    "excel", "power bi", "tableau", "looker", "jupyter",
    "hadoop", "spark", "airflow", "kafka", "etl",
    "llm", "generative ai", "prompt engineering", "fine-tuning"
]

# Minimum number of resumes a skill must appear in to be added from the dataset
MIN_SKILL_FREQUENCY = 10


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading dataset")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

required_columns = {"Resume_str", "Category"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

df = df.dropna(subset=["Resume_str", "Category"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total samples  : {len(df)}")
print(f"  Categories     : {df['Category'].nunique()}")
print(f"  Category list  : {sorted(df['Category'].unique())}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build skill vocabulary from dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Building skill vocabulary from dataset")
print("=" * 60)

def extract_skill_counts(texts, candidate_skills):
    """Count how many documents each skill appears in."""
    counts = Counter()
    for text in texts:
        text_lower = text.lower()
        for skill in candidate_skills:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                counts[skill] += 1
    return counts

# First pass: count seed skills in the dataset
print("  Counting seed skills across resumes…")
counts = extract_skill_counts(df["Resume_str"].tolist(), SEED_SKILLS)

# Build final vocab: seed skills + any that meet frequency threshold
data_driven_skills = [s for s, c in counts.items() if c >= MIN_SKILL_FREQUENCY]
final_skill_vocab  = sorted(set(SEED_SKILLS) | set(data_driven_skills))

print(f"  Seed skills          : {len(SEED_SKILLS)}")
print(f"  Data-driven additions: {len(set(data_driven_skills) - set(SEED_SKILLS))}")
print(f"  Final vocab size     : {len(final_skill_vocab)}")

# Save vocab so skills.py can load it at runtime
with open(SKILL_VOCAB_OUT, "w") as f:
    json.dump(final_skill_vocab, f, indent=2)

print(f"  Saved → {SKILL_VOCAB_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Generate embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Generating sentence embeddings")
print("=" * 60)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

resumes    = df["Resume_str"].tolist()
categories = df["Category"].tolist()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)

print("  Encoding resumes (this may take a few minutes)…")
features = embedding_model.encode(resumes, show_progress_bar=True, batch_size=64)
print(f"  Embedding shape: {features.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train classifier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Training classifier")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"  Train samples: {len(X_train)}")
print(f"  Test  samples: {len(X_test)}")

classifier = LogisticRegression(max_iter=2000, C=1.0)
classifier.fit(X_train, y_train)
print("  Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Evaluation")
print("=" * 60)

predictions = classifier.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"\n  Accuracy : {accuracy:.4f}")
print("\n  Classification Report:")
print(classification_report(
    y_test, predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save models
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6 — Saving models")
print("=" * 60)

joblib.dump(classifier,    CLASSIFIER_OUT)
joblib.dump(label_encoder, LABEL_ENCODER_OUT)

print(f"  Saved → {CLASSIFIER_OUT}")
print(f"  Saved → {LABEL_ENCODER_OUT}")
print(f"  Saved → {SKILL_VOCAB_OUT}")
print("\n✅  All done!")