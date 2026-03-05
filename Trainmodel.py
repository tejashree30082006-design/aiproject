import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv("data/resume_dataset.csv")

# -----------------------------
# Validate Required Columns
# -----------------------------
required_columns = {"Resume_str", "Category"}

if not required_columns.issubset(df.columns):
    raise ValueError("CSV must contain Resume_str and Category columns")

# -----------------------------
# Data Cleaning
# -----------------------------
print("Cleaning dataset...")

# Remove missing rows
df = df.dropna(subset=["Resume_str", "Category"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(df))

# -----------------------------
# Extract Text and Labels
# -----------------------------
resumes = df["Resume_str"].tolist()
categories = df["Category"].tolist()

# Convert labels to numeric
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)

print("Number of categories:", len(label_encoder.classes_))
print("Categories:", label_encoder.classes_)

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Generate Resume Embeddings
# -----------------------------
print("Generating embeddings...")

features = embedding_model.encode(
    resumes,
    show_progress_bar=True
)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# Train Classifier
# -----------------------------
print("Training model...")

classifier = LogisticRegression(max_iter=2000)

classifier.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
print("\nEvaluating model...")

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Save Model
# -----------------------------
print("\nSaving model...")

joblib.dump(classifier, "resume_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved as: resume_classifier.pkl")
print("Label encoder saved as: label_encoder.pkl")