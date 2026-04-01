"""
training/train_from_hf_dataset.py
===============================================================================
Train the Resume <-> Job ML Matcher using the HuggingFace dataset:
  cnamuangtoun/resume-job-description-fit

Dataset labels (only 2 classes exist in this dataset):
  "No Fit"   -> 0  (Weak)
  "Good Fit" -> 1  (Good / Strong)

Because the dataset has no "Strong Fit" samples, we train a 2-class model
and remap scores so the app's 3-label display still works:
  Class 0 probability  -> Weak
  Class 1 probability  -> Good  (displayed as Strong when score >= 0.70)

Run (from ANY directory)
────
  python training/train_from_hf_dataset.py
  -- OR --
  cd training && python train_from_hf_dataset.py

Output
──────
  models/matcher.pkl  — picked up automatically by app.py
===============================================================================
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Anchor to project root — works from aiproject/ OR aiproject/training/ ────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.feature_engineer import FeatureEngineer
from core.matcher import ResumeJobMatcher
from utils.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

HF_DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
OUTPUT_PATH     = os.path.join(PROJECT_ROOT, "models", "matcher.pkl")
EMBED_MODEL     = "all-MiniLM-L6-v2"
TEST_SIZE       = 0.2
RANDOM_SEED     = 42


def _sep(title: str) -> None:
    log.info("\n" + "=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


# ── Step 1: Load from HuggingFace ─────────────────────────────────────────────

def load_hf_data() -> tuple[list[str], list[str], list[int], dict[int, str], list[str]]:
    """
    Returns:
        resume_texts, jd_texts, labels (int),
        int_to_name (e.g. {0: 'No Fit', 1: 'Good Fit'}),
        label_names (ordered list for classification_report)
    """
    _sep("STEP 1 — Loading HuggingFace dataset")
    log.info("Dataset: %s", HF_DATASET_NAME)

    ds = load_dataset(HF_DATASET_NAME)

    splits_to_use = []
    for split_name in ["train", "test", "validation"]:
        if split_name in ds:
            splits_to_use.append(ds[split_name])
            log.info("  Split '%s': %d rows", split_name, len(ds[split_name]))

    combined = concatenate_datasets(splits_to_use)
    log.info("Total rows: %d", len(combined))

    # ── Discover actual labels in the data ───────────────────────────────────
    unique_labels = sorted(set(combined["label"]))
    log.info("Unique labels found: %s", unique_labels)

    # Build label map dynamically — priority order: No Fit=0, Good Fit=1, Strong Fit=2
    label_map: dict[str, int] = {}
    int_to_name: dict[int, str] = {}

    # First pass: assign by keyword
    priority = []
    for raw in unique_labels:
        lower = raw.strip().lower()
        if "no" in lower or "weak" in lower:
            priority.append((0, raw))
        elif "strong" in lower:
            priority.append((2, raw))
        else:
            priority.append((1, raw))   # "good fit" or anything else

    # Sort by intended class id, then re-assign contiguously (0, 1, 2, ...)
    # so sklearn never sees a gap in class indices
    priority.sort(key=lambda x: x[0])
    for new_idx, (_, raw) in enumerate(priority):
        label_map[raw] = new_idx
        int_to_name[new_idx] = raw

    log.info("Label mapping: %s", label_map)

    # ── Build parallel lists ─────────────────────────────────────────────────
    resume_texts, jd_texts, labels = [], [], []
    skipped = 0

    for row in combined:
        rt  = str(row["resume_text"]).strip()
        jt  = str(row["job_description_text"]).strip()
        lbl = row["label"]

        if not rt or not jt or lbl not in label_map:
            skipped += 1
            continue

        resume_texts.append(clean_text(rt))
        jd_texts.append(clean_text(jt))
        labels.append(label_map[lbl])

    if skipped:
        log.warning("Skipped %d rows (empty text or unknown label)", skipped)

    label_arr  = np.array(labels)
    label_names = [int_to_name[i] for i in sorted(int_to_name.keys())]

    for i, name in enumerate(label_names):
        log.info("  Class %d (%s): %d samples", i, name, (label_arr == i).sum())

    return resume_texts, jd_texts, labels, int_to_name, label_names


# ── Step 2: Build feature matrix ──────────────────────────────────────────────

def build_features(
    resume_texts: list[str],
    jd_texts: list[str],
    embed_model: SentenceTransformer,
    fe: FeatureEngineer,
) -> np.ndarray:
    log.info("Encoding %d resume texts...", len(resume_texts))
    resume_vecs = embed_model.encode(
        resume_texts, show_progress_bar=True, batch_size=64
    )
    log.info("Encoding %d JD texts...", len(jd_texts))
    jd_vecs = embed_model.encode(
        jd_texts, show_progress_bar=True, batch_size=64
    )

    log.info("Engineering features per pair...")
    rows = []
    for rv, jv, rt, jt in zip(resume_vecs, jd_vecs, resume_texts, jd_texts):
        row = fe._feature_vector(rt, jt, rv[np.newaxis], jv[np.newaxis])
        rows.append(row)

    X = np.array(rows, dtype=np.float32)
    log.info("Feature matrix shape: %s", X.shape)
    return X


# ── Step 3: Build classifier ──────────────────────────────────────────────────

def build_classifier():
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
        )
        log.info("Classifier: XGBClassifier")
    except ImportError:
        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        log.info("xgboost not found — using LogisticRegression fallback")
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Project root: %s", PROJECT_ROOT)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

    # Step 1: Load
    resume_texts, jd_texts, labels, int_to_name, label_names = load_hf_data()
    y = np.array(labels)

    n_classes = len(label_names)
    log.info("Training a %d-class model: %s", n_classes, label_names)

    # Step 2: Split
    indices = np.arange(len(resume_texts))
    idx_tr, idx_te = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    res_tr = [resume_texts[i] for i in idx_tr]
    jd_tr  = [jd_texts[i]     for i in idx_tr]
    y_tr   = y[idx_tr]
    res_te = [resume_texts[i] for i in idx_te]
    jd_te  = [jd_texts[i]     for i in idx_te]
    y_te   = y[idx_te]

    log.info("Train: %d | Test: %d", len(res_tr), len(res_te))

    # Step 3: Embed + engineer features
    _sep("STEP 2 — Loading embedding model")
    embed_model = SentenceTransformer(EMBED_MODEL)
    fe = FeatureEngineer()

    _sep("STEP 3 — Building feature matrices")
    X_tr = build_features(res_tr, jd_tr, embed_model, fe)
    X_te = build_features(res_te, jd_te, embed_model, fe)

    # Step 4: Train
    _sep("STEP 4 — Training classifier")
    base_clf = build_classifier()
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    calibrated_clf.fit(X_tr, y_tr)

    # Step 5: Evaluate — use only the classes actually present in the data
    _sep("STEP 5 — Evaluation")
    y_pred = calibrated_clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    log.info("Accuracy: %.4f  (%.2f%%)", acc, acc * 100)

    present_classes = sorted(set(y_te))
    present_names   = [label_names[i] for i in present_classes]
    log.info(
        "\n%s",
        classification_report(
            y_te, y_pred,
            labels=present_classes,
            target_names=present_names,
            zero_division=0,
        ),
    )

    # Step 6: Save
    _sep("STEP 6 — Saving model")
    matcher = ResumeJobMatcher(
        classifier=calibrated_clf,
        feature_engineer=fe,
        embedding_model=embed_model,
    )
    matcher.save(OUTPUT_PATH)
    log.info("Matcher saved -> %s", OUTPUT_PATH)
    log.info("Classes trained: %s", int_to_name)
    log.info("app.py will pick it up automatically on next launch.")


if __name__ == "__main__":
    main()