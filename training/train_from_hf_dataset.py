"""
training/train_from_hf_dataset.py
===============================================================================
Train the Resume <-> Job ML Matcher using the HuggingFace dataset:
  cnamuangtoun/resume-job-description-fit

Fixes applied vs previous version
───────────────────────────────────
1. Class imbalance   — class_weight balancing so model doesn't just predict
                       "No Fit" for everything (80/20 imbalance in dataset)
2. Overfitting       — XGBoost tuned conservatively (fewer estimators, more
                       regularisation, lower depth) for 8k-row dataset
3. Dynamic labels    — target_names derived from actual data, no hardcoded 3
4. Stable accuracy   — fixed RANDOM_SEED used everywhere; results are
                       reproducible across runs

Run (from ANY directory)
────
  python training/train_from_hf_dataset.py
  -- OR --
  cd training && python train_from_hf_dataset.py

Output
──────
  models/matcher.pkl
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
from sklearn.utils.class_weight import compute_class_weight

# ── Anchor to project root ────────────────────────────────────────────────────
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
RANDOM_SEED     = 42      # fixed — results are reproducible across runs


def _sep(title: str) -> None:
    log.info("\n" + "=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_hf_data() -> tuple[list[str], list[str], list[int], list[str]]:
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

    unique_labels = sorted(set(combined["label"]))
    log.info("Unique labels found: %s", unique_labels)

    # Build label map — priority: No/Weak=0, Good=1, Strong=2
    priority_map = {}
    for raw in unique_labels:
        lower = raw.strip().lower()
        if "no" in lower or "weak" in lower:
            priority_map[raw] = 0
        elif "strong" in lower:
            priority_map[raw] = 2
        else:
            priority_map[raw] = 1

    # Re-index contiguously: 0,1,... (no gaps — sklearn requires this)
    sorted_pairs = sorted(set(priority_map.values()))
    remap = {old: new for new, old in enumerate(sorted_pairs)}
    label_map  = {raw: remap[pri] for raw, pri in priority_map.items()}
    int_to_name = {v: k for k, v in label_map.items()}
    label_names = [int_to_name[i] for i in sorted(int_to_name)]

    log.info("Label mapping: %s", label_map)

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
        log.warning("Skipped %d rows", skipped)

    label_arr = np.array(labels)
    for i, name in enumerate(label_names):
        count = (label_arr == i).sum()
        pct   = count / len(label_arr) * 100
        log.info("  Class %d (%s): %d samples (%.1f%%)", i, name, count, pct)

    return resume_texts, jd_texts, labels, label_names


# ── Step 2: Build features ────────────────────────────────────────────────────

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

def build_classifier(y_train: np.ndarray, n_classes: int):
    """
    Build a classifier with class_weight balancing to handle the ~80/20
    No Fit / Good Fit imbalance in the HF dataset.

    XGBoost is tuned conservatively for ~6k training rows:
      - fewer estimators (150 vs 300) to avoid overfitting
      - shallower depth (4 vs 6)
      - higher min_child_weight and reg_lambda for regularisation
    """
    # Compute balanced class weights
    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    log.info("Class weights (balanced): %s", dict(zip(classes, weights.round(3))))

    try:
        from xgboost import XGBClassifier

        # scale_pos_weight only works for binary — use sample_weight instead
        clf = XGBClassifier(
            n_estimators=150,        # reduced: 300 overfits on 6k rows
            max_depth=4,             # shallower: less overfitting
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,      # require more samples per leaf
            reg_lambda=2.0,          # L2 regularisation
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
        )
        log.info("Classifier: XGBClassifier (regularised for small dataset)")

        # Attach class weights as sample weights (applied in .fit())
        sample_weights = np.array([weights[y] for y in y_train])
        clf._sample_weights = sample_weights   # stored for use in main()

    except ImportError:
        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",           # built-in balancing for LR
            multi_class="multinomial",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        clf._sample_weights = None
        log.info("xgboost not found — using LogisticRegression (balanced)")

    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Project root: %s", PROJECT_ROOT)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

    # Step 1
    resume_texts, jd_texts, labels, label_names = load_hf_data()
    y = np.array(labels)
    n_classes = len(label_names)
    log.info("Training a %d-class model: %s", n_classes, label_names)

    # Step 2: Split — fixed seed so accuracy is stable across runs
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

    # Step 3: Embed + engineer
    _sep("STEP 2 — Loading embedding model")
    embed_model = SentenceTransformer(EMBED_MODEL)
    fe = FeatureEngineer()

    _sep("STEP 3 — Building feature matrices")
    X_tr = build_features(res_tr, jd_tr, embed_model, fe)
    X_te = build_features(res_te, jd_te, embed_model, fe)

    # Step 4: Train with class balancing
    _sep("STEP 4 — Training classifier")
    base_clf = build_classifier(y_tr, n_classes)

    # CalibratedClassifierCV wraps the classifier — pass sample_weight via fit_params
    sample_weights = getattr(base_clf, "_sample_weights", None)

    calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")

    if sample_weights is not None:
        calibrated_clf.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
        )
    else:
        calibrated_clf.fit(X_tr, y_tr)

    # Step 5: Evaluate
    _sep("STEP 5 — Evaluation")
    y_pred = calibrated_clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
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
    log.info("Classes: %s", dict(enumerate(label_names)))
    log.info("app.py will pick it up automatically on next launch.")


if __name__ == "__main__":
    main()