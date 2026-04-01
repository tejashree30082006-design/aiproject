# AI Resume Screening System

ML-powered resume ↔ job description matching. **No cosine similarity** — a trained XGBoost classifier on 20 engineered features replaces it entirely.

---

## Architecture

```
aiproject/
├── app.py                          # Streamlit UI
├── core/
│   ├── config.py                   # All constants & paths (one place)
│   ├── matcher.py                  # ResumeJobMatcher — ML model API
│   ├── feature_engineer.py         # 20-feature engineering pipeline
│   ├── fallback_scorer.py          # Zero-shot fallback (no training needed)
│   ├── model_registry.py           # Centralised model loading & health
│   └── visualisation.py            # All charts & HTML components
├── training/
│   ├── train_matcher.py            # Train the Resume↔Job ML model
│   └── train_job_classifier.py     # Train the job title classifier
├── utils/
│   ├── preprocessing.py            # Text cleaning
│   ├── skills.py                   # Skill extraction (SkillMatchResult)
│   ├── file_extractor.py           # PDF / DOCX extraction
│   └── gemini_explainer.py         # Gemini LLM wrapper
├── models/                         # Trained .pkl files (git-ignored)
├── data/                           # CSV datasets (git-ignored)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place datasets
#    data/job_dataset.csv     (columns: Title, Skills, Keywords, Responsibilities)
#    data/resume_dataset.csv  (columns: resume_text or similar, category)

# 3. Train job title classifier + build skill index
python training/train_job_classifier.py

# 4. Train the ML resume↔job matcher (replaces cosine similarity)
python training/train_matcher.py

# 5. Run the app
streamlit run app.py
```

---

## ML Matching Model

The core of the system. `training/train_matcher.py`:

1. Generates synthetic Strong / Good / Weak resume↔job pairs from your CSVs
2. Encodes each pair with `all-MiniLM-L6-v2` sentence embeddings
3. Engineers 20 features per pair (skill overlap, keyword density, section signals, embedding geometry)
4. Trains `XGBClassifier` with `CalibratedClassifierCV` for calibrated probabilities
5. Outputs `models/matcher.pkl`

**Feature list** (`core/feature_engineer.py`):

| Feature | Description |
|---|---|
| `cosine_sim` | Embedding cosine (one input, not the final score) |
| `skill_jaccard` | Jaccard similarity over skill vocab |
| `skill_coverage_ratio` | Overlap skills / required skills |
| `skill_gap_ratio` | 1 − coverage (missing ratio) |
| `keyword_overlap` | Bag-of-words Jaccard |
| `resume_yoe` | Years of experience mentioned |
| `has_experience_sec` | Resume has an experience section |
| `has_skills_sec` | Resume has a skills section |
| `l2_distance` | Embedding L2 distance |
| `dot_product_norm` | Normalised embedding dot product |
| … + 10 more | Section presence, counts, magnitude |

---

## Fallback Mode

If `models/matcher.pkl` is not yet trained, the app runs automatically in
**EmbeddingOnlyScorer** fallback mode — a weighted combination of skill
coverage, Jaccard, and cosine gives a reasonable zero-shot baseline.
The sidebar shows `⚠️ ML Matcher — fallback mode`.

---

## Models

| File | Trained by | Purpose |
|---|---|---|
| `models/matcher.pkl` | `train_matcher.py` | Resume↔Job ML match score |
| `models/job_classifier.pkl` | `train_job_classifier.py` | Job title prediction |
| `models/job_label_encoder.pkl` | `train_job_classifier.py` | Label decoding |
| `models/resume_classifier.pkl` | *(original Trainmodel.py)* | Resume category |
| `models/label_encoder.pkl` | *(original Trainmodel.py)* | Label decoding |
| `models/skill_index.json` | `train_job_classifier.py` | Title → skills lookup |

---

## Configuration

All constants are in `core/config.py`. Change thresholds, paths, training
hyperparameters, or UI defaults there — nothing else needs editing.

```python
from core.config import config
print(config.matcher.strong_threshold)   # 0.70
print(config.training.n_pairs)           # 5000
```
