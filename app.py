"""
app.py
===============================================================================
AI Resume Screening System — Production Streamlit App

Tabs
----
  1. Bulk ML Screening   : ML match scores for all uploaded resumes vs. a job
  2. Deep Resume Analysis: Full ML + skill gap + feature explainability + Gemini

Architecture
------------
  All model loading → core.model_registry.ModelRegistry
  All visualisation  → core.visualisation
  All ML matching    → core.matcher.ResumeJobMatcher  (NO cosine_similarity)
  All skill logic    → utils.skills
  File extraction    → utils.file_extractor
  Config             → core.config.config
===============================================================================
"""

from __future__ import annotations

import streamlit as st

from core.config import config
from core.model_registry import ModelRegistry
from core import visualisation as viz
from utils.file_extractor import extract_resume
from utils.gemini_explainer import get_hiring_explanation
from utils.preprocessing import clean_text, truncate
from utils.skills import match_typed_skills, skill_gap_analysis


# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("🤖 AI Resume Screening System")


# ═══════════════════════════════════════════════════════════
# MODEL REGISTRY  (loaded once per Streamlit session)
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_registry() -> ModelRegistry:
    return ModelRegistry()


registry = get_registry()


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "PDF or DOCX — one or more",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for uf in uploaded_files:
            st.caption(f"• {uf.name}")
    else:
        st.caption("No resumes uploaded yet.")

    # Model health
    st.markdown("---")
    st.subheader("⚙️ Model Status")
    for emoji, msg in registry.health.sidebar_items():
        st.caption(f"{emoji} {msg}")

    # Gemini key
    st.markdown("---")
    st.subheader("🔑 Gemini API Key")
    gemini_key = st.text_input(
        "For AI hiring recommendations",
        type="password",
        value=st.session_state.get("gemini_key", ""),
    )
    if gemini_key:
        st.session_state["gemini_key"] = gemini_key


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🎯 Bulk ML Screening", "📊 Deep Resume Analysis"])


# ───────────────────────────────────────────────────────────
# Shared: build job description text from title or free text
# ───────────────────────────────────────────────────────────

def _jd_input_widget(key_suffix: str, show_threshold: bool = True):
    """
    Renders the job description input section and returns
    (job_text: str, threshold: int).
    """
    skill_index = registry.skill_index

    mode = st.radio(
        "Job description source",
        ["📋 Select Job Title", "✏️ Paste Job Description"],
        horizontal=True,
        key=f"mode_{key_suffix}",
    )

    job_text  = ""
    threshold = config.ui.default_threshold

    if mode == "📋 Select Job Title":
        if not skill_index:
            st.error(
                "Skill index not loaded. "
                "Run: `python training/train_job_classifier.py`"
            )
        else:
            col_t, col_s = st.columns([3, 1])
            with col_t:
                title = st.selectbox(
                    "Select Job Title",
                    ["— choose —"] + sorted(skill_index.keys()),
                    key=f"title_{key_suffix}",
                )
            with col_s:
                if show_threshold:
                    threshold = st.slider(
                        "Min ML Score %", 0, 100,
                        config.ui.default_threshold, 5,
                        key=f"thresh_{key_suffix}",
                    )

            if title != "— choose —":
                skills_str = ", ".join(skill_index[title])
                job_text   = f"Job Title: {title}\nRequired Skills: {skills_str}"
                st.markdown(f"**Skills for '{title}':**")
                st.markdown(
                    viz.skill_badges_html(
                        skill_index[title], "#d6eaf8", "#1a5276"
                    ),
                    unsafe_allow_html=True,
                )
    else:
        col_j, col_s = st.columns([3, 1])
        with col_j:
            job_text = st.text_area(
                "Paste Job Description",
                height=160,
                placeholder="Paste the full job description here…",
                key=f"jd_paste_{key_suffix}",
            )
        with col_s:
            if show_threshold:
                threshold = st.slider(
                    "Min ML Score %", 0, 100,
                    config.ui.default_threshold, 5,
                    key=f"thresh2_{key_suffix}",
                )

    return job_text, threshold


# ═══════════════════════════════════════════════════════════
# TAB 1 — BULK ML SCREENING
# ═══════════════════════════════════════════════════════════

with tab1:
    st.subheader("Bulk ML Screening")
    st.markdown(
        "Screen all uploaded resumes against a job description using the "
        "**ML matcher** — trained on 20 engineered features, not cosine similarity."
    )

    job_text_bulk, threshold_bulk = _jd_input_widget("bulk")
    st.markdown("")

    if st.button("▶ Run ML Screening", use_container_width=True, key="btn_bulk"):
        if not uploaded_files:
            st.warning("Upload at least one resume from the sidebar.")
        elif not job_text_bulk.strip():
            st.warning("Select a job title or paste a job description.")
        else:
            results = []
            progress = st.progress(0, text="Scoring resumes…")

            for idx, uf in enumerate(uploaded_files):
                raw = extract_resume(uf)
                if not raw.strip():
                    st.warning(f"Could not read {uf.name} — skipped.")
                    continue

                # ── ML MATCH ──────────────────────────────────────────────
                match_result = registry.matcher.predict(raw, job_text_bulk)

                clean_res = clean_text(raw)
                clean_jd  = clean_text(job_text_bulk)
                matched_sk, missing_sk = skill_gap_analysis(clean_res, clean_jd)

                results.append({
                    "name":       uf.name,
                    "match":      match_result,
                    "matched_sk": matched_sk,
                    "missing_sk": missing_sk,
                })
                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Scored {idx+1}/{len(uploaded_files)}",
                )

            progress.empty()

            if not results:
                st.error("No results — check that resume files are readable.")
            else:
                qualified = sorted(
                    [r for r in results if r["match"].score_pct >= threshold_bulk],
                    key=lambda x: x["match"].score, reverse=True,
                )
                rejected = sorted(
                    [r for r in results if r["match"].score_pct < threshold_bulk],
                    key=lambda x: x["match"].score, reverse=True,
                )

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Screened",          len(results))
                c2.metric("✅ Qualified",       len(qualified))
                c3.metric("❌ Below Threshold", len(rejected))

                if not qualified:
                    st.error(
                        f"No resumes reached {threshold_bulk}%. "
                        "Lower the threshold or broaden the job description."
                    )
                else:
                    st.markdown("### ✅ Qualified Candidates")

                    for i in range(0, len(qualified), config.ui.cards_per_row):
                        row  = qualified[i : i + config.ui.cards_per_row]
                        cols = st.columns(len(row))

                        for col, r in zip(cols, row):
                            with col:
                                m = r["match"]
                                st.markdown(
                                    viz.result_card_html(r["name"], m),
                                    unsafe_allow_html=True,
                                )
                                pie = viz.skill_pie(
                                    r["matched_sk"], r["missing_sk"], r["name"]
                                )
                                if pie:
                                    st.image(pie, use_container_width=True)
                                if r["matched_sk"]:
                                    st.markdown(
                                        "**Matched:** "
                                        + viz.skill_badges_html(
                                            r["matched_sk"], "#d5f5e3", "#1e8449", "✓"
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                if r["missing_sk"]:
                                    st.markdown(
                                        "**Missing:** "
                                        + viz.skill_badges_html(
                                            r["missing_sk"], "#fadbd8", "#922b21", "✗"
                                        ),
                                        unsafe_allow_html=True,
                                    )

                if rejected:
                    with st.expander(
                        f"❌ {len(rejected)} resume(s) below {threshold_bulk}%"
                    ):
                        for r in rejected:
                            m = r["match"]
                            st.markdown(
                                f"- **{r['name']}** — {m.score_pct}%  "
                                + viz.match_badge(m.label),
                                unsafe_allow_html=True,
                            )


# ═══════════════════════════════════════════════════════════
# TAB 2 — DEEP RESUME ANALYSIS
# ═══════════════════════════════════════════════════════════

with tab2:
    st.subheader("Deep Resume Analysis")
    st.markdown(
        "Full ML match breakdown — score, label, confidence, "
        "feature contributions, skill gap, and Gemini hiring recommendation."
    )

    if not uploaded_files:
        st.info("Upload resumes from the sidebar to use this tab.")
    else:
        selected_name = st.selectbox(
            "Select resume to analyse",
            [uf.name for uf in uploaded_files],
            key="sel_deep",
        )
        selected_file = next(
            uf for uf in uploaded_files if uf.name == selected_name
        )

        job_text_deep, _ = _jd_input_widget("deep", show_threshold=False)

        if st.button("▶ Analyse Resume", use_container_width=True, key="btn_deep"):
            if not job_text_deep.strip():
                st.warning("Select a job title or paste a job description.")
            else:
                with st.spinner("Running full ML analysis…"):
                    raw_text    = extract_resume(selected_file)
                    resume_text = clean_text(raw_text)
                    jd_clean    = clean_text(job_text_deep)

                    # ── ML MATCH ──────────────────────────────────────────
                    match_result = registry.matcher.predict(resume_text, job_text_deep)

                    # Resume category
                    category = registry.predict_resume_category(resume_text)

                    # Predicted job title from free-text JD
                    predicted_title = registry.predict_job_title(jd_clean) or "—"

                    # Skill gap
                    matched_sk, missing_sk = skill_gap_analysis(resume_text, jd_clean)

                # ── Results ───────────────────────────────────────────────
                st.markdown("### 📊 ML Match Results")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Resume Category",   category)
                c2.metric("Predicted JD Title",predicted_title)
                c3.metric("ML Match Score",    f"{match_result.score_pct}%")
                c4.metric("Label",             match_result.label)
                c5.metric("Confidence",        f"{match_result.confidence*100:.1f}%")

                st.markdown(
                    viz.score_bar_html(match_result.score_pct, match_result.label),
                    unsafe_allow_html=True,
                )

                # Feature explainability
                st.markdown("### 🔬 Feature Contributions")
                feat_buf = viz.feature_bar(match_result.feature_contributions)
                if feat_buf:
                    _, mid, _ = st.columns([1, 3, 1])
                    with mid:
                        st.image(feat_buf, use_container_width=True)

                with st.expander("View all feature values"):
                    cols = st.columns(3)
                    for i, (k, v) in enumerate(
                        match_result.feature_contributions.items()
                    ):
                        cols[i % 3].metric(
                            k.replace("_", " ").title(), f"{v:.4f}"
                        )

                # Skill gap
                st.markdown("### 📋 Skill Gap Analysis")
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Matched Skills**")
                    for s in sorted(matched_sk):
                        st.markdown(f"✅ `{s}`")
                    if not matched_sk:
                        st.caption("None detected")
                with cb:
                    st.markdown("**Missing Skills**")
                    for s in sorted(missing_sk):
                        st.markdown(f"❌ `{s}`")
                    if not missing_sk:
                        st.caption("No gaps detected")

                if matched_sk or missing_sk:
                    st.markdown("### Skill Coverage")
                    pie = viz.skill_pie(
                        list(matched_sk), list(missing_sk), selected_name
                    )
                    if pie:
                        _, mid, _ = st.columns([1, 2, 1])
                        with mid:
                            st.image(pie, use_container_width=True)

                # Gemini AI recommendation
                st.markdown("### 🤖 AI Hiring Recommendation")
                api_key = st.session_state.get("gemini_key", "")
                if not api_key:
                    st.info(
                        "Enter your Gemini API key in the sidebar "
                        "to generate an AI hiring recommendation."
                    )
                else:
                    with st.spinner("Asking Gemini…"):
                        explanation = get_hiring_explanation(
                            api_key=api_key,
                            resume_text=truncate(
                                resume_text, config.ui.max_resume_words
                            ),
                            job_description=job_text_deep[
                                : config.ui.max_jd_chars
                            ],
                            match_result=match_result,
                            matched_skills=matched_sk,
                            missing_skills=missing_sk,
                        )
                    if explanation:
                        st.markdown(explanation)
                    else:
                        st.warning(
                            "Gemini explanation unavailable — check your API key."
                        )
