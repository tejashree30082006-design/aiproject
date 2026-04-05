"""
app.py — RecruitIQ  (clean rewrite)
Uses native Streamlit components properly.
Minimal CSS — no inline style spam.
JD carries forward from Bulk Screening into Candidate Review.
"""
from __future__ import annotations
import streamlit as st
from core.config import config
from core.model_registry import ModelRegistry
from core import visualisation as viz
from utils.file_extractor import extract_resume
from utils.gemini_explainer import get_hiring_explanation
from utils.preprocessing import clean_text, truncate
from utils.skills import skill_gap_analysis

st.set_page_config(
    page_title="RecruitIQ · AI Resume Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Single CSS block — written once, no inline styles ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1280px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 14px 24px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #f8fafc !important;
    padding: 2rem 0 0 !important;
}

/* Buttons */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 24px !important; font-weight: 600 !important; }

/* Radio */
.stRadio label { font-size: 14px !important; color: #374151 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* Progress */
.stProgress > div { background: #e2e8f0 !important; border-radius: 99px !important; }
.stProgress > div > div { background: #2563eb !important; border-radius: 99px !important; }

/* Alerts */
.stAlert { border-radius: 8px !important; font-size: 14px !important; }

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 1.25rem 0 !important; }

/* Slider */
.stSlider > div > div > div { background: #2563eb !important; }

/* Caption */
.stCaption { color: #64748b !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Registry ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_registry():
    return ModelRegistry()
registry = get_registry()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### RecruitIQ")
    st.caption("AI-powered resume screening")
    st.divider()

    st.markdown("**Upload Resumes**")
    uploaded_files = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''} ready")
        for uf in uploaded_files:
            st.caption(f"📄 {uf.name}")
    else:
        st.caption("No files uploaded yet")

    st.divider()
    st.markdown("**Model Status**")
    for emoji, msg in registry.health.sidebar_items():
        st.caption(f"{emoji} {msg}")

    st.divider()
    st.markdown("**Gemini API Key**")
    gemini_key = st.text_input(
        "key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.get("gemini_key", ""),
        label_visibility="collapsed",
    )
    if gemini_key:
        st.session_state["gemini_key"] = gemini_key
        st.caption("✅ Key saved for this session")
    else:
        st.caption("Required for AI recommendations")


# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("## RecruitIQ")
st.caption("Screen resumes against a job description using XGBoost + Gemini")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_color(label: str) -> str:
    return {"Strong": "green", "Good": "orange", "Weak": "red"}.get(label, "gray")


def _score_bar_html(score_pct: float, label: str) -> str:
    colors = {"Strong": "#16a34a", "Good": "#d97706", "Weak": "#dc2626"}
    c = colors.get(label, "#2563eb")
    return (
        f'<div style="background:#e2e8f0;border-radius:99px;height:6px;margin:8px 0 2px;">'
        f'<div style="background:{c};width:{score_pct}%;height:100%;border-radius:99px;"></div>'
        f'</div>'
    )


def _skill_badges_html(skills: list[str], matched: bool) -> str:
    if not skills:
        return ""
    if matched:
        style = "background:#dcfce7;color:#166534;border:1px solid #bbf7d0;"
        icon  = "✓"
    else:
        style = "background:#fee2e2;color:#991b1b;border:1px solid #fecaca;"
        icon  = "✗"
    base = (
        "display:inline-block;font-size:12px;font-weight:500;"
        "padding:3px 10px;border-radius:99px;margin:2px;"
    )
    return "".join(
        f'<span style="{base}{style}">{icon} {s}</span>' for s in skills
    )


def _jd_widget(key: str, show_threshold: bool = True):
    """
    Renders JD source picker. Reads from session_state first so the
    JD selected in Bulk Screening is pre-filled in Candidate Review.
    Returns (job_text, threshold).
    """
    skill_index = registry.skill_index

    col_mode, col_thresh = (
        st.columns([4, 1]) if show_threshold else (st.container(), None)
    )

    with col_mode:
        mode = st.radio(
            "Source",
            ["Select job title", "Paste job description"],
            horizontal=True,
            key=f"jd_mode_{key}",
            label_visibility="collapsed",
        )

    threshold = config.ui.default_threshold
    if show_threshold and col_thresh:
        with col_thresh:
            threshold = st.slider(
                "Min score %", 0, 100,
                config.ui.default_threshold, 5,
                key=f"thresh_{key}",
            )

    job_text = ""

    if mode == "Select job title":
        if not skill_index:
            st.error("Skill index missing — run train_job_classifier.py")
        else:
            # pre-fill from session_state if set by bulk screening
            saved_title = st.session_state.get("last_jd_title", "— choose —")
            options     = ["— choose —"] + sorted(skill_index.keys())
            default_idx = options.index(saved_title) if saved_title in options else 0

            title = st.selectbox(
                "Job title",
                options,
                index=default_idx,
                key=f"jd_title_{key}",
                label_visibility="collapsed",
            )
            if title != "— choose —":
                st.session_state["last_jd_title"]  = title
                st.session_state["last_jd_mode"]   = "title"
                st.session_state["last_jd_text"]   = ""
                skills   = skill_index[title]
                job_text = f"Job Title: {title}\nRequired Skills: {', '.join(skills)}"
                with st.expander(f"Skills required for {title}", expanded=False):
                    st.markdown(
                        _skill_badges_html(skills, matched=True),
                        unsafe_allow_html=True,
                    )
    else:
        # pre-fill textarea if user previously pasted something
        saved_text = st.session_state.get("last_jd_text", "")
        job_text   = st.text_area(
            "Job description",
            value=saved_text,
            height=140,
            placeholder="Paste the full job description here…",
            key=f"jd_paste_{key}",
            label_visibility="collapsed",
        )
        if job_text.strip():
            st.session_state["last_jd_text"] = job_text
            st.session_state["last_jd_mode"] = "paste"
            st.session_state["last_jd_title"] = "— choose —"

    return job_text, threshold


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Bulk Screening", "Candidate Review"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BULK SCREENING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Bulk Screening")
    st.caption("Score all uploaded resumes against a job description in one go")

    job_text_bulk, threshold_bulk = _jd_widget("bulk", show_threshold=True)
    st.write("")

    run = st.button("Run Screening", use_container_width=True, key="btn_bulk")

    if run:
        if not uploaded_files:
            st.warning("Upload at least one resume from the sidebar.")
        elif not job_text_bulk.strip():
            st.warning("Choose a job title or paste a job description.")
        else:
            results = []
            bar = st.progress(0, text="Scoring…")

            for idx, uf in enumerate(uploaded_files):
                raw = extract_resume(uf)
                if not raw.strip():
                    st.warning(f"{uf.name} could not be read — skipped.")
                    continue
                mr             = registry.matcher.predict(raw, job_text_bulk)
                ms, xs         = skill_gap_analysis(clean_text(raw), clean_text(job_text_bulk))
                results.append({"name": uf.name, "match": mr, "matched": ms, "missing": xs})
                bar.progress((idx + 1) / len(uploaded_files),
                             text=f"Scored {idx + 1}/{len(uploaded_files)}")
            bar.empty()

            if not results:
                st.error("No readable resumes found.")
            else:
                qualified = sorted(
                    [r for r in results if r["match"].score_pct >= threshold_bulk],
                    key=lambda x: x["match"].score, reverse=True,
                )
                rejected = sorted(
                    [r for r in results if r["match"].score_pct < threshold_bulk],
                    key=lambda x: x["match"].score, reverse=True,
                )

                # Save qualified names + the JD text for Candidate Review
                st.session_state["qualified_names"]  = [r["name"] for r in qualified]
                st.session_state["bulk_job_text"]    = job_text_bulk

                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Screened",        len(results))
                c2.metric("Qualified",        len(qualified))
                c3.metric("Below threshold",  len(rejected))
                st.write("")

                if not qualified:
                    st.warning(
                        f"No resumes reached {threshold_bulk}%. "
                        "Try lowering the threshold."
                    )
                else:
                    st.markdown(f"#### Qualified candidates — above {threshold_bulk}%")
                    for r in qualified:
                        m  = r["match"]
                        lc = _label_color(m.label)

                        with st.container():
                            col_name, col_score, col_label, col_conf = st.columns([3, 1, 1, 1])
                            col_name.markdown(f"**📄 {r['name']}**")
                            col_score.metric("Score", f"{m.score_pct}%")
                            col_label.metric("Label", m.label)
                            col_conf.metric("Confidence", f"{m.confidence*100:.0f}%")

                            st.markdown(
                                _score_bar_html(m.score_pct, m.label),
                                unsafe_allow_html=True,
                            )

                            # Skill gap — always shown
                            if r["matched"] or r["missing"]:
                                sk_col1, sk_col2 = st.columns(2)
                                with sk_col1:
                                    st.caption("✅ Matched skills")
                                    if r["matched"]:
                                        st.markdown(
                                            _skill_badges_html(r["matched"], True),
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.caption("None detected")
                                with sk_col2:
                                    st.caption("❌ Missing skills")
                                    if r["missing"]:
                                        st.markdown(
                                            _skill_badges_html(r["missing"], False),
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.caption("No gaps")
                            else:
                                st.caption("No skills detected from vocabulary in this pair")

                            st.divider()

                if rejected:
                    with st.expander(
                        f"Below threshold — {len(rejected)} resume(s) under {threshold_bulk}%"
                    ):
                        for r in rejected:
                            m = r["match"]
                            st.markdown(
                                f"**{r['name']}** — "
                                f"Score: **{m.score_pct}%** · "
                                f"Label: :{_label_color(m.label)}[{m.label}]"
                            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CANDIDATE REVIEW
# JD carries forward from Bulk Screening automatically.
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Candidate Review")
    st.caption("Full ML breakdown, skill gap and Gemini recommendation for one candidate")

    if not uploaded_files:
        st.info("Upload resumes from the sidebar to begin.")
    else:
        # ── Candidate selector — qualified only if screening was run ──────
        qualified_names = st.session_state.get("qualified_names")
        if qualified_names:
            available = [uf for uf in uploaded_files if uf.name in qualified_names]
            st.info(
                f"Showing {len(available)} qualified candidate(s) from last screening. "
                "Run screening again to refresh."
            )
        else:
            available = list(uploaded_files)
            st.info("Run Bulk Screening first to filter to qualified candidates.")

        if not available:
            st.warning("No qualified candidates. Lower the threshold and re-run screening.")
            st.stop()

        selected_name = st.selectbox(
            "Select candidate",
            [uf.name for uf in available],
            key="sel_review",
        )
        selected_file = next(uf for uf in available if uf.name == selected_name)

        st.divider()

        # ── JD — pre-filled from bulk screening ──────────────────────────
        st.markdown("**Job description**")

        # If bulk screening was run, use that JD and show it read-only
        bulk_jd = st.session_state.get("bulk_job_text", "")
        if bulk_jd:
            st.success("Using job description from Bulk Screening.")
            with st.expander("View job description", expanded=False):
                st.text(bulk_jd[:800] + ("…" if len(bulk_jd) > 800 else ""))
            job_text_review = bulk_jd
            # Still allow override
            if st.checkbox("Use a different job description for this candidate"):
                _, job_text_review = _jd_widget("review_override", show_threshold=False)
        else:
            _, job_text_review = _jd_widget("review", show_threshold=False)

        st.write("")
        analyse = st.button("Analyse Candidate", use_container_width=True, key="btn_review")

        if analyse:
            if not job_text_review.strip():
                st.warning("Choose a job title or paste a job description.")
                st.stop()

            with st.spinner("Running analysis…"):
                raw_text    = extract_resume(selected_file)
                resume_text = clean_text(raw_text)
                jd_clean    = clean_text(job_text_review)

                match_result    = registry.matcher.predict(resume_text, job_text_review)
                category        = registry.predict_resume_category(resume_text)
                predicted_title = registry.predict_job_title(jd_clean) or "—"
                matched_sk, missing_sk = skill_gap_analysis(resume_text, jd_clean)

            st.divider()

            # ── Score banner ──────────────────────────────────────────────
            lc = _label_color(match_result.label)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ML Score",        f"{match_result.score_pct}%")
            c2.metric("Label",           match_result.label)
            c3.metric("Confidence",      f"{match_result.confidence*100:.0f}%")
            c4.metric("Resume Category", category)
            c5.metric("Predicted Title", predicted_title)

            st.markdown(
                _score_bar_html(match_result.score_pct, match_result.label),
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Skill gap ─────────────────────────────────────────────────
            st.markdown("#### Skill Gap")
            col_m, col_x = st.columns(2)

            with col_m:
                st.markdown(f"**✅ Matched skills** ({len(matched_sk)})")
                if matched_sk:
                    st.markdown(
                        _skill_badges_html(sorted(matched_sk), True),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No matching skills found in vocabulary")

            with col_x:
                st.markdown(f"**❌ Missing skills** ({len(missing_sk)})")
                if missing_sk:
                    st.markdown(
                        _skill_badges_html(sorted(missing_sk), False),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No skill gaps detected")

            st.divider()

            # ── Feature contributions ─────────────────────────────────────
            st.markdown("#### Feature Contributions")
            st.caption("The 20 signals the model used to arrive at this score")
            buf = viz.feature_bar(match_result.feature_contributions)
            if buf:
                _, mid, _ = st.columns([1, 3, 1])
                with mid:
                    st.image(buf, use_container_width=True)

            with st.expander("All 20 feature values"):
                cols3 = st.columns(3)
                for i, (k, v) in enumerate(match_result.feature_contributions.items()):
                    cols3[i % 3].metric(k.replace("_", " ").title(), f"{v:.4f}")

            st.divider()

            # ── Gemini ────────────────────────────────────────────────────
            st.markdown("#### AI Hiring Recommendation")
            api_key = st.session_state.get("gemini_key", "")
            if not api_key:
                st.info("Add your Gemini API key in the sidebar for an AI recommendation.")
            else:
                with st.spinner("Asking Gemini…"):
                    explanation = get_hiring_explanation(
                        api_key=api_key,
                        resume_text=truncate(resume_text, config.ui.max_resume_words),
                        job_description=job_text_review[:config.ui.max_jd_chars],
                        match_result=match_result,
                        matched_skills=matched_sk,
                        missing_skills=missing_sk,
                    )
                if explanation:
                    st.markdown(explanation)
                else:
                    st.warning("Gemini unavailable — check your API key.")