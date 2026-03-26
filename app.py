import json
import os
import io

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdfplumber
import docx
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text
from utils.skills import match_typed_skills, skill_match_analysis


# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("🤖 AI Resume Screening System")


# ═══════════════════════════════════════════════════════════
# LOAD MODELS & SKILL INDEX
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    classifier      = joblib.load("resume_classifier.pkl")
    label_encoder   = joblib.load("label_encoder.pkl")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load job classifier if available (trained from job_dataset.csv)
    job_classifier    = joblib.load("job_classifier.pkl")    if os.path.exists("job_classifier.pkl")    else None
    job_label_encoder = joblib.load("job_label_encoder.pkl") if os.path.exists("job_label_encoder.pkl") else None

    return classifier, label_encoder, embedding_model, job_classifier, job_label_encoder


@st.cache_data
def load_skill_index():
    """Load title → skills mapping built during training."""
    if os.path.exists("skill_index.json"):
        with open("skill_index.json") as f:
            return json.load(f)
    return {}


classifier, label_encoder, embedding_model, job_classifier, job_label_encoder = load_models()
skill_index = load_skill_index()


# ═══════════════════════════════════════════════════════════
# HELPERS — file extraction
# ═══════════════════════════════════════════════════════════
def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_docx(file):
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])

def get_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_pdf(uploaded_file)
    return extract_docx(uploaded_file)


# ═══════════════════════════════════════════════════════════
# HELPERS — pie chart
# ═══════════════════════════════════════════════════════════
def draw_skill_pie(matched, missing, candidate_name):
    n_matched = len(matched)
    n_missing = len(missing)
    total     = n_matched + n_missing
    if total == 0:
        return None

    sizes  = [1] * n_matched + [1] * n_missing
    colors = ["#2ecc71"] * n_matched + ["#e74c3c"] * n_missing

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    _, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 6 else "",
        startangle=140,
        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
        at.set_fontweight("bold")

    legend_handles = (
        [mpatches.Patch(color="#2ecc71", label=s) for s in matched] +
        [mpatches.Patch(color="#e74c3c", label=s) for s in missing]
    )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        frameon=False,
    )
    pct = round(n_matched / total * 100, 1)
    ax.set_title(f"{candidate_name}\n{pct}% matched", fontsize=9, fontweight="bold", pad=8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════
# SIDEBAR — resume upload
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    if uploaded_files:
        st.success(f"{len(uploaded_files)} resume(s) uploaded")
        for uf in uploaded_files:
            st.caption(f"• {uf.name}")
    else:
        st.caption("No resumes uploaded yet.")

    if skill_index:
        st.markdown("---")
        st.caption(f"✅ Skill index loaded — {len(skill_index)} job titles available")
    else:
        st.markdown("---")
        st.warning("skill_index.json not found. Run Trainmodel.py first.")


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🎯 Bulk Skill Screening", "📊 Full Resume Analysis"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — BULK SKILL SCREENING
# ═══════════════════════════════════════════════════════════
with tab1:

    st.subheader("Bulk Skill Screening")
    st.markdown(
        "Select a job title to auto-load its required skills, "
        "or type skills manually. Then screen all uploaded resumes at once."
    )

    # ── Skill input mode toggle ────────────────────────────
    input_mode = st.radio(
        "How would you like to define required skills?",
        options=["📋 Select Job Title (auto-fill skills)", "✏️ Type Skills Manually"],
        horizontal=True,
    )

    skills_for_screening = ""   # will be set by either mode

    if input_mode == "📋 Select Job Title (auto-fill skills)":

        if not skill_index:
            st.error("skill_index.json not found. Please run Trainmodel.py first.")
        else:
            col_title, col_thresh = st.columns([3, 1])

            with col_title:
                sorted_titles = sorted(skill_index.keys())
                selected_title = st.selectbox(
                    "Select Job Title",
                    options=["— choose a title —"] + sorted_titles,
                )

            with col_thresh:
                threshold = st.slider("Min Match %", 0, 100, 50, 5)

            if selected_title != "— choose a title —":
                auto_skills = skill_index[selected_title]
                skills_for_screening = ", ".join(auto_skills)

                st.markdown(f"**Skills auto-loaded for '{selected_title}'** ({len(auto_skills)} skills):")
                badges_html = " ".join(
                    f"<span style='background:#d6eaf8;color:#1a5276;padding:3px 10px;"
                    f"border-radius:12px;font-size:12px;display:inline-block;margin:2px;'>"
                    f"{s}</span>"
                    for s in auto_skills
                )
                st.markdown(badges_html, unsafe_allow_html=True)

                # Allow HR to edit/remove skills after auto-fill
                edited = st.text_input(
                    "Edit skills if needed (comma-separated)",
                    value=skills_for_screening,
                )
                skills_for_screening = edited

    else:
        col_manual, col_thresh2 = st.columns([3, 1])
        with col_manual:
            skills_for_screening = st.text_input(
                "Required Skills (comma-separated)",
                placeholder="e.g.  python, machine learning, docker, sql",
            )
        with col_thresh2:
            threshold = st.slider("Min Match %", 0, 100, 50, 5, key="thresh_manual")

    st.markdown("")

    # ── Screen button ──────────────────────────────────────
    if st.button("▶ Screen Resumes", use_container_width=True):

        if not uploaded_files:
            st.warning("Upload at least one resume from the sidebar.")

        elif not skills_for_screening.strip():
            st.warning("Please select a job title or enter skills first.")

        else:
            results = []
            with st.spinner(f"Screening {len(uploaded_files)} resume(s)…"):
                for uf in uploaded_files:
                    raw     = get_resume_text(uf)
                    cleaned = clean_text(raw)
                    matched, missing, pct = match_typed_skills(cleaned, skills_for_screening)
                    results.append({
                        "name":    uf.name,
                        "matched": matched,
                        "missing": missing,
                        "pct":     pct,
                    })

            qualified = sorted(
                [r for r in results if r["pct"] >= threshold],
                key=lambda x: x["pct"], reverse=True
            )
            rejected = sorted(
                [r for r in results if r["pct"] < threshold],
                key=lambda x: x["pct"], reverse=True
            )

            # ── Summary metrics ────────────────────────────
            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Resumes Screened",  len(results))
            sc2.metric("✅ Qualified",       len(qualified))
            sc3.metric("❌ Below Threshold", len(rejected))

            if not qualified:
                st.error(
                    f"No resumes met the {threshold}% threshold. "
                    "Try lowering the slider or broadening the skills."
                )

            else:
                st.markdown(
                    f"### ✅ Qualified Candidates  "
                    f"<span style='font-size:14px;color:#888;font-weight:normal;'>"
                    f"sorted by match score</span>",
                    unsafe_allow_html=True,
                )

                # 2 cards per row
                for i in range(0, len(qualified), 2):
                    row = qualified[i : i + 2]
                    cols = st.columns(len(row))

                    for col, r in zip(cols, row):
                        with col:
                            bar_color = (
                                "#2ecc71" if r["pct"] >= 70
                                else "#f39c12" if r["pct"] >= 40
                                else "#e74c3c"
                            )

                            # Card with progress bar
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ddd;border-radius:12px;
                                            padding:16px;margin-bottom:8px;">
                                  <h4 style="margin:0 0 8px 0;">📄 {r['name']}</h4>
                                  <div style="background:#e8e8e8;border-radius:8px;
                                              height:24px;width:100%;margin-bottom:6px;">
                                    <div style="background:{bar_color};width:{r['pct']}%;
                                                height:100%;border-radius:8px;
                                                display:flex;align-items:center;
                                                padding-left:10px;color:#fff;
                                                font-weight:700;font-size:13px;">
                                      {r['pct']}%
                                    </div>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Pie chart
                            pie_buf = draw_skill_pie(r["matched"], r["missing"], r["name"])
                            if pie_buf:
                                st.image(pie_buf, use_container_width=True)

                            # Matched badges
                            if r["matched"]:
                                matched_html = " ".join(
                                    f"<span style='background:#d5f5e3;color:#1e8449;"
                                    f"padding:3px 9px;border-radius:12px;"
                                    f"font-size:12px;display:inline-block;margin:2px;'>"
                                    f"✓ {s}</span>"
                                    for s in sorted(r["matched"])
                                )
                                st.markdown(f"**Matched:** {matched_html}", unsafe_allow_html=True)

                            # Missing badges
                            if r["missing"]:
                                missing_html = " ".join(
                                    f"<span style='background:#fadbd8;color:#922b21;"
                                    f"padding:3px 9px;border-radius:12px;"
                                    f"font-size:12px;display:inline-block;margin:2px;'>"
                                    f"✗ {s}</span>"
                                    for s in sorted(r["missing"])
                                )
                                st.markdown(f"**Missing:** {missing_html}", unsafe_allow_html=True)

            # ── Rejected summary ───────────────────────────
            if rejected:
                with st.expander(f"❌ {len(rejected)} resume(s) below {threshold}% threshold"):
                    for r in rejected:
                        st.markdown(
                            f"- **{r['name']}** — {r['pct']}% &nbsp; "
                            f"({len(r['matched'])} / {len(r['matched']) + len(r['missing'])} skills matched)"
                        )


# ═══════════════════════════════════════════════════════════
# TAB 2 — FULL RESUME ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab2:

    st.subheader("Full Resume Analysis")
    st.markdown(
        "Pick one resume, select or paste a job description, "
        "and get a full AI-powered analysis with skill gap and hiring recommendation."
    )

    if not uploaded_files:
        st.info("Upload resumes from the sidebar to use this tab.")

    else:
        selected_name = st.selectbox(
            "Select resume to analyse",
            options=[uf.name for uf in uploaded_files],
        )
        selected_file = next(uf for uf in uploaded_files if uf.name == selected_name)

        # ── JD source: job title or free-text ─────────────
        jd_mode = st.radio(
            "Job description source",
            ["📋 Pick from Job Titles", "✏️ Paste manually"],
            horizontal=True,
        )

        job_description = ""

        if jd_mode == "📋 Pick from Job Titles" and skill_index:
            jd_title = st.selectbox(
                "Select Job Title for JD",
                options=["— choose —"] + sorted(skill_index.keys()),
                key="jd_title_select",
            )
            if jd_title != "— choose —":
                # Build a synthetic JD from the skill index
                skills_str    = ", ".join(skill_index[jd_title])
                job_description = (
                    f"Job Title: {jd_title}\n"
                    f"Required Skills: {skills_str}"
                )
                st.info(f"JD auto-built from skill index for **{jd_title}**")
        else:
            job_description = st.text_area(
                "Paste Job Description",
                height=180,
                placeholder="Paste the full job description here…",
            )

        if st.button("▶ Analyse Resume", use_container_width=True):

            if not job_description.strip():
                st.warning("Please provide a job description.")

            else:
                with st.spinner("Analysing…"):

                    raw_text    = get_resume_text(selected_file)
                    resume_text = clean_text(raw_text)
                    jd_clean    = clean_text(job_description)

                    resume_vec  = embedding_model.encode([resume_text])
                    job_vec     = embedding_model.encode([jd_clean])
                    similarity  = cosine_similarity(resume_vec, job_vec)[0][0]

                    prediction  = classifier.predict(resume_vec)
                    category    = label_encoder.inverse_transform(prediction)[0]

                    # ── Metrics ────────────────────────────
                    st.markdown("### Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Resume Category",  category)
                    c2.metric("Job Match Score",  f"{round(similarity * 100, 2)} %")

                    # ── Skill gap ──────────────────────────
                    st.markdown("### 📋 Skill Gap")
                    matched_jd, missing_jd = skill_match_analysis(resume_text, jd_clean)

                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**Matched**")
                        for s in sorted(matched_jd):
                            st.markdown(f"✅ `{s}`")
                        if not matched_jd:
                            st.write("None detected")
                    with cb:
                        st.markdown("**Missing**")
                        for s in sorted(missing_jd):
                            st.markdown(f"❌ `{s}`")
                        if not missing_jd:
                            st.write("No gaps detected")

                    # ── Pie chart for this candidate ───────
                    if matched_jd or missing_jd:
                        st.markdown("### Skill Coverage")
                        pie_buf = draw_skill_pie(
                            list(matched_jd), list(missing_jd), selected_name
                        )
                        if pie_buf:
                            _, mid, _ = st.columns([1, 2, 1])
                            with mid:
                                st.image(pie_buf, use_container_width=True)

                    # ── Gemini explanation ─────────────────
                    try:
                        import google.generativeai as genai
                        GEMINI_API_KEY = "GEMINI_API_KEY"   # ← replace
                        genai.configure(api_key=GEMINI_API_KEY)
                        gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                        prompt = f"""
You are an AI hiring assistant.

Job Description:
{job_description}

Resume:
{resume_text[:2000]}

Explain:
1. Why this resume matches or doesn't match the job.
2. Strengths of the candidate.
3. Missing skills or experience gaps.
4. Final hiring recommendation (Shortlist / Maybe / Reject).

Keep it professional and structured.
"""
                        response = gemini_model.generate_content(prompt)
                        st.markdown("### 🤖 AI Hiring Explanation")
                        st.write(response.text)

                    except Exception as e:
                        st.warning(f"Gemini explanation skipped — check your API key. ({e})")