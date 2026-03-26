import streamlit as st
import joblib
import pdfplumber
import docx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text
from utils.skills import match_typed_skills, skill_match_analysis


# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI Resume Screening System")


# ─────────────────────────────
# LOAD MODELS
# ─────────────────────────────
@st.cache_resource
def load_models():
    classifier      = joblib.load("resume_classifier.pkl")
    label_encoder   = joblib.load("label_encoder.pkl")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return classifier, label_encoder, embedding_model

classifier, label_encoder, embedding_model = load_models()


# ─────────────────────────────
# HELPERS
# ─────────────────────────────
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

def draw_skill_pie(matched, missing, candidate_name):
    """Draw a pie chart showing matched vs missing skills breakdown."""
    n_matched = len(matched)
    n_missing = len(missing)
    total     = n_matched + n_missing

    if total == 0:
        return None

    # Each skill gets its own slice so labels show individual skills
    sizes  = [1] * n_matched + [1] * n_missing
    labels = matched + missing
    colors = ["#2ecc71"] * n_matched + ["#e74c3c"] * n_missing

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    wedges, texts, autotexts = ax.pie(
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

    # Legend
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


# ─────────────────────────────
# SIDEBAR — resume upload only
# ─────────────────────────────
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.caption(f"{len(uploaded_files)} resume(s) uploaded" if uploaded_files else "No resumes uploaded yet.")


# ─────────────────────────────
# TABS
# ─────────────────────────────
tab1, tab2 = st.tabs(["🎯 Bulk Skill Screening", "📊 Full Resume Analysis"])


# ══════════════════════════════════════════════════════════
# TAB 1 — BULK SKILL SCREENING
# ══════════════════════════════════════════════════════════
with tab1:

    st.subheader("Bulk Skill Screening")
    st.markdown(
        "Upload multiple resumes in the sidebar, type the required skills below, "
        "set a minimum match threshold, and only qualifying candidates will be shown."
    )

    col_input, col_thresh = st.columns([3, 1])

    with col_input:
        typed_skills_input = st.text_input(
            "Required Skills (comma-separated)",
            placeholder="e.g.  python, machine learning, docker, sql, communication"
        )

    with col_thresh:
        threshold = st.slider("Min Match %", min_value=0, max_value=100, value=50, step=5)

    if st.button("▶ Screen Resumes", use_container_width=True):

        if not uploaded_files:
            st.warning("Upload at least one resume from the sidebar.")

        elif not typed_skills_input.strip():
            st.warning("Enter at least one required skill.")

        else:
            required_skills = [s.strip().lower() for s in typed_skills_input.split(",") if s.strip()]

            results = []

            with st.spinner(f"Screening {len(uploaded_files)} resume(s)…"):
                for uf in uploaded_files:
                    raw      = get_resume_text(uf)
                    cleaned  = clean_text(raw)
                    matched, missing, pct = match_typed_skills(cleaned, typed_skills_input)
                    results.append({
                        "name":     uf.name,
                        "matched":  matched,
                        "missing":  missing,
                        "pct":      pct,
                    })

            # Filter by threshold
            qualified = [r for r in results if r["pct"] >= threshold]
            qualified.sort(key=lambda x: x["pct"], reverse=True)

            rejected  = [r for r in results if r["pct"] < threshold]

            # ── Summary bar ────────────────────────────────────────
            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Resumes Screened",  len(results))
            sc2.metric("✅ Qualified",       len(qualified))
            sc3.metric("❌ Below Threshold", len(rejected))

            if not qualified:
                st.error(f"No resumes met the {threshold}% match threshold. Try lowering it.")

            else:
                st.markdown(f"### ✅ Qualified Candidates — top matches for: `{typed_skills_input}`")

                # Show 2 candidate cards per row
                for i in range(0, len(qualified), 2):
                    row_candidates = qualified[i:i+2]
                    cols = st.columns(len(row_candidates))

                    for col, r in zip(cols, row_candidates):
                        with col:
                            # Score colour
                            bar_color = (
                                "#2ecc71" if r["pct"] >= 70
                                else "#f39c12" if r["pct"] >= 40
                                else "#e74c3c"
                            )

                            # Card header
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ddd; border-radius:12px;
                                            padding:16px; margin-bottom:8px;">
                                  <h4 style="margin:0 0 8px 0;">📄 {r['name']}</h4>
                                  <div style="background:#e8e8e8;border-radius:8px;
                                              height:22px;width:100%;margin-bottom:10px;">
                                    <div style="background:{bar_color};width:{r['pct']}%;
                                                height:100%;border-radius:8px;
                                                display:flex;align-items:center;
                                                padding-left:8px;color:#fff;
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

                            # Skill badges
                            if r["matched"]:
                                matched_html = " ".join(
                                    f"<span style='background:#d5f5e3;color:#1e8449;"
                                    f"padding:3px 9px;border-radius:12px;"
                                    f"font-size:12px;display:inline-block;margin:2px;'>"
                                    f"✓ {s}</span>"
                                    for s in sorted(r["matched"])
                                )
                                st.markdown(f"**Matched:** {matched_html}", unsafe_allow_html=True)

                            if r["missing"]:
                                missing_html = " ".join(
                                    f"<span style='background:#fadbd8;color:#922b21;"
                                    f"padding:3px 9px;border-radius:12px;"
                                    f"font-size:12px;display:inline-block;margin:2px;'>"
                                    f"✗ {s}</span>"
                                    for s in sorted(r["missing"])
                                )
                                st.markdown(f"**Missing:** {missing_html}", unsafe_allow_html=True)

            # ── Rejected summary ───────────────────────────────────
            if rejected:
                with st.expander(f"❌ {len(rejected)} resume(s) below {threshold}% threshold"):
                    for r in sorted(rejected, key=lambda x: x["pct"], reverse=True):
                        st.markdown(
                            f"- **{r['name']}** — {r['pct']}% matched "
                            f"({len(r['matched'])}/{len(r['matched'])+len(r['missing'])} skills)"
                        )


# ══════════════════════════════════════════════════════════
# TAB 2 — FULL RESUME ANALYSIS (single resume + JD)
# ══════════════════════════════════════════════════════════
with tab2:

    st.subheader("Full Resume Analysis")
    st.markdown("Select one resume from those uploaded, then paste a job description for a deep AI analysis.")

    if not uploaded_files:
        st.info("Upload resumes from the sidebar to use this tab.")

    else:
        selected_name = st.selectbox(
            "Select resume to analyse",
            options=[uf.name for uf in uploaded_files]
        )
        selected_file = next(uf for uf in uploaded_files if uf.name == selected_name)

        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the job description here…"
        )

        if st.button("▶ Analyse Resume", use_container_width=True):

            if not job_description.strip():
                st.warning("Please enter a job description above.")

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

                    st.markdown("### Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Resume Category", category)
                    c2.metric("Job Match Score", f"{round(similarity * 100, 2)} %")

                    st.markdown("### 📋 Skill Gap (from Job Description)")
                    matched_jd, missing_jd = skill_match_analysis(resume_text, jd_clean)

                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**Matched**")
                        for s in sorted(matched_jd): st.markdown(f"✅ `{s}`")
                        if not matched_jd: st.write("None detected")
                    with cb:
                        st.markdown("**Missing**")
                        for s in sorted(missing_jd): st.markdown(f"❌ `{s}`")
                        if not missing_jd: st.write("No gaps detected")

                    # Gemini explanation — only if API key is set
                    try:
                        import google.generativeai as genai
                        GEMINI_API_KEY = "AIzaSyDb0C7XctW10eN1FbO5jBqffQ7YvWD9m5I"   # ← replace
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
3. Missing skills.
4. Final hiring recommendation.

Keep it professional and structured.
"""
                        response = gemini_model.generate_content(prompt)
                        st.markdown("### 🤖 AI Hiring Explanation")
                        st.write(response.text)

                    except Exception as e:
                        st.warning(f"Gemini explanation skipped: {e}")