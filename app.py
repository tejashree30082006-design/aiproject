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

            # Save to session state so Tab 2 can filter by matched/unmatched
            st.session_state["screening_results"]   = results
            st.session_state["screening_threshold"] = threshold

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

    if not uploaded_files:
        st.info("Upload resumes from the sidebar to use this tab.")

    else:

        # ── Step 1: Filter resumes ─────────────────────────────
        st.markdown("#### Step 1 — Choose resumes to analyse")

        screening_results = st.session_state.get("screening_results", [])
        threshold_used    = st.session_state.get("screening_threshold", 50)

        if screening_results:
            matched_names   = [r["name"] for r in screening_results if r["pct"] >= threshold_used]
            unmatched_names = [r["name"] for r in screening_results if r["pct"] < threshold_used]
            filter_options  = ["✅ Matched resumes only", "❌ Unmatched resumes only", "📂 All uploaded resumes"]
        else:
            matched_names   = []
            unmatched_names = []
            filter_options  = ["📂 All uploaded resumes"]

        resume_filter = st.radio(
            "Which resumes to include?",
            options=filter_options,
            horizontal=True,
        )

        if resume_filter == "✅ Matched resumes only" and matched_names:
            pool = [uf for uf in uploaded_files if uf.name in matched_names]
            st.success(f"{len(pool)} matched resume(s) from Tab 1 screening.")
        elif resume_filter == "❌ Unmatched resumes only" and unmatched_names:
            pool = [uf for uf in uploaded_files if uf.name in unmatched_names]
            st.warning(f"{len(pool)} unmatched resume(s) from Tab 1 screening.")
        else:
            pool = list(uploaded_files)
            if resume_filter != "📂 All uploaded resumes" and not screening_results:
                st.info("Run Tab 1 screening first to filter. Showing all resumes.")

        if not pool:
            st.warning("No resumes in this filter group. Try a different option or run Tab 1 first.")

        else:
            st.markdown("---")

            # ── Step 2: Job Description ────────────────────────────
            st.markdown("#### Step 2 — Job Description")

            jd_mode = st.radio(
                "Job description source",
                ["📋 Pick from Job Titles", "✏️ Paste manually"],
                horizontal=True,
                key="tab2_jd_mode",
            )

            job_description = ""

            if jd_mode == "📋 Pick from Job Titles" and skill_index:
                jd_title = st.selectbox(
                    "Select Job Title",
                    options=["— choose —"] + sorted(skill_index.keys()),
                    key="jd_title_select",
                )
                if jd_title != "— choose —":
                    skills_str      = ", ".join(skill_index[jd_title])
                    job_description = f"Job Title: {jd_title}\nRequired Skills: {skills_str}"
                    st.info(f"JD auto-built for **{jd_title}**")
            else:
                job_description = st.text_area(
                    "Paste Job Description",
                    height=160,
                    placeholder="Paste the full job description here…",
                    key="tab2_jd_text",
                )

            st.markdown("---")

            # ── Step 3: Analysis mode ──────────────────────────────
            st.markdown("#### Step 3 — Choose analysis mode")

            analysis_mode = st.radio(
                "How would you like to analyse?",
                options=[
                    "🏆 Analyse all at once — find the best candidate",
                    "🔍 Analyse one by one manually",
                ],
                horizontal=True,
            )

            st.markdown("")

            # ══════════════════════════════════════════════════════
            # MODE A — ALL AT ONCE
            # ══════════════════════════════════════════════════════
            if analysis_mode == "🏆 Analyse all at once — find the best candidate":

                if st.button("▶ Analyse All Resumes", use_container_width=True):

                    if not job_description.strip():
                        st.warning("Please provide a job description first.")

                    else:
                        all_results = []

                        with st.spinner(f"Analysing {len(pool)} resume(s)…"):
                            jd_clean = clean_text(job_description)
                            job_vec  = embedding_model.encode([jd_clean])

                            for uf in pool:
                                raw_text    = get_resume_text(uf)
                                resume_text = clean_text(raw_text)
                                resume_vec  = embedding_model.encode([resume_text])
                                similarity  = cosine_similarity(resume_vec, job_vec)[0][0]
                                prediction  = classifier.predict(resume_vec)
                                category    = label_encoder.inverse_transform(prediction)[0]
                                matched_jd, missing_jd = skill_match_analysis(resume_text, jd_clean)
                                skill_pct = round(
                                    len(matched_jd) / max(len(matched_jd) + len(missing_jd), 1) * 100, 1
                                )
                                all_results.append({
                                    "name":        uf.name,
                                    "resume_text": resume_text,
                                    "similarity":  float(similarity),
                                    "category":    category,
                                    "matched_jd":  list(matched_jd),
                                    "missing_jd":  list(missing_jd),
                                    "skill_pct":   skill_pct,
                                })

                        all_results.sort(key=lambda x: x["similarity"], reverse=True)
                        best = all_results[0]

                        # ── Ranking table ──────────────────────────
                        st.markdown("### 📊 Candidate Rankings")
                        h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
                        h1.markdown("**Resume**")
                        h2.markdown("**Category**")
                        h3.markdown("**Match Score**")
                        h4.markdown("**Skill Coverage**")
                        st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

                        for i, r in enumerate(all_results):
                            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                            c1.markdown(f"{medal} {r['name']}")
                            c2.markdown(f"`{r['category']}`")
                            sc = "#2ecc71" if r["similarity"] >= 0.7 else "#f39c12" if r["similarity"] >= 0.4 else "#e74c3c"
                            c3.markdown(
                                f"<span style='color:{sc};font-weight:700'>{round(r['similarity']*100,2)}%</span>",
                                unsafe_allow_html=True,
                            )
                            c4.markdown(f"{r['skill_pct']}%")

                        st.markdown("---")

                        # ── Best candidate ─────────────────────────
                        st.markdown(
                            f"### 🏆 Best Candidate: `{best['name']}`  "
                            f"<span style='font-size:15px;color:#888'>({round(best['similarity']*100,2)}% match)</span>",
                            unsafe_allow_html=True,
                        )

                        left_col, right_col = st.columns([1, 1])

                        with left_col:
                            st.markdown("**Skill Coverage**")
                            ca, cb = st.columns(2)
                            with ca:
                                st.markdown("Matched")
                                for s in sorted(best["matched_jd"]):
                                    st.markdown(f"✅ `{s}`")
                                if not best["matched_jd"]:
                                    st.write("None")
                            with cb:
                                st.markdown("Missing")
                                for s in sorted(best["missing_jd"]):
                                    st.markdown(f"❌ `{s}`")
                                if not best["missing_jd"]:
                                    st.write("None")

                        with right_col:
                            # Probability pie — each candidate slice sized by similarity
                            st.markdown("**Candidate Selection Probability**")
                            names       = [r["name"] for r in all_results]
                            scores      = [r["similarity"] for r in all_results]
                            total_score = sum(scores)
                            probs       = [round(s / total_score * 100, 1) for s in scores]

                            palette = [
                                "#2ecc71","#3498db","#9b59b6","#f39c12",
                                "#1abc9c","#e67e22","#e74c3c","#2980b9",
                                "#8e44ad","#27ae60","#d35400","#c0392b",
                            ]
                            pie_colors = [palette[i % len(palette)] for i in range(len(names))]

                            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="none")
                            _, _, autotexts = ax.pie(
                                probs,
                                labels=None,
                                colors=pie_colors,
                                autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
                                startangle=140,
                                wedgeprops=dict(linewidth=0.5, edgecolor="white"),
                                pctdistance=0.78,
                            )
                            for at in autotexts:
                                at.set_fontsize(7)
                                at.set_color("white")
                                at.set_fontweight("bold")

                            short_names = [n[:18] + "…" if len(n) > 18 else n for n in names]
                            legend_handles = [
                                mpatches.Patch(color=pie_colors[i], label=f"{short_names[i]} ({probs[i]}%)")
                                for i in range(len(names))
                            ]
                            ax.legend(handles=legend_handles, loc="center left",
                                      bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
                            ax.set_title("Selection Probability", fontsize=9, fontweight="bold", pad=8)

                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
                            plt.close(fig)
                            buf.seek(0)
                            st.image(buf, use_container_width=True)

                        st.markdown("---")

                        # ── Gemini comparative analysis ────────────
                        try:
                            import google.generativeai as genai
                            GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"   # ← replace
                            genai.configure(api_key=GEMINI_API_KEY)
                            gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                            others_summary = "\n".join([
                                f"- {r['name']}: {round(r['similarity']*100,2)}% match, "
                                f"matched skills: {r['matched_jd']}"
                                for r in all_results[1:]
                            ])

                            prompt = f"""
You are an AI hiring assistant comparing multiple candidates for a role.

Job Description:
{job_description}

BEST CANDIDATE — {best["name"]}:
Resume excerpt: {best["resume_text"][:1500]}
Match score: {round(best["similarity"]*100,2)}%
Matched skills: {best["matched_jd"]}
Missing skills: {best["missing_jd"]}

OTHER CANDIDATES:
{others_summary}

Your task:
1. Explain clearly why {best["name"]} is the best fit for this role.
2. Highlight what sets them apart from the other candidates.
3. Mention any risks or gaps even in the best candidate.
4. Final recommendation table: Shortlist / Interview / Reject for each candidate.

Keep it structured, professional, and concise.
"""
                            response = gemini_model.generate_content(prompt)
                            st.markdown("### 🤖 AI Comparative Analysis")
                            st.write(response.text)

                        except Exception as e:
                            st.warning(f"Gemini explanation skipped — check your API key. ({e})")

            # ══════════════════════════════════════════════════════
            # MODE B — ONE BY ONE
            # ══════════════════════════════════════════════════════
            else:

                selected_name = st.selectbox(
                    "Select resume to analyse",
                    options=[uf.name for uf in pool],
                    key="tab2_manual_select",
                )
                selected_file = next(uf for uf in pool if uf.name == selected_name)

                if st.button("▶ Analyse This Resume", use_container_width=True):

                    if not job_description.strip():
                        st.warning("Please provide a job description first.")

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

                            if matched_jd or missing_jd:
                                st.markdown("### Skill Coverage")
                                pie_buf = draw_skill_pie(
                                    list(matched_jd), list(missing_jd), selected_name
                                )
                                if pie_buf:
                                    _, mid, _ = st.columns([1, 2, 1])
                                    with mid:
                                        st.image(pie_buf, use_container_width=True)

                            try:
                                import google.generativeai as genai
                                GEMINI_API_KEY = "AIzaSyCaImY5GJzxboU276xg6rkOF41brClZvI8"   # ← replace
                                genai.configure(api_key=GEMINI_API_KEY)
                                gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                                prompt = f"""
You are an AI hiring assistant.

Job Description:
{job_description}

Resume:
{resume_text[:2000]}

Explain:
1. Why this resume matches or does not match the job.
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