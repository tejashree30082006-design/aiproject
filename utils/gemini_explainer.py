"""
utils/gemini_explainer.py
═══════════════════════════════════════════════════════════════════════════════
Gemini LLM explanation layer.

Wraps the Gemini API to produce hiring recommendations from structured
ML matcher output, rather than passing raw text directly (which was the
previous approach).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Optional

from core.matcher import MatchResult

log = logging.getLogger(__name__)


def get_hiring_explanation(
    api_key: str,
    resume_text: str,
    job_description: str,
    match_result: MatchResult,
    matched_skills: list[str],
    missing_skills: list[str],
) -> Optional[str]:
    """
    Generate a structured hiring recommendation from the ML match result.

    Parameters
    ----------
    api_key         : Gemini API key
    resume_text     : raw (truncated) resume text
    job_description : raw job description text
    match_result    : output of ResumeJobMatcher.predict()
    matched_skills  : skills found in both resume and JD
    missing_skills  : skills required by JD but absent from resume

    Returns
    -------
    Formatted explanation string, or None on failure.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        log.warning("google-generativeai not installed. Skipping Gemini explanation.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""You are a senior technical recruiter reviewing a candidate.

ML Match Score : {match_result.score_pct:.1f}%
Match Label    : {match_result.label}
Model Confidence: {match_result.confidence * 100:.1f}%

Matched Skills : {", ".join(matched_skills) or "None detected"}
Missing Skills : {", ".join(missing_skills) or "None"}

Job Description (excerpt):
{job_description[:1000]}

Resume (excerpt):
{resume_text[:1500]}

Please provide a structured hiring assessment with:
1. **Match Summary** — why the score is {match_result.label}
2. **Candidate Strengths** — top 3-5 strengths relevant to this role
3. **Skill Gaps** — critical missing skills and their importance
4. **Recommendation** — one of: Shortlist / Consider / Reject
5. **Next Steps** — suggested interview focus areas

Keep it professional, concise, and actionable."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as exc:
        log.error("Gemini explanation failed: %s", exc)
        return None
