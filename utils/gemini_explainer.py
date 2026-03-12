import google.generativeai as genai


def get_llm_explanation(api_key, resume_text, job_desc, score):

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an AI HR assistant.

Analyze the resume against the job description.

Resume:
{resume_text[:2000]}

Job Description:
{job_desc}

Match Score: {score}

Explain:
1. Why the resume matches or not
2. Candidate strengths
3. Missing skills
4. Hiring recommendation
"""

    response = model.generate_content(prompt)

    return response.text