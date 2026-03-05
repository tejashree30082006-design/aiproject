import re

skills_list = [
"python","java","c++","sql","machine learning","deep learning",
"data analysis","pandas","numpy","tensorflow","pytorch",
"aws","azure","docker","kubernetes","linux",
"javascript","react","node","html","css",
"communication","leadership","management",
"excel","power bi","tableau"
]


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills_list:

        if re.search(r"\b" + re.escape(skill) + r"\b", text):

            found.append(skill)

    return list(set(found))


def skill_match_analysis(resume_text, job_desc):

    resume_skills = set(extract_skills(resume_text))

    job_skills = set(extract_skills(job_desc))

    matched = resume_skills.intersection(job_skills)

    missing = job_skills - resume_skills

    return list(matched), list(missing)