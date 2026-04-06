"""
utils/skills.py
═══════════════════════════════════════════════════════════════════════════════
Skill extraction and gap analysis.
Vocabulary expanded to cover: Cybersecurity, Networking, Finance, Healthcare,
HR, Marketing, Legal, Mechanical/Civil Engineering, and more job categories
present in the resume classifier.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


SKILLS_VOCAB: list[str] = [
    # ── Programming languages ─────────────────────────────────────────────
    "python", "java", "c++", "c#", "c", "r", "scala", "go", "rust",
    "kotlin", "swift", "php", "ruby", "matlab", "typescript", "javascript",
    "perl", "shell", "powershell", "vba", "assembly", "dart", "elixir",

    # ── Databases ─────────────────────────────────────────────────────────
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle",
    "redis", "cassandra", "elasticsearch", "dynamodb", "neo4j",
    "mariadb", "ms sql", "firebase", "supabase",

    # ── ML / AI ───────────────────────────────────────────────────────────
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "data analysis", "data science",
    "statistics", "data visualization", "feature engineering", "model deployment",
    "transfer learning", "fine-tuning", "prompt engineering", "llm",
    "generative ai", "rag", "vector database", "embeddings",
    "time series", "anomaly detection", "recommendation systems",
    "neural networks", "convolutional neural network", "cnn",
    "recurrent neural network", "rnn", "lstm", "transformer",

    # ── ML Libraries ──────────────────────────────────────────────────────
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "opencv", "hugging face", "transformers", "langchain", "spacy", "nltk",
    "xgboost", "lightgbm", "catboost", "mlflow", "wandb", "dvc",
    "matplotlib", "seaborn", "plotly",

    # ── Cloud / DevOps ────────────────────────────────────────────────────
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "linux", "bash", "git", "github", "gitlab", "ci/cd", "devops", "terraform",
    "jenkins", "airflow", "ansible", "helm", "prometheus", "grafana",
    "nginx", "apache", "rabbitmq", "celery",

    # ── Web ───────────────────────────────────────────────────────────────
    "react", "angular", "vue", "node", "nodejs", "html", "css",
    "flask", "django", "fastapi", "rest api", "graphql", "spring boot",
    "next.js", "nuxt", "svelte", "bootstrap", "tailwind",
    "wordpress", "shopify", "jquery", "webpack", "sass",

    # ── Data Engineering ──────────────────────────────────────────────────
    "hadoop", "spark", "kafka", "etl", "dbt", "bigquery", "snowflake",
    "databricks", "flink", "airflow", "data warehouse", "data pipeline",

    # ── BI / Analytics ────────────────────────────────────────────────────
    "excel", "power bi", "tableau", "looker", "jupyter",
    "google analytics", "mixpanel", "amplitude",

    # ── Cybersecurity ─────────────────────────────────────────────────────
    "penetration testing", "ethical hacking", "vulnerability assessment",
    "network security", "application security", "web application security",
    "kali linux", "metasploit", "nmap", "wireshark", "burp suite",
    "nessus", "snort", "splunk", "siem", "soc",
    "firewall", "ids", "ips", "vpn", "zero trust",
    "malware analysis", "reverse engineering", "forensics", "osint",
    "cryptography", "pki", "ssl", "tls", "oauth", "owasp",
    "incident response", "threat intelligence", "red team", "blue team",
    "security auditing", "risk assessment", "compliance",
    "iso 27001", "nist", "gdpr", "hipaa", "pci dss",
    "active directory", "ldap", "sso", "mfa",

    # ── Networking ────────────────────────────────────────────────────────
    "tcp/ip", "dns", "dhcp", "http", "ftp", "ssh", "routing", "switching",
    "cisco", "juniper", "bgp", "ospf", "vlan", "load balancing",
    "network administration", "wireless networking", "5g", "sd-wan",

    # ── IT Support / Systems ─────────────────────────────────────────────
    "windows server", "active directory", "vmware", "hyper-v",
    "itil", "helpdesk", "ticketing", "jira", "servicenow",
    "system administration", "hardware", "troubleshooting",

    # ── Mobile ────────────────────────────────────────────────────────────
    "android", "ios", "react native", "flutter", "xamarin",
    "mobile development", "app development",

    # ── Finance / Accounting ──────────────────────────────────────────────
    "financial analysis", "financial modelling", "accounting", "auditing",
    "taxation", "bookkeeping", "payroll", "budgeting", "forecasting",
    "accounts payable", "accounts receivable", "tally", "sap",
    "quickbooks", "ms excel", "bloomberg", "risk management",
    "investment banking", "equity research", "portfolio management",
    "valuation", "ifrs", "gaap", "cfa", "ca",

    # ── Marketing / Sales ─────────────────────────────────────────────────
    "digital marketing", "seo", "sem", "social media marketing",
    "content marketing", "email marketing", "ppc", "google ads",
    "facebook ads", "marketing automation", "hubspot", "salesforce",
    "crm", "brand management", "market research", "copywriting",
    "public relations", "media buying", "influencer marketing",

    # ── HR / People ───────────────────────────────────────────────────────
    "recruitment", "talent acquisition", "onboarding", "performance management",
    "employee relations", "hr policies", "payroll management",
    "hris", "workday", "succession planning", "training and development",
    "compensation", "benefits",

    # ── Healthcare / Medical ──────────────────────────────────────────────
    "patient care", "clinical research", "medical coding", "ehr",
    "epic", "cerner", "nursing", "pharmacy", "radiology",
    "laboratory", "anatomy", "physiology", "pharmacology",
    "healthcare management", "telemedicine", "medical billing",

    # ── Legal ─────────────────────────────────────────────────────────────
    "legal research", "contract drafting", "litigation", "compliance",
    "intellectual property", "corporate law", "employment law",
    "legal writing", "due diligence", "arbitration", "mediation",

    # ── Mechanical / Civil / Electrical Engineering ───────────────────────
    "autocad", "solidworks", "catia", "ansys", "matlab simulink",
    "structural analysis", "finite element analysis", "fea",
    "project management", "construction management", "civil engineering",
    "mechanical design", "manufacturing", "quality control", "six sigma",
    "lean manufacturing", "plc", "scada", "embedded systems",
    "circuit design", "pcb design", "vhdl", "fpga",

    # ── Design / Creative ─────────────────────────────────────────────────
    "photoshop", "illustrator", "figma", "sketch", "adobe xd",
    "ui design", "ux design", "graphic design", "video editing",
    "after effects", "premiere pro", "3d modeling", "blender",
    "indesign", "canva",

    # ── Soft Skills ───────────────────────────────────────────────────────
    "communication", "leadership", "management", "teamwork", "problem solving",
    "agile", "scrum", "project management", "critical thinking",
    "time management", "presentation", "negotiation", "decision making",
    "customer service", "stakeholder management",
]

# Pre-compile all patterns once at import time
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (skill, re.compile(r"\b" + re.escape(skill) + r"\b", re.I))
    for skill in SKILLS_VOCAB
]


def extract_skills(text: str) -> list[str]:
    """Return all skills found in text (de-duplicated, sorted)."""
    found = {skill for skill, pat in _PATTERNS if pat.search(text)}
    return sorted(found)


@dataclass
class SkillMatchResult:
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    match_pct: float = 0.0
    total_required: int = 0


def match_typed_skills(resume_text: str, required_skills_csv: str) -> SkillMatchResult:
    required = [s.strip().lower() for s in required_skills_csv.split(",") if s.strip()]
    if not required:
        return SkillMatchResult()
    matched, missing = [], []
    for skill in required:
        pat = re.compile(r"\b" + re.escape(skill) + r"\b", re.I)
        (matched if pat.search(resume_text) else missing).append(skill)
    pct = round(len(matched) / len(required) * 100, 1)
    return SkillMatchResult(matched=matched, missing=missing,
                            match_pct=pct, total_required=len(required))


def skill_gap_analysis(resume_text: str, job_text: str) -> tuple[list[str], list[str]]:
    resume_skills = set(extract_skills(resume_text))
    job_skills    = set(extract_skills(job_text))
    matched       = sorted(resume_skills & job_skills)
    missing       = sorted(job_skills - resume_skills)
    return matched, missing