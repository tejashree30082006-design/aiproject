"""
utils/preprocessing.py
═══════════════════════════════════════════════════════════════════════════════
Text cleaning utilities shared by training and inference.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re


def clean_text(text: str) -> str:
    """
    Normalise raw text from resumes or job descriptions.

    Steps
    -----
    1. Lowercase
    2. Collapse newlines → space
    3. Strip non-alphanumeric (keep hyphens for compound terms)
    4. Collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-z0-9+#\-\. ]", " ", text)   # keep + for C++, # for C#
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def truncate(text: str, max_words: int = 800) -> str:
    """Truncate to at most max_words words (for LLM prompt safety)."""
    words = text.split()
    return " ".join(words[:max_words])
