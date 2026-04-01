"""
utils/file_extractor.py
═══════════════════════════════════════════════════════════════════════════════
Resume file extraction — PDF and DOCX.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging

log = logging.getLogger(__name__)


def extract_pdf(file) -> str:
    """Extract text from a PDF file object."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        log.error("PDF extraction failed: %s", e)
        return ""


def extract_docx(file) -> str:
    """Extract text from a DOCX file object."""
    try:
        import docx
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log.error("DOCX extraction failed: %s", e)
        return ""


def extract_resume(uploaded_file) -> str:
    """
    Auto-detect format and extract text.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile object

    Returns
    -------
    Extracted text string (may be empty on error).
    """
    if uploaded_file.type == "application/pdf":
        return extract_pdf(uploaded_file)
    return extract_docx(uploaded_file)
