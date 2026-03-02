import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)   # remove special characters
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces
    return text.strip()