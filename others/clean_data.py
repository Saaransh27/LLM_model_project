import re
from unidecode import unidecode

def clean_text(text):
    # Remove extra whitespaces (including tabs and newlines)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('\n', '').replace('\r', '')
    text = unidecode(text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text
