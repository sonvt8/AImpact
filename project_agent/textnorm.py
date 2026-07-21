import re
import unicodedata


def normalize(text) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip().lower()
