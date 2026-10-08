"""Text cleaning used BOTH at training time (ml/) and at inference time (backend/).

The classifier was trained on text cleaned like this, so every ticket must be
cleaned the same way before prediction.
"""
import re


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)          # urls
    t = re.sub(r"\{\$[\d,\.]+\}", " ", t)            # masked amounts like {$250.00}
    t = re.sub(r"xx/xx/xxxx|xx/xx/xx", " ", t)       # masked dates
    t = re.sub(r"\bx{2,}\b", " ", t)                 # leftover XXXX masks
    t = re.sub(r"[^a-z0-9\s']", " ", t)              # punctuation and symbols
    t = re.sub(r"\s+", " ", t).strip()
    return t
