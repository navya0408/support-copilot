"""All settings in one place. Values come from environment variables or backend/.env"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent          # .../backend
load_dotenv(BASE / ".env")


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


KB_DIR = Path(os.getenv("KB_DIR", BASE.parent / "knowledge_base"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE / "chroma_db"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "models" / "ticket_classifier.joblib"))
FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", BASE / "feedback.jsonl"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
# Tried in order if the main model is overloaded (503) or rate-limited (429)
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest").split(",") if m.strip()]
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

TOP_K = int(os.getenv("TOP_K", "3"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.60"))
# Chosen on the validation set (ml/confidence_analysis.py): flags ~15% of tickets.
LOW_CLASSIFIER_CONFIDENCE = float(os.getenv("LOW_CLASSIFIER_CONFIDENCE", "0.50"))

# Public-demo protections (0 = off)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
MAX_LLM_CALLS_PER_DAY = int(os.getenv("MAX_LLM_CALLS_PER_DAY", "200"))

# Privacy: by default feedback stores only a short hash and lengths, NOT the ticket or draft text.
STORE_TICKET_TEXT = _flag("STORE_TICKET_TEXT", "false")

# Browsers may only call the API from these pages. Set your real frontend URL when you deploy.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
