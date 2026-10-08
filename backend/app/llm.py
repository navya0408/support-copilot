"""Drafts the reply with Gemini, using ONLY the retrieved passages.

If no API key is set (or every model fails) we fall back to showing the top passage,
so the rest of the system still works.
"""
import logging

from . import messages
from .config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL, LLM_TIMEOUT_SECONDS

log = logging.getLogger("support-copilot")
_client = None


def is_configured() -> bool:
    return bool(GEMINI_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),  # milliseconds
        )
    return _client


def extractive(chunks) -> str:
    """No LLM: show the top passage as it is."""
    top = chunks[0]
    body = top["text"].split("\n", 1)[-1]
    return messages.EXTRACTIVE_FORMAT.format(title=top["title"], body=body[: messages.EXTRACTIVE_MAX_CHARS])


def draft_reply(ticket: str, category: str, chunks):
    """Return (draft_text_or_None, status)."""
    if not is_configured():
        return extractive(chunks), messages.STATUS_EXTRACTIVE_NO_KEY

    context = "\n\n".join(
        f"[{i + 1}] ({c['title']} - {c['section']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = messages.PROMPT.format(context=context, category=category, ticket=ticket[:3000])
    text, last_error = None, None
    models = list(dict.fromkeys([GEMINI_MODEL] + GEMINI_FALLBACK_MODELS))   # unique, in order
    for model_name in models:
        try:
            resp = _get_client().models.generate_content(model=model_name, contents=prompt)
            text = (resp.text or "").strip()
            log.info("Gemini model used: %s", model_name)
            break
        except Exception as exc:  # overloaded (503), quota (429), network, bad model id ...
            last_error = exc
            log.warning("Gemini call failed on %s: %s", model_name, str(exc)[:200])
    else:
        log.warning("All Gemini models failed; using extractive fallback (%s)", type(last_error).__name__)
        return extractive(chunks), messages.STATUS_EXTRACTIVE_ERROR

    if not text or "INSUFFICIENT_CONTEXT" in text:
        return None, messages.STATUS_INSUFFICIENT
    return text, messages.STATUS_LLM
