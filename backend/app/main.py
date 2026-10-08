"""FastAPI backend: classify -> urgency -> retrieve -> draft reply -> (feedback)."""
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import classify, llm, messages, rag, urgency
from .config import (ALLOWED_ORIGINS, FEEDBACK_FILE, GEMINI_API_KEY, LOW_CLASSIFIER_CONFIDENCE,
                     MAX_DISTANCE, MAX_LLM_CALLS_PER_DAY, RATE_LIMIT_PER_MINUTE,
                     STORE_TICKET_TEXT, TOP_K)
from .ratelimit import DailyBudget, RateLimiter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("support-copilot")

limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
llm_budget = DailyBudget(MAX_LLM_CALLS_PER_DAY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = rag.ensure_index()
    log.info("Knowledge base ready: %d chunks", n)
    try:
        classify.get_model()
        log.info("Classifier loaded")
    except RuntimeError as exc:
        log.warning(str(exc))
    yield


app = FastAPI(title="Support Copilot API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


class Ticket(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


class Feedback(BaseModel):
    ticket: str = Field(max_length=5000)
    category: str
    draft: str = Field(default="", max_length=5000)
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=1000)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()      # last hop = the address our own proxy saw
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request):
    if not limiter.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail=messages.RATE_LIMITED, headers={"Retry-After": "60"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "kb_chunks": rag.count(),
        "classifier_loaded": classify.is_loaded(),
        "llm_configured": bool(GEMINI_API_KEY),
    }


@app.post("/analyze")
def analyze(ticket: Ticket, request: Request):
    check_rate_limit(request)
    t0 = time.perf_counter()
    try:
        category, confidence, top3 = classify.predict(ticket.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    level, reasons = urgency.assess(ticket.text)
    chunks = rag.retrieve(ticket.text, TOP_K)
    best_distance = chunks[0]["distance"] if chunks else 1.0

    # Guardrail: if retrieval is not confident, do not let the LLM guess.
    if best_distance > MAX_DISTANCE:
        draft, status = None, messages.STATUS_LOW_RETRIEVAL
    elif llm.is_configured() and not llm_budget.try_consume():
        draft, status = llm.extractive(chunks), messages.STATUS_EXTRACTIVE_BUDGET
    else:
        draft, status = llm.draft_reply(ticket.text, category, chunks)

    refused = status in messages.REFUSED_STATUSES
    return {
        "category": category,
        "category_confidence": confidence,
        "category_low_confidence": confidence < LOW_CLASSIFIER_CONFIDENCE,
        "category_top3": top3,
        "urgency": level,
        "urgency_reasons": reasons,
        "sources": [
            {
                "id": c["id"], "title": c["title"], "section": c["section"],
                "text": c["text"], "distance": c["distance"],
                "similarity": round(1 - c["distance"], 3),
            }
            for c in chunks
        ],
        "draft": None if refused else draft,
        "status": status,
        "message": messages.STATUS_MESSAGES.get(status, ""),
        "needs_human": refused,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


@app.post("/feedback")
def feedback(fb: Feedback, request: Request):
    """Agent thumbs up/down. By default NO ticket or draft text is stored (privacy)."""
    check_rate_limit(request)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": fb.category,
        "rating": fb.rating,
        "comment": fb.comment,
        "ticket_sha256": hashlib.sha256(fb.ticket.encode("utf-8")).hexdigest()[:16],
        "ticket_chars": len(fb.ticket),
        "draft_chars": len(fb.draft),
    }
    if STORE_TICKET_TEXT:
        record["ticket"] = fb.ticket
        record["draft"] = fb.draft
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"saved": True}
