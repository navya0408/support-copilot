"""API behaviour with the slow parts (model, vector search, Gemini) replaced by fakes.

Needs fastapi + httpx (pip install -r requirements-dev.txt); skipped otherwise.
"""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import classify, llm, main, messages, rag  # noqa: E402
from app.ratelimit import DailyBudget, RateLimiter  # noqa: E402

CHUNK = {"id": "x#0", "source": "x", "title": "Billing errors", "section": "What to do",
         "text": "Billing errors - What to do\nSend a written notice.", "distance": 0.30}
TOP3 = [{"label": "CREDIT_CARD", "probability": 0.9}]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "predict", lambda text: ("CREDIT_CARD", 0.9, TOP3))
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(CHUNK)])
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "draft_reply", lambda t, c, ch: ("Please send a written notice [1].", messages.STATUS_LLM))
    monkeypatch.setattr(main, "limiter", RateLimiter(100))
    monkeypatch.setattr(main, "llm_budget", DailyBudget(100))
    monkeypatch.setattr(main, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    return TestClient(main.app)          # no `with`: skips startup (no index build)


def test_answers_when_retrieval_is_confident(client):
    r = client.post("/analyze", json={"text": "I was charged twice on my card"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "llm" and body["draft"] and not body["needs_human"]
    assert body["urgency"] == "medium"


def test_refuses_when_retrieval_is_far(client, monkeypatch):
    far = dict(CHUNK, distance=0.95)
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [far])
    body = client.post("/analyze", json={"text": "how is the food today"}).json()
    assert body["status"] == messages.STATUS_LOW_RETRIEVAL
    assert body["needs_human"] and body["draft"] is None and body["message"]


def test_llm_insufficient_context_escalates(client, monkeypatch):
    monkeypatch.setattr(llm, "draft_reply", lambda t, c, ch: (None, messages.STATUS_INSUFFICIENT))
    body = client.post("/analyze", json={"text": "some unusual ticket text"}).json()
    assert body["needs_human"] and body["draft"] is None


def test_daily_budget_falls_back_to_top_passage(client, monkeypatch):
    monkeypatch.setattr(main, "llm_budget", DailyBudget(1))
    client.post("/analyze", json={"text": "first ticket here"})
    body = client.post("/analyze", json={"text": "second ticket here"}).json()
    assert body["status"] == messages.STATUS_EXTRACTIVE_BUDGET
    assert body["draft"].startswith("[1]")


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(2))
    codes = [client.post("/analyze", json={"text": "ticket number one"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_too_short_ticket_is_rejected(client):
    assert client.post("/analyze", json={"text": "hi"}).status_code == 422


def test_low_confidence_flag(client, monkeypatch):
    monkeypatch.setattr(classify, "predict", lambda text: ("CREDIT_CARD", 0.45, TOP3))
    body = client.post("/analyze", json={"text": "unclear ticket text"}).json()
    assert body["category_low_confidence"] is True


def test_feedback_does_not_store_ticket_text_by_default(client, tmp_path):
    secret = "my SSN is 123-45-6789"
    r = client.post("/feedback", json={"ticket": secret, "category": "CREDIT_CARD",
                                       "draft": "a draft", "rating": "up"})
    assert r.status_code == 200
    saved = (tmp_path / "feedback.jsonl").read_text()
    assert "123-45-6789" not in saved and "a draft" not in saved
    record = json.loads(saved.splitlines()[0])
    assert record["ticket_chars"] == len(secret) and record["rating"] == "up"
