"""Tests for FastAPI web UI endpoints."""

from fastapi.testclient import TestClient
from src.web import web_app

client = TestClient(web_app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "HardwarePC Agent" in resp.text


def test_chat_endpoint_returns_build(monkeypatch):
    monkeypatch.setattr("src.agent.nodes._get_llm", lambda: None)
    resp = client.post("/api/chat", json={"query": "Gaming PC for 1440p, budget $1200"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Recommended PC Configuration" in data["response"]
    assert data["meta"]["budget_usd"] == 1200
    assert data["meta"]["resolution"] == "1440P"
