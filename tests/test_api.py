"""Tests for FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from chains.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_providers():
    r = client.get("/providers")
    assert r.status_code == 200
    assert "ollama" in r.json()["providers"]


def test_analyze_endpoint():
    r = client.post("/analyze", json={
        "traces": [
            {
                "trace_id": "t1",
                "steps": [
                    {"name": "retrieval", "step_type": "retriever",
                     "inputs": {"query": "short"}, "outputs": {"text": "doc doc doc doc"}},
                    {"name": "generation", "step_type": "llm",
                     "inputs": {"text": "doc"}, "outputs": {"text": "answer"}},
                ],
                "quality_score": 0.9,
            },
            {
                "trace_id": "t2",
                "steps": [
                    {"name": "retrieval", "step_type": "retriever",
                     "inputs": {"query": "x" * 300}, "outputs": {"text": "bad"}},
                    {"name": "generation", "step_type": "llm",
                     "inputs": {"text": "bad"}, "outputs": {"text": "wrong"}},
                ],
                "quality_score": 0.2,
            },
        ]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["n_failures"] >= 1
