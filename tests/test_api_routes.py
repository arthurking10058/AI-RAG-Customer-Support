from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint_returns_success():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "healthy"


def test_demo_context_endpoint_returns_defaults():
    response = client.get("/config/demo-context")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["mode"] in {"normal", "report"}
    assert "user_id" in payload["data"]


def test_chat_endpoint_returns_service_error_payload(monkeypatch):
    def fake_run_chat(**_: object):
        raise __import__("services.exceptions", fromlist=["ServiceError"]).ServiceError(
            "message cannot be empty",
            code="empty_message",
            status_code=400,
        )

    monkeypatch.setattr("api.routes.run_chat", fake_run_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "empty_message"


def test_report_endpoint_returns_success(monkeypatch):
    monkeypatch.setattr(
        "api.routes.run_report",
        lambda **_: {
            "final_answer": "report body",
            "reasoning_steps": ["step-1"],
            "tool_calls": ["tool-a"],
        },
    )

    response = client.post(
        "/report",
        json={
            "message": "generate report",
            "user_id": "1001",
            "city": "深圳",
            "month": "2025-06",
            "history": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["answer"] == "report body"


def test_validation_error_shape():
    response = client.post("/report", json={"message": "generate report"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "validation_error"
