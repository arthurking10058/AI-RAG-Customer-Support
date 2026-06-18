from __future__ import annotations

import pytest

from schemas.chat import ChatHistoryItem
from services.chat_service import get_demo_context_snapshot, run_chat, run_report
from services.exceptions import ServiceError


def test_run_chat_rejects_blank_message():
    with pytest.raises(ServiceError) as exc_info:
        run_chat("   ")

    assert exc_info.value.code == "empty_message"


def test_run_report_rejects_blank_message():
    with pytest.raises(ServiceError) as exc_info:
        run_report("   ", user_id="1001", city="深圳", month="2025-06")

    assert exc_info.value.code == "empty_message"


def test_run_chat_passes_history_and_mode(monkeypatch):
    captured: dict[str, object] = {}

    class DummyAgent:
        def execute(self, message, message_history=None, mode="normal"):
            captured["message"] = message
            captured["history"] = message_history
            captured["mode"] = mode
            return {
                "final_answer": "ok",
                "reasoning_steps": [],
                "tool_calls": [],
            }

    monkeypatch.setattr("services.chat_service._build_agent", lambda: DummyAgent())

    result = run_chat(
        "hello",
        history=[ChatHistoryItem(role="user", content="before")],
        user_id="1001",
        city="深圳",
        month="2025-06",
    )

    assert result["final_answer"] == "ok"
    assert captured["mode"] == "normal"
    assert captured["history"] == [{"role": "user", "content": "before"}]


def test_run_report_updates_demo_context_and_uses_report_mode(monkeypatch):
    captured: dict[str, object] = {}

    class DummyAgent:
        def execute(self, message, message_history=None, mode="normal"):
            captured["mode"] = mode
            return {
                "final_answer": "ok",
                "reasoning_steps": [],
                "tool_calls": [],
            }

    monkeypatch.setattr("services.chat_service._build_agent", lambda: DummyAgent())

    run_report("generate", user_id="1002", city="杭州", month="2025-07")

    snapshot = get_demo_context_snapshot()
    assert captured["mode"] == "report"
    assert snapshot["user_id"] == "1002"
    assert snapshot["city"] == "杭州"
    assert snapshot["month"] == "2025-07"
