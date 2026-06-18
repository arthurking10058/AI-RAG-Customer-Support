from __future__ import annotations

from core.demo_context import get_demo_context, set_demo_context
from schemas.chat import ChatHistoryItem
from services.exceptions import ServiceError


def _history_to_dicts(history: list[ChatHistoryItem]) -> list[dict]:
    return [{"role": item.role, "content": item.content} for item in history]


def _build_agent():
    from agent.react_agent import ReactAgent

    return ReactAgent()


def get_demo_context_snapshot() -> dict:
    return get_demo_context()


def run_chat(
    message: str,
    history: list[ChatHistoryItem] | None = None,
    user_id: str | None = None,
    city: str | None = None,
    month: str | None = None,
) -> dict:
    if not message.strip():
        raise ServiceError("message cannot be empty", code="empty_message", status_code=400)

    set_demo_context(
        user_id=user_id,
        city=city,
        month=month,
        mode="normal",
        report=False,
    )

    agent = _build_agent()
    return agent.execute(
        message,
        message_history=_history_to_dicts(history or []),
        mode="normal",
    )


def run_report(
    message: str,
    user_id: str,
    city: str,
    month: str,
    history: list[ChatHistoryItem] | None = None,
) -> dict:
    if not message.strip():
        raise ServiceError("message cannot be empty", code="empty_message", status_code=400)

    set_demo_context(
        user_id=user_id,
        city=city,
        month=month,
        mode="report",
        report=True,
    )

    agent = _build_agent()
    return agent.execute(
        message,
        message_history=_history_to_dicts(history or []),
        mode="report",
    )
