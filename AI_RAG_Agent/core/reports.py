from __future__ import annotations

from typing import Any


def normalize_report_payload(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None

    normalized: dict[str, str] = {}
    for key, value in payload.items():
        normalized[str(key)] = str(value)
    return normalized


def format_report_payload(payload: dict[str, str] | None) -> str:
    if not payload:
        return ""

    lines = []
    for key, value in payload.items():
        lines.append(f"{key}：{value}")
    return "\n".join(lines)
