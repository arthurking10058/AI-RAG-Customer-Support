from __future__ import annotations

from copy import deepcopy


DEFAULT_DEMO_CONTEXT = {
    "user_id": "1001",
    "city": "深圳",
    "month": "2025-06",
    "mode": "normal",
    "report": False,
}

_demo_context = deepcopy(DEFAULT_DEMO_CONTEXT)


def get_demo_context() -> dict:
    return deepcopy(_demo_context)


def set_demo_context(**kwargs) -> dict:
    global _demo_context
    next_context = deepcopy(_demo_context)
    for key, value in kwargs.items():
        if value is not None:
            next_context[key] = value
    _demo_context = next_context
    return get_demo_context()


def reset_demo_context() -> dict:
    global _demo_context
    _demo_context = deepcopy(DEFAULT_DEMO_CONTEXT)
    return get_demo_context()
