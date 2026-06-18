from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from core.demo_context import get_demo_context
from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute(self, query: str, message_history: list[dict] | None = None, mode: str = "normal") -> dict[str, Any]:
        history = message_history or []
        history_messages = []
        for message in history[-6:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                history_messages.append({"role": role, "content": content})

        history_messages.append({"role": "user", "content": query})
        input_dict = {"messages": history_messages}

        context = get_demo_context()
        context["report"] = mode == "report"
        context["mode"] = mode

        result = {
            "final_answer": "",
            "reasoning_steps": [],
            "tool_calls": [],
            "reasoning_trace": [],
            "error": None,
        }
        seen_messages: set[tuple] = set()

        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context=context):
                latest_message = chunk["messages"][-1]
                message_key = self._build_message_key(latest_message)
                if message_key in seen_messages:
                    continue

                seen_messages.add(message_key)
                self._collect_message(latest_message, result)
        except Exception as exc:
            logger.exception("[ReactAgent.execute] 本轮对话执行失败: %s", exc)
            result["error"] = "本轮检索或生成失败，请重试。"
            if not result["final_answer"]:
                result["final_answer"] = result["error"]

        if not result["final_answer"]:
            result["final_answer"] = "本轮未生成有效回答，请换一种问法后重试。"

        return result

    def _collect_message(self, message: BaseMessage, result: dict[str, Any]) -> None:
        if not isinstance(message, AIMessage):
            return

        content = self._normalize_content(message.content)
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            if content and content not in result["reasoning_steps"]:
                result["reasoning_steps"].append(content)
                result["reasoning_trace"].append({"type": "reasoning", "content": content})

            for tool_call in tool_calls:
                summary = self._format_tool_call(tool_call)
                if summary not in result["tool_calls"]:
                    result["tool_calls"].append(summary)
                    result["reasoning_trace"].append({"type": "tool_call", "content": summary})
            return

        if content:
            result["final_answer"] = content

    def _build_message_key(self, message: BaseMessage) -> tuple:
        message_id = getattr(message, "id", None)
        if message_id:
            return ("id", message_id)

        content = self._normalize_content(getattr(message, "content", ""))
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_signature = tuple(
            (tool_call.get("id"), tool_call.get("name"), str(tool_call.get("args")))
            for tool_call in tool_calls
        )
        return (type(message).__name__, content, tool_signature)

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue

                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    text = str(text).strip()
                    if text:
                        parts.append(text)

            return "\n".join(parts).strip()

        if content is None:
            return ""

        return str(content).strip()

    def _format_tool_call(self, tool_call: dict[str, Any]) -> str:
        tool_name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args") or {}
        if not args:
            return f"调用工具：{tool_name}()"

        arg_text = ", ".join(f"{key}={value}" for key, value in args.items())
        return f"调用工具：{tool_name}({arg_text})"


if __name__ == "__main__":
    agent = ReactAgent()
    result = agent.execute("给我生成我的使用报告", mode="report")
    print(result["final_answer"])
