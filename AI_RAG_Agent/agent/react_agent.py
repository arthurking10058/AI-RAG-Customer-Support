from langchain.agents import create_agent

from core.demo_context import get_demo_context
from model.factory import chat_model
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, message_history: list[dict] | None = None, mode: str = "normal"):
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

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context=context):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
