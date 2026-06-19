import importlib

import streamlit as st

from agent import react_agent as react_agent_module
from core.demo_context import DEFAULT_DEMO_CONTEXT, set_demo_context

# Streamlit 重新运行脚本时会复用已导入模块，这里强制重载，避免拿到旧版 ReactAgent
react_agent_module = importlib.reload(react_agent_module)
ReactAgent = react_agent_module.ReactAgent

# 标题
st.set_page_config(page_title="智扫通机器人智能客服", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(18, 29, 51, 0.96), rgba(37, 99, 235, 0.75));
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    }
    .hero-card h1 {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .hero-card p {
        margin: 0.2rem 0 0;
        color: rgba(255, 255, 255, 0.9);
    }
    .soft-panel {
        padding: 0.9rem 1rem;
        border-radius: 0.9rem;
        background: rgba(248, 250, 252, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>智扫通机器人智能客服</h1>
        <p>Agent + 混合检索 RAG + 动态提示词切换</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("面向扫地机器人 / 扫拖一体机器人场景，支持知识问答、外部数据查询和月度报告生成。")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "demo_context" not in st.session_state:
    st.session_state["demo_context"] = DEFAULT_DEMO_CONTEXT.copy()

with st.sidebar:
    st.subheader("演示配置")
    mode = st.radio(
        "当前模式",
        options=["normal", "report"],
        format_func=lambda value: "普通咨询模式" if value == "normal" else "月度报告模式",
        index=0 if st.session_state["demo_context"]["mode"] == "normal" else 1,
    )
    user_id = st.selectbox("用户 ID", options=[str(i) for i in range(1001, 1011)], index=0)
    city = st.selectbox("城市", options=["深圳", "杭州", "合肥"], index=0)
    month = st.selectbox(
        "月份",
        options=[
            "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
            "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
        ],
        index=5,
    )
    st.session_state["demo_context"] = set_demo_context(
        user_id=user_id,
        city=city,
        month=month,
        mode=mode,
        report=mode == "report",
    )

    st.divider()
    st.markdown("**当前上下文**")
    st.code(
        f"user_id={st.session_state['demo_context']['user_id']}\n"
        f"city={st.session_state['demo_context']['city']}\n"
        f"month={st.session_state['demo_context']['month']}\n"
        f"mode={st.session_state['demo_context']['mode']}",
        language="text",
    )

    st.markdown("**项目看点**")
    st.write("• Agent 决定是否调用工具")
    st.write("• RAG 使用 Chroma + BM25 + RRF")
    st.write("• 报告模式输出固定 Markdown 结构")

    st.markdown("**推荐演示问题**")
    if mode == "normal":
        st.code("深圳最近比较潮湿，扫拖一体机器人要怎么保养？", language="text")
        st.code("小户型适合什么类型的扫地机器人？", language="text")
    else:
        st.code("给我生成这个月的使用报告", language="text")
        st.code("结合本月表现，给我一些保养建议", language="text")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="soft-panel">
            <strong>Agent 调度</strong><br>
            负责工具选择、上下文串联和多轮对话流转。
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="soft-panel">
            <strong>混合检索</strong><br>
            结合向量召回、关键词召回与 RRF 融合排序。
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="soft-panel">
            <strong>报告模式</strong><br>
            输出结构化 Markdown，适合演示。
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

for message in st.session_state["messages"]:
    role = message["role"]
    if role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message.get("content", ""))
            reasoning_trace = message.get("reasoning_trace") or []
            if reasoning_trace:
                with st.expander("查看思考过程", expanded=False):
                    for item in reasoning_trace:
                        if item.get("type") == "reasoning":
                            st.markdown(item.get("content", ""))
                        elif item.get("type") == "tool_call":
                            st.markdown(item.get("content", ""))
        continue

    st.chat_message(role).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("智能客服思考中..."):
        agent = ReactAgent()
        if not hasattr(agent, "execute"):
            result = {
                "final_answer": "当前页面加载到了旧版 Agent 实例，请重启 Streamlit 后重试。",
                "reasoning_steps": [],
                "tool_calls": [],
                "reasoning_trace": [],
            }
        else:
            result = agent.execute(
                prompt,
                message_history=st.session_state["messages"][:-1],
                mode=st.session_state["demo_context"]["mode"],
            )

        with st.chat_message("assistant"):
            st.markdown(result["final_answer"])
            if result["reasoning_trace"]:
                with st.expander("查看思考过程", expanded=False):
                    for item in result["reasoning_trace"]:
                        if item["type"] == "reasoning":
                            st.markdown(item["content"])
                        elif item["type"] == "tool_call":
                            st.markdown(item["content"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["final_answer"],
                "reasoning_steps": result["reasoning_steps"],
                "tool_calls": result["tool_calls"],
                "reasoning_trace": result["reasoning_trace"],
            }
        )
        st.rerun()
