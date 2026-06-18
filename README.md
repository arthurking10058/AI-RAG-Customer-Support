# AI-RAG-Customer-Support

一个面向扫地机器人 / 扫拖一体机器人场景的智能客服项目仓库。

## 项目简介

基于 `LangChain Agent + 混合检索 RAG + Streamlit` 构建，支持知识问答、外部使用记录查询、动态提示词切换和结构化月度报告生成。

## 核心能力

- `Agent` 工具调度
- `Chroma + BM25 + RRF` 混合检索
- `Streamlit` 演示页面
- 结构化月度报告生成

## 项目截图

### 首页总览

![首页总览](assets/01-home-overview.png)

### 普通知识问答

![普通知识问答](assets/02-normal-rag-answer.png)

### 月度报告模式

![月度报告模式](assets/03-report-mode-answer.png)

### 核心调用链

![核心调用链](assets/04-architecture-flow.png)

## 仓库结构

```text
AI-RAG-Customer-Support
├─ app.py
├─ agent/
├─ assets/
├─ config/
├─ core/
├─ data/
├─ model/
├─ prompts/
├─ rag/
├─ utils/
└─ requirements.txt
```

## 核心链路

### 普通知识问答

```text
用户输入 -> Streamlit -> ReactAgent -> Agent 判断是否调用工具 -> 混合检索 RAG -> 模型总结 -> 页面展示
```

### 月度报告

```text
用户输入 -> Streamlit -> ReactAgent -> 切换报告 Prompt -> 查询外部记录 -> 生成 Markdown 报告 -> 页面展示
```

## 运行方式

### 1. 构建知识库

```powershell
python -m rag.vector_store
```

### 2. 启动 Streamlit 演示页

```powershell
python -m streamlit run app.py
```

### 3. 启动 FastAPI 服务

```powershell
uvicorn main:app --reload
```

启动后可访问：

- Swagger 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## 推荐演示问题

### 普通咨询

- `深圳最近比较潮湿，扫拖一体机器人要怎么保养？`
- `小户型适合什么类型的扫地机器人？`

### 月度报告

- `给我生成这个月的使用报告`
- `结合本月表现，给我一些保养建议`

## 简历亮点

- 基于 `LangChain + Agent + RAG` 构建扫地机器人智能客服系统，支持知识问答、外部数据查询、动态提示词切换和结构化月度报告生成。
- 基于 `Chroma + BM25 + RRF` 实现轻量混合检索，兼顾语义召回与关键词匹配，提升问答稳定性。
- 使用 `Streamlit` 搭建可复现演示页面，支持多轮对话、固定上下文和报告模式切换，便于项目展示和功能验证。

## 后续可选

- 继续完善 FastAPI 接口层
- 补充基础测试
- 增强会话存储和历史管理
- 继续完善演示视频与项目讲解稿
