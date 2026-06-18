# AI-RAG-Customer-Support

An intelligent customer-support demo project for robot vacuum and vacuum-mop scenarios, built around `Agent + RAG + FastAPI + Streamlit`.

## Project Overview

This project started from a course-style `LangChain Agent + RAG` demo and was gradually refined into a more complete AI backend showcase.

It currently supports:

- knowledge-based Q&A
- monthly report generation
- hybrid retrieval with `Chroma + BM25 + RRF`
- a lightweight `FastAPI` service layer
- a `Streamlit` demo interface
- retrieval comparison and fallback validation

## Core Capabilities

- `Agent` orchestration for tool selection and multi-step reasoning
- `Chroma + BM25 + RRF` hybrid retrieval
- `FastAPI` API layer with unified response models and exception handling
- `Streamlit` demo UI for interactive showcase
- structured report mode output
- `BM25-only fallback` when vector embedding retrieval is unavailable

## Project Screenshots

### Home Overview

![Home Overview](assets/01-home-overview.png)

### Normal RAG Answer

![Normal RAG Answer](assets/02-normal-rag-answer.png)

### Report Mode

![Report Mode](assets/03-report-mode-answer.png)

### Core Flow

![Core Flow](assets/04-architecture-flow.png)

## Repository Structure

```text
AI-RAG-Customer-Support
├── app.py
├── main.py
├── agent/
├── api/
├── assets/
├── config/
├── core/
├── data/
├── model/
├── prompts/
├── rag/
├── schemas/
├── services/
├── tests/
├── utils/
└── requirements.txt
```

## Core Flows

### Normal Knowledge Q&A

```text
User Input
  -> Streamlit / FastAPI
  -> ReactAgent
  -> Tool selection
  -> Hybrid RAG retrieval
  -> LLM summarization
  -> Final answer
```

### Monthly Report

```text
User Input
  -> Streamlit / FastAPI
  -> ReactAgent
  -> Report-mode prompt switch
  -> External usage record lookup
  -> Structured report generation
  -> Final answer
```

### Retrieval Fallback

```text
Query
  -> Vector retrieval
  -> if vector retrieval unavailable:
       fallback to BM25-only retrieval
  -> return retrievable context instead of failing hard
```

## FastAPI Service

The project now includes a lightweight FastAPI layer as the second-stage backend upgrade.

Current endpoints:

- `GET /health`
- `GET /config/demo-context`
- `POST /chat`
- `POST /report`

Current backend improvements:

- clearer request/response schemas
- unified success/error response structure
- centralized exception handling
- route-level test coverage
- service-layer validation for common bad inputs

After startup you can access:

- Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Retrieval Comparison And Fallback

To validate retrieval behavior, the project includes a lightweight comparison script:

```powershell
python -m rag.compare_retrieval
```

What it does:

- compares `vector-only` and `hybrid` retrieval behavior
- records structured output to `outputs/retrieval_comparison.json`
- automatically falls back to `BM25-only` when vector embedding service is unavailable

Why this matters:

- the project is not only implementing retrieval strategies
- it is also starting to handle real engineering constraints such as dependency failure and degraded environments

In the current terminal environment, the comparison result shows that:

- all 6 evaluation queries completed
- all 6 were handled via `BM25-only fallback`
- no request crashed completely

This means the current result mainly proves the usefulness of the fallback path, rather than proving that hybrid retrieval is already superior to vector-only retrieval in every environment.

## How To Run

### 1. Build The Knowledge Base

```powershell
python -m rag.vector_store
```

### 2. Start The Streamlit Demo

```powershell
python -m streamlit run app.py
```

### 3. Start The FastAPI Service

```powershell
uvicorn main:app --reload
```

### 4. Run Tests

```powershell
pytest tests -q
```

## Recommended Demo Questions

### Normal Q&A

- `深圳最近比较潮湿，扫拖一体机器人怎么做维护保养？`
- `小户型更适合什么类型的扫地机器人？`
- `扫地机器人出现迷路或者反复打转时怎么排查？`

### Report Mode

- `给我生成这个月的使用报告`
- `结合本月表现，给我一些保养建议`

## Resume-Friendly Highlights

- Built an intelligent customer-support project based on `LangChain + Agent + RAG`, supporting knowledge Q&A, external usage-record lookup, prompt switching, and structured monthly report generation.
- Implemented lightweight hybrid retrieval with `Chroma + BM25 + RRF`, balancing semantic recall and keyword matching.
- Added a lightweight `FastAPI` backend layer with request/response schemas, unified error handling, and basic automated tests.
- Added a retrieval comparison script and `BM25-only fallback`, improving system robustness when vector embedding service is unavailable.
- Kept `Streamlit` as the demo entry while exposing backend capabilities through API routes for a more complete AI service showcase.

## Current Progress

Already completed:

- Streamlit demo stabilization
- GitHub showcase cleanup
- lightweight FastAPI second-stage backendization
- response models and exception handling
- retrieval comparison script
- BM25 fallback for restricted environments
- basic test skeleton and route/service coverage

## Next Steps

- improve README/API usage examples further
- add more fixed evaluation queries and manual comparison notes
- strengthen RAG quality validation
- extend backend testing and failure-case coverage
