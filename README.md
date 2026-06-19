# AI-RAG-Customer-Support

An intelligent customer-support demo project for robot vacuum and vacuum-mop scenarios, built around `Agent + RAG + FastAPI + Streamlit`.

## Project Overview

This project started from a course-style `LangChain Agent + RAG` demo and was gradually refined into a more complete AI backend showcase.

The current version is mainly centered around four engineering points:

- a lightweight `FastAPI` backend layer
- a fixed-query retrieval evaluation set
- `BM25-only fallback` in restricted environments
- clearer failure-path handling and automated tests

It currently supports:

- knowledge-based Q&A
- monthly report generation
- hybrid retrieval with `Chroma + BM25 + RRF`
- a lightweight `FastAPI` service layer
- a `Streamlit` demo interface
- fixed-query retrieval evaluation
- retrieval fallback validation

## Core Capabilities

- `Agent` orchestration for tool selection and multi-step reasoning
- `FastAPI` API layer with unified response models and exception handling
- fixed-query retrieval evaluation samples with manual review checkpoints
- `BM25-only fallback` when vector embedding retrieval is unavailable
- route-level failure-path handling and automated validation coverage
- `Chroma + BM25 + RRF` hybrid retrieval
- structured report mode output
- `Streamlit` demo UI for interactive showcase

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
python -m rag.compare_retrieval --query-file data/evaluation_queries.json
```

What it does:

- compares `vector-only` and `hybrid` retrieval behavior
- records structured output to `outputs/retrieval_comparison.json`
- automatically falls back to `BM25-only` when vector embedding service is unavailable
- supports categorized evaluation samples with manual review placeholders

Why this matters:

- the project is not only implementing retrieval strategies
- it is also starting to handle real engineering constraints such as dependency failure and degraded environments
- it now has a more reusable evaluation set instead of only a few ad-hoc test queries

In the current terminal environment, the comparison result shows that:

- all 15 evaluation queries completed
- all 15 were handled via `BM25-only fallback`
- no request crashed completely

This means the current result mainly proves the usefulness of the fallback path, rather than proving that hybrid retrieval is already superior to vector-only retrieval in every environment.

Current evaluation categories:

- `维护保养`
- `故障排查`
- `选购建议`
- `环境适配`
- `报告型问题`

The sample set lives in:

- `data/evaluation_queries.json`

Each entry now carries:

- query category
- expected answer focus
- manual review checkpoints

This makes it easier to move the retrieval comparison from "there is a script" to "there is a reusable evaluation set with clearer review structure".

What the latest run tells us:

- the fallback path is now verified by a real 15-query run
- the current result is useful for engineering validation
- most query categories can now return to the correct knowledge domain
- a few weak samples still have non-optimal ranking
- it is not yet the final evidence for hybrid quality superiority
- some BM25-only hits still repeatedly land on `选购指南.txt` chunks, so retrieval quality refinement is still needed

The current retrieval conclusion is therefore intentionally conservative:

- most query categories already return to the correct knowledge domain
- the remaining gap is mainly ranking quality on a few weak samples
- it is no longer worth spending disproportionate time on single-query tuning unless a stronger reranker, classifier, or rule layer is introduced

## API Examples

### `POST /chat`

Request example:

```json
{
  "message": "深圳最近比较潮湿，扫拖一体机器人怎么做维护保养？",
  "history": [
    {
      "role": "user",
      "content": "我家最近比较潮。"
    }
  ],
  "user_id": "1001",
  "city": "深圳",
  "month": "2025-06"
}
```

Success response example:

```json
{
  "success": true,
  "message": "chat completed",
  "data": {
    "answer": "建议重点清洁拖布、水箱和尘盒，并保持机器通风干燥。",
    "reasoning_steps": [
      "识别用户问题属于维护保养场景",
      "调用 RAG 检索相关维护知识"
    ],
    "tool_calls": [
      "rag_search"
    ]
  }
}
```

### `POST /report`

Request example:

```json
{
  "message": "结合本月表现，给我一些保养建议",
  "user_id": "1001",
  "city": "深圳",
  "month": "2025-06",
  "history": []
}
```

Typical error response example:

```json
{
  "success": false,
  "message": "Vector retrieval is unavailable.",
  "error": {
    "code": "vector_retrieval_unavailable",
    "details": [
      "embedding service offline"
    ]
  }
}
```

## Stability And Error Handling

The project now includes a more explicit stability layer for both the API and RAG service:

- unified `ServiceError` / `RetrievalError` handling in FastAPI
- `503` responses for retrieval-side failures
- `500` responses for unexpected internal errors
- friendly fallback messages when retrieval context is unavailable
- automatic `BM25-only` degradation when vector retrieval cannot be used

Current automated coverage includes:

- route-level validation and exception response tests
- RAG fallback behavior tests
- retrieval degradation tests for vector failure scenarios

At the moment, the test suite passes with:

- `36 passed`

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

If pytest hits temp-directory permission issues on Windows, run:

```powershell
pytest tests -q --basetemp .pytest_tmp
```

## Recommended Demo Questions

### Normal Q&A

- `深圳最近比较潮湿，扫拖一体机器人怎么做维护保养？`
- `小户型更适合什么类型的扫地机器人？`
- `扫地机器人出现迷路或者反复打转时怎么排查？`

### Report Mode

- `给我生成这个月的使用报告`
- `结合本月表现，给我一些保养建议`

## Implementation Notes

- The project combines `LangChain Agent`, hybrid retrieval, `FastAPI`, and `Streamlit` in a single customer-support demo workflow.
- The retrieval layer uses `Chroma + BM25 + RRF`, and the evaluation layer adds a fixed-query sample set with manual review checkpoints.
- The backend layer includes unified response models, explicit failure-path handling, and `BM25-only fallback` when vector retrieval is unavailable.
- `Streamlit` remains the interactive demo entry, while `FastAPI` exposes the same capabilities through documented API routes.

## Troubleshooting

### Pytest temp-directory issue on Windows

If `pytest` hits a temp-directory permission error on Windows, run:

```powershell
pytest tests -q --basetemp .pytest_tmp
```

### Chinese response text looks garbled in Windows PowerShell 5.1

If API responses look garbled in Windows PowerShell 5.1, verify the service is returning `application/json; charset=utf-8`, or use Swagger / Python / Postman for response validation instead of relying on legacy console decoding.

## Next Steps

- improve retrieval evaluation coverage
- add stronger reranking / classification experiments when needed
- improve deployment and environment documentation
- reopen retrieval optimization only if stronger reranking/classification capability or a stable vector environment becomes available
