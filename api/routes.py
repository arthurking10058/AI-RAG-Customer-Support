from __future__ import annotations

from fastapi import APIRouter

from schemas import ApiResponse, ChatRequest, ChatResponseData, ReportRequest
from services import (
    ServiceError,
    error_response,
    get_demo_context_snapshot,
    run_chat,
    run_report,
    success_response,
)

router = APIRouter(tags=["AI-RAG-Customer-Support"])


@router.get("/health", response_model=ApiResponse, summary="健康检查")
def health() -> ApiResponse:
    return success_response("ok", {"status": "healthy"})


@router.get("/config/demo-context", response_model=ApiResponse, summary="获取默认演示上下文")
def demo_context() -> ApiResponse:
    return success_response("ok", get_demo_context_snapshot())


@router.post(
    "/chat",
    response_model=ApiResponse,
    summary="普通问答接口",
    description="调用 Agent + 混合检索 RAG 完成普通咨询问答。",
)
def chat(request: ChatRequest) -> ApiResponse:
    try:
        result = run_chat(
            message=request.message,
            history=request.history,
            user_id=request.user_id,
            city=request.city,
            month=request.month,
        )
        data = ChatResponseData(
            answer=result["final_answer"],
            reasoning_steps=result["reasoning_steps"],
            tool_calls=result["tool_calls"],
        )
        return success_response("chat completed", data.model_dump())
    except ServiceError as exc:
        return error_response(exc.message)
    except Exception:
        return error_response("普通问答执行失败，请稍后重试。")


@router.post(
    "/report",
    response_model=ApiResponse,
    summary="报告模式接口",
    description="生成结构化月度使用报告。",
)
def report(request: ReportRequest) -> ApiResponse:
    try:
        result = run_report(
            message=request.message,
            user_id=request.user_id,
            city=request.city,
            month=request.month,
            history=request.history,
        )
        data = ChatResponseData(
            answer=result["final_answer"],
            reasoning_steps=result["reasoning_steps"],
            tool_calls=result["tool_calls"],
        )
        return success_response("report completed", data.model_dump())
    except ServiceError as exc:
        return error_response(exc.message)
    except Exception:
        return error_response("报告生成失败，请稍后重试。")
