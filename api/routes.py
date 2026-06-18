from __future__ import annotations

from fastapi import APIRouter, status

from schemas import (
    ChatRequest,
    ChatResponseData,
    DemoContextData,
    ErrorResponse,
    HealthData,
    ReportRequest,
    SuccessResponse,
)
from services import build_success_response, get_demo_context_snapshot, run_chat, run_report

router = APIRouter(tags=["AI-RAG-Customer-Support"])

COMMON_ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "Request validation inside the service layer failed.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Unexpected internal service error.",
    },
}


@router.get(
    "/health",
    response_model=SuccessResponse[HealthData],
    summary="Health check",
    description="Check whether the FastAPI service is running normally.",
)
def health() -> SuccessResponse[HealthData]:
    return build_success_response("ok", HealthData(status="healthy"))


@router.get(
    "/config/demo-context",
    response_model=SuccessResponse[DemoContextData],
    summary="Get demo context",
    description="Return the default demo context currently used by the Streamlit and API flows.",
)
def demo_context() -> SuccessResponse[DemoContextData]:
    data = DemoContextData(**get_demo_context_snapshot())
    return build_success_response("ok", data)


@router.post(
    "/chat",
    response_model=SuccessResponse[ChatResponseData],
    summary="Chat with Agent + RAG",
    description="Run the normal Q&A flow through the ReactAgent and hybrid retrieval pipeline.",
    responses=COMMON_ERROR_RESPONSES,
)
def chat(request: ChatRequest) -> SuccessResponse[ChatResponseData]:
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
    return build_success_response("chat completed", data)


@router.post(
    "/report",
    response_model=SuccessResponse[ChatResponseData],
    summary="Generate monthly report",
    description="Run the report mode flow and return a structured monthly usage report answer.",
    responses=COMMON_ERROR_RESPONSES,
)
def report(request: ReportRequest) -> SuccessResponse[ChatResponseData]:
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
    return build_success_response("report completed", data)
