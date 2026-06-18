from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import router
from schemas.responses import ErrorResponse
from services.exceptions import ServiceError

app = FastAPI(
    title="AI-RAG-Customer-Support API",
    description="Lightweight FastAPI service for the Agent + Hybrid RAG demo project.",
    version="0.1.0",
)

app.include_router(router)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    payload = ErrorResponse(
        message=exc.message,
        error={"code": exc.code, "details": exc.details},
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        message="Request validation failed.",
        error={"code": "validation_error", "details": [str(item) for item in exc.errors()]},
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def fallback_error_handler(_: Request, __: Exception) -> JSONResponse:
    payload = ErrorResponse(
        message="Internal server error.",
        error={"code": "internal_error", "details": []},
    )
    return JSONResponse(status_code=500, content=payload.model_dump())
