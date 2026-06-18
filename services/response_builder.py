from __future__ import annotations

from schemas.responses import ErrorDetail, ErrorResponse, SuccessResponse


def build_success_response(message: str, data):
    return SuccessResponse(message=message, data=data)


def build_error_response(
    message: str,
    *,
    code: str,
    details: list[str] | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        message=message,
        error=ErrorDetail(code=code, details=details or []),
    )
