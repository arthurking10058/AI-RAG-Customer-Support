from __future__ import annotations

from schemas.responses import ApiResponse


def success_response(message: str, data=None) -> ApiResponse:
    return ApiResponse(success=True, message=message, data=data)


def error_response(message: str, data=None) -> ApiResponse:
    return ApiResponse(success=False, message=message, data=data)

