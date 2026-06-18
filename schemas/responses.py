from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code.")
    details: list[str] = Field(default_factory=list, description="Optional human-readable detail messages.")


class SuccessResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True, description="Whether the request succeeded.")
    message: str = Field(..., description="High-level response message.")
    data: T = Field(..., description="Structured response payload.")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="Whether the request succeeded.")
    message: str = Field(..., description="High-level error message.")
    error: ErrorDetail = Field(..., description="Structured error metadata.")


class HealthData(BaseModel):
    status: str = Field(..., description="Current service health status.")


class DemoContextData(BaseModel):
    user_id: str = Field(..., description="Default demo user identifier.")
    city: str = Field(..., description="Default demo city.")
    month: str = Field(..., description="Default demo month in YYYY-MM format.")
    mode: str = Field(..., description="Current demo mode.")
    report: bool = Field(..., description="Whether report mode is enabled.")
