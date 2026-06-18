from __future__ import annotations

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: str = Field(..., description="消息角色，支持 user / assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="历史消息")
    user_id: str | None = Field(default=None, description="用户 ID")
    city: str | None = Field(default=None, description="城市")
    month: str | None = Field(default=None, description="月份，格式 YYYY-MM")


class ReportRequest(BaseModel):
    message: str = Field(..., description="报告请求提示词")
    user_id: str = Field(..., description="用户 ID")
    city: str = Field(..., description="城市")
    month: str = Field(..., description="月份，格式 YYYY-MM")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="历史消息")


class ChatResponseData(BaseModel):
    answer: str = Field(..., description="最终回答")
    reasoning_steps: list[str] = Field(default_factory=list, description="思考过程摘要")
    tool_calls: list[str] = Field(default_factory=list, description="工具调用摘要")
