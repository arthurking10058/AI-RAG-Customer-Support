from __future__ import annotations

from fastapi import FastAPI

from api import router

app = FastAPI(
    title="AI-RAG-Customer-Support API",
    description="Lightweight FastAPI service for the Agent + Hybrid RAG demo project.",
    version="0.1.0",
)

app.include_router(router)

