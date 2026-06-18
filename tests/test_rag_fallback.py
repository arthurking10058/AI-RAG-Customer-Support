from __future__ import annotations

from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from services.exceptions import RetrievalError


def make_doc(text: str, source: str, chunk_id: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "source_name": source, "chunk_id": chunk_id},
    )


def test_rag_summarize_returns_bm25_mode_note(monkeypatch):
    service = RagSummarizeService.__new__(RagSummarizeService)

    monkeypatch.setattr(
        service,
        "retrieve_context",
        lambda query: {
            "mode": "bm25-only",
            "docs": [make_doc("doc body", "a.txt", 1)],
        },
    )
    monkeypatch.setattr(service, "get_sources", lambda docs: ["a.txt#chunk-1"])

    class DummyChain:
        def invoke(self, payload):
            assert payload["input"] == "query"
            return "summary answer"

    service.chain = DummyChain()

    result = service.rag_summarize("query")

    assert "summary answer" in result
    assert "检索模式：BM25 fallback" in result


def test_rag_summarize_returns_friendly_message_when_no_docs(monkeypatch):
    service = RagSummarizeService.__new__(RagSummarizeService)

    monkeypatch.setattr(
        service,
        "retrieve_context",
        lambda query: (_ for _ in ()).throw(
            RetrievalError("No relevant knowledge snippets were found.", code="no_retrieval_docs", status_code=404)
        ),
    )

    result = service.rag_summarize("query")

    assert "未检索到相关知识资料" in result


def test_rag_summarize_raises_when_generation_fails(monkeypatch):
    service = RagSummarizeService.__new__(RagSummarizeService)

    monkeypatch.setattr(
        service,
        "retrieve_context",
        lambda query: {
            "mode": "hybrid",
            "docs": [make_doc("doc body", "a.txt", 1)],
        },
    )
    monkeypatch.setattr(service, "get_sources", lambda docs: ["a.txt#chunk-1"])

    class FailingChain:
        def invoke(self, payload):
            raise RuntimeError("llm offline")

    service.chain = FailingChain()

    try:
        service.rag_summarize("query")
    except RetrievalError as exc:
        assert exc.code == "rag_summarization_failed"
    else:
        raise AssertionError("Expected RetrievalError to be raised.")
