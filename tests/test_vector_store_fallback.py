from __future__ import annotations

from langchain_core.documents import Document

from rag.vector_store import VectorStoreService
from services.exceptions import RetrievalError


def make_doc(text: str, source: str, chunk_id: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "source_name": source, "chunk_id": chunk_id},
    )


def test_safe_hybrid_search_falls_back_to_bm25(monkeypatch):
    service = VectorStoreService.__new__(VectorStoreService)

    monkeypatch.setattr(
        service,
        "vector_search",
        lambda query: (_ for _ in ()).throw(
            RetrievalError("Vector retrieval is unavailable.", code="vector_retrieval_unavailable")
        ),
    )
    monkeypatch.setattr(
        service,
        "bm25_search",
        lambda query: [make_doc("bm25 doc", "a.txt", 1)],
    )

    result = service.safe_hybrid_search("query")

    assert result["mode"] == "bm25-only"
    assert len(result["docs"]) == 1
    assert result["vector_error"] == "Vector retrieval is unavailable."


def test_safe_hybrid_search_returns_hybrid_when_vector_available(monkeypatch):
    service = VectorStoreService.__new__(VectorStoreService)
    vector_docs = [make_doc("vector doc", "v.txt", 1)]
    bm25_docs = [make_doc("bm25 doc", "b.txt", 2)]

    monkeypatch.setattr(service, "vector_search", lambda query: vector_docs)
    monkeypatch.setattr(service, "bm25_search", lambda query: bm25_docs)

    result = service.safe_hybrid_search("query")

    assert result["mode"] == "hybrid"
    assert len(result["docs"]) >= 1
