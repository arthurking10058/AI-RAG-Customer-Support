from __future__ import annotations

from langchain_core.documents import Document

from rag.compare_retrieval import compare_query, docs_to_rows, summarize


def make_doc(text: str, source: str, chunk_id: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "source_name": source, "chunk_id": chunk_id},
    )


def test_docs_to_rows_formats_rank_and_source():
    rows = docs_to_rows([make_doc("content", "a.txt", 1)])

    assert rows[0]["rank"] == 1
    assert rows[0]["source"] == "a.txt#chunk-1"


def test_compare_query_marks_hybrid_when_new_docs_are_introduced():
    class DummyVectorStore:
        def vector_search(self, query: str):
            assert query == "test query"
            return [make_doc("vector only", "v.txt", 1)]

        def hybrid_search(self, query: str):
            assert query == "test query"
            return {
                "bm25_docs": [make_doc("bm25 result", "b.txt", 2)],
                "docs": [
                    make_doc("vector only", "v.txt", 1),
                    make_doc("hybrid new", "h.txt", 3),
                ],
            }

    result = compare_query(DummyVectorStore(), "test query")

    assert result["decision"] == "hybrid"
    assert len(result["hybrid"]) == 2


def test_compare_query_marks_bm25_only_when_vector_returns_no_hits():
    class DummyVectorStore:
        def vector_search(self, query: str):
            assert query == "test query"
            return []

        def hybrid_search(self, query: str):
            assert query == "test query"
            docs = [make_doc("bm25 result", "b.txt", 2)]
            return {
                "bm25_docs": docs,
                "docs": docs,
            }

    result = compare_query(DummyVectorStore(), "test query")

    assert result["decision"] == "bm25_only"
    assert result["status"] == "success"


def test_summarize_prefers_hybrid_when_it_wins_most_queries():
    summary = summarize(
        [
            {"decision": "hybrid", "mode": "full", "status": "success"},
            {"decision": "hybrid", "mode": "full", "status": "success"},
            {"decision": "tie", "mode": "full", "status": "success"},
        ]
    )

    assert summary["total_queries"] == 3
    assert summary["recommended_retrieval"] == "hybrid"


def test_summarize_prefers_bm25_only_when_all_results_are_fallback():
    summary = summarize(
        [
            {"decision": "bm25_only", "mode": "bm25-only", "status": "fallback"},
            {"decision": "bm25_only", "mode": "bm25-only", "status": "fallback"},
        ]
    )

    assert summary["fallback_queries"] == 2
    assert summary["recommended_retrieval"] == "bm25_only"
