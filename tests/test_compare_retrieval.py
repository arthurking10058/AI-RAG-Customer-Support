from __future__ import annotations

import json

from langchain_core.documents import Document

from rag.compare_retrieval import compare_query, docs_to_rows, load_queries, normalize_query_entry, summarize


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

    result = compare_query(
        DummyVectorStore(),
        {
            "id": "buying_01",
            "query": "test query",
            "category": "选购建议",
            "subcategory": "buying_subtype",
            "expected_focus": "focus",
            "manual_checkpoints": ["checkpoint"],
        },
    )

    assert result["decision"] == "hybrid"
    assert len(result["hybrid"]) == 2
    assert result["category"] == "选购建议"
    assert result["subcategory"] == "buying_subtype"
    assert result["manual_review"]["checkpoints"][0]["item"] == "checkpoint"


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
    assert result["category"] == "未分类"


def test_normalize_query_entry_supports_plain_string():
    entry = normalize_query_entry("hello", 2)

    assert entry["id"] == "query_02"
    assert entry["query"] == "hello"
    assert entry["category"] == "未分类"
    assert entry["subcategory"] == ""


def test_load_queries_supports_rich_json_entries(tmp_path):
    query_file = tmp_path / "queries.json"
    query_file.write_text(
        json.dumps(
            [
                {
                    "id": "maintenance_01",
                    "query": "query text",
                    "category": "维护保养",
                    "subcategory": "maintenance_subtype",
                    "expected_focus": "focus",
                    "manual_checkpoints": ["a", "b"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries = load_queries(str(query_file))

    assert entries[0]["id"] == "maintenance_01"
    assert entries[0]["subcategory"] == "maintenance_subtype"
    assert entries[0]["manual_checkpoints"] == ["a", "b"]


def test_summarize_prefers_hybrid_when_it_wins_most_queries():
    summary = summarize(
        [
            {"decision": "hybrid", "mode": "full", "status": "success", "category": "选购建议", "subcategory": "buying_01", "manual_review": {"completed": True}},
            {"decision": "hybrid", "mode": "full", "status": "success", "category": "选购建议", "subcategory": "buying_01", "manual_review": {"completed": False}},
            {"decision": "tie", "mode": "full", "status": "success", "category": "维护保养", "subcategory": "maintenance_01", "manual_review": {"completed": False}},
        ]
    )

    assert summary["total_queries"] == 3
    assert summary["recommended_retrieval"] == "hybrid"
    assert summary["manual_review_completed"] == 1
    assert summary["categories"]["选购建议"]["total_queries"] == 2
    assert summary["subcategories"]["buying_01"]["total_queries"] == 2


def test_summarize_prefers_bm25_only_when_all_results_are_fallback():
    summary = summarize(
        [
            {"decision": "bm25_only", "mode": "bm25-only", "status": "fallback", "category": "环境适配", "subcategory": "environment_01", "manual_review": {"completed": False}},
            {"decision": "bm25_only", "mode": "bm25-only", "status": "fallback", "category": "环境适配", "subcategory": "environment_01", "manual_review": {"completed": False}},
        ]
    )

    assert summary["fallback_queries"] == 2
    assert summary["recommended_retrieval"] == "bm25_only"
