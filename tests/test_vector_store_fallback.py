from __future__ import annotations

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vector_store import (
    VectorStoreService,
    build_bm25_text,
    classify_query_type,
    classify_troubleshooting_subtype,
    get_source_weight,
    rewrite_query_for_retrieval,
    split_txt_document_by_structure,
    tokenize_for_bm25,
)
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
        lambda query, **kwargs: [make_doc("bm25 doc", "a.txt", 1)],
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
    monkeypatch.setattr(service, "bm25_search", lambda query, **kwargs: bm25_docs)

    result = service.safe_hybrid_search("query")

    assert result["mode"] == "hybrid"
    assert len(result["docs"]) >= 1


def test_tokenize_for_bm25_handles_chinese_text():
    tokens = tokenize_for_bm25("拖布异味和水箱清洁一般怎么处理？")

    assert "拖布" in tokens
    assert "水箱" in tokens
    assert "清洁" in tokens


def test_build_bm25_text_includes_metadata_context():
    doc = Document(
        page_content="正文内容",
        metadata={"source_name": "维护保养.txt", "section_title": "环境适配维护", "item_number": 9},
    )

    text = build_bm25_text(doc)

    assert "维护保养.txt" in text
    assert "环境适配维护" in text
    assert "9" in text


def test_split_txt_document_by_structure_splits_numbered_items():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    doc = Document(
        page_content=(
            "# 标题\n"
            "## 维护章节\n"
            "1. 第一条：拖布异味处理。\n"
            "继续说明。\n"
            "2. 第二条：水箱清洁。\n"
            "补充说明。"
        ),
        metadata={"source_name": "维护保养.txt", "source": "data/维护保养.txt"},
    )

    docs = split_txt_document_by_structure(splitter, doc)

    assert len(docs) == 2
    assert docs[0].metadata["section_title"] == "维护章节"
    assert docs[0].metadata["item_number"] == 1
    assert "拖布异味处理" in docs[0].page_content


def test_classify_query_type_detects_report_query():
    assert classify_query_type("结合本月表现，给我一些保养建议") == "report"


def test_classify_troubleshooting_subtype_detects_navigation():
    subtype = classify_troubleshooting_subtype("扫地机器人出现迷路或者反复打转时怎么排查？")

    assert subtype == "troubleshooting_navigation"


def test_classify_troubleshooting_subtype_detects_suction():
    subtype = classify_troubleshooting_subtype("扫地机器人吸力明显变弱，一般先检查什么？")

    assert subtype == "troubleshooting_suction"


def test_rewrite_query_for_retrieval_strengthens_report_query():
    rewritten = rewrite_query_for_retrieval("结合本月表现，给我一些保养建议", "report")

    assert "保养建议" in rewritten
    assert "滤网" in rewritten


def test_rewrite_query_for_retrieval_strengthens_navigation_troubleshooting():
    rewritten = rewrite_query_for_retrieval(
        "扫地机器人出现迷路或者反复打转时怎么排查？",
        "troubleshooting",
        "troubleshooting_navigation",
    )

    assert "迷路" in rewritten
    assert "地图错乱" in rewritten


def test_get_source_weight_penalizes_faq_for_report_queries():
    faq_doc = Document(page_content="faq", metadata={"source_name": "扫地机器人100问2.txt"})
    primary_doc = Document(page_content="main", metadata={"source_name": "维护保养.txt"})

    assert get_source_weight(primary_doc, "report") > get_source_weight(faq_doc, "report")


def test_get_source_weight_penalizes_buying_source_for_report_queries():
    buying_doc = Document(page_content="buy", metadata={"source_name": "选购指南.txt"})
    maintenance_doc = Document(page_content="maintain", metadata={"source_name": "维护保养.txt"})

    assert get_source_weight(maintenance_doc, "report") > get_source_weight(buying_doc, "report")


def test_get_source_weight_prefers_navigation_section_for_navigation_troubleshooting():
    nav_doc = Document(
        page_content="导航故障",
        metadata={"source_name": "故障排除.txt", "section_title": "导航建图与清扫异常（21-40）"},
    )
    generic_doc = Document(
        page_content="其他故障",
        metadata={"source_name": "故障排除.txt", "section_title": "基础通电与连接故障（1-20）"},
    )

    assert (
        get_source_weight(nav_doc, "troubleshooting", "troubleshooting_navigation")
        > get_source_weight(generic_doc, "troubleshooting", "troubleshooting_navigation")
    )
