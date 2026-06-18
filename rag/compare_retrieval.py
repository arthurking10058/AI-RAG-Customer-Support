from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from core.retrieval import document_identity, format_source_reference
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

DEFAULT_QUERIES = [
    "深圳最近比较潮湿，扫拖一体机器人怎么做维护保养？",
    "小户型更适合什么类型的扫地机器人？",
    "扫地机器人出现迷路或者反复打转时怎么排查？",
    "有宠物毛发的家庭选购扫地机器人要注意什么？",
    "拖布异味和水箱清洁一般怎么处理？",
    "地毯较多的家庭应该关注哪些功能？",
]


def build_preview(doc: Document, max_length: int = 120) -> str:
    text = " ".join(doc.page_content.split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def docs_to_rows(docs: list[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        rows.append(
            {
                "rank": rank,
                "source": format_source_reference(doc),
                "identity": document_identity(doc),
                "preview": build_preview(doc),
            }
        )
    return rows


def _select_available_results(result: dict[str, Any]) -> list[Document]:
    if result.get("vector_docs"):
        return result["vector_docs"]
    return result.get("bm25_docs", [])


def compare_query(service: VectorStoreService, query: str) -> dict[str, Any]:
    try:
        vector_docs = service.vector_search(query)
        hybrid_result = service.hybrid_search(query)
        bm25_docs = hybrid_result.get("bm25_docs")
        if bm25_docs is None:
            bm25_docs = service.bm25_search(query)
        hybrid_docs = hybrid_result["docs"]

        vector_ids = {document_identity(doc) for doc in vector_docs}
        bm25_ids = {document_identity(doc) for doc in bm25_docs}
        hybrid_ids = {document_identity(doc) for doc in hybrid_docs}
        hybrid_only_ids = hybrid_ids - vector_ids

        decision = "tie"
        if not vector_docs and bm25_docs and hybrid_ids == bm25_ids:
            decision = "bm25_only"
        elif hybrid_only_ids:
            decision = "hybrid"
        elif not hybrid_docs and vector_docs:
            decision = "vector"

        notes = ["vector-only and hybrid returned the same top documents"]
        if decision == "hybrid":
            notes = ["hybrid introduces new top results compared with vector-only retrieval"]
        elif decision == "bm25_only":
            notes = ["vector retrieval returned no hits; the current result is effectively carried by BM25"]

        return {
            "query": query,
            "status": "success",
            "mode": "full",
            "decision": decision,
            "vector": docs_to_rows(vector_docs),
            "bm25": docs_to_rows(bm25_docs),
            "hybrid": docs_to_rows(hybrid_docs),
            "notes": notes,
        }
    except Exception as exc:
        logger.warning("[compare_retrieval] full comparison failed, fallback to bm25-only: %s", query)

        try:
            bm25_docs = service.bm25_search(query)
            return {
                "query": query,
                "status": "fallback",
                "mode": "bm25-only",
                "decision": "bm25_only",
                "vector": [],
                "bm25": docs_to_rows(bm25_docs),
                "hybrid": docs_to_rows(bm25_docs),
                "notes": [
                    "vector embedding unavailable in this environment; used bm25-only fallback",
                ],
                "error": str(exc),
            }
        except Exception as fallback_exc:
            logger.exception("[compare_retrieval] bm25 fallback also failed: %s", query)
            return {
                "query": query,
                "status": "error",
                "mode": "failed",
                "decision": "error",
                "vector": [],
                "bm25": [],
                "hybrid": [],
                "notes": ["comparison failed before retrieval results could be collected"],
                "error": str(fallback_exc),
            }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = {"hybrid": 0, "vector": 0, "tie": 0, "bm25_only": 0, "error": 0}
    mode_counts = {"full": 0, "bm25-only": 0, "failed": 0}
    success_count = 0
    fallback_count = 0
    for item in results:
        decision = item.get("decision", "error")
        mode = item.get("mode", "failed")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if item.get("status") in {"success", "fallback"}:
            success_count += 1
        if item.get("status") == "fallback":
            fallback_count += 1

    return {
        "total_queries": len(results),
        "successful_queries": success_count,
        "fallback_queries": fallback_count,
        "failed_queries": len(results) - success_count,
        "decision_counts": decision_counts,
        "mode_counts": mode_counts,
        "recommended_retrieval": _recommend_retrieval(decision_counts, mode_counts),
    }


def _recommend_retrieval(decision_counts: dict[str, int], mode_counts: dict[str, int]) -> str:
    if mode_counts.get("bm25-only", 0) and not mode_counts.get("full", 0):
        return "bm25_only"
    if decision_counts["hybrid"] >= decision_counts["vector"]:
        return "hybrid"
    return "vector"


def load_queries(query_file: str | None) -> list[str]:
    if not query_file:
        return DEFAULT_QUERIES

    path = Path(query_file)
    if not path.is_absolute():
        path = Path(get_abs_path(query_file))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Query file must be a JSON array of strings.")

    return data


def write_output(payload: dict[str, Any], output_file: str | None) -> None:
    if not output_file:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    path = Path(output_file)
    if not path.is_absolute():
        path = Path(get_abs_path(output_file))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison result to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vector-only retrieval against hybrid retrieval.")
    parser.add_argument(
        "--query-file",
        help="Optional JSON file containing a list of evaluation queries.",
    )
    parser.add_argument(
        "--output",
        default="outputs/retrieval_comparison.json",
        help="Output JSON file path relative to project root.",
    )
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    logger.info("[compare_retrieval] loaded %s evaluation queries", len(queries))

    service = VectorStoreService()
    results = [compare_query(service, query) for query in queries]
    payload = {
        "summary": summarize(results),
        "results": results,
    }
    write_output(payload, args.output)


if __name__ == "__main__":
    main()
