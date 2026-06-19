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

DEFAULT_QUERY_SET = [
    {
        "id": "maintenance_01",
        "query": "深圳最近比较潮湿，扫拖一体机器人怎么做维护保养？",
        "category": "维护保养",
        "expected_focus": "应该提到潮湿环境下的拖布、尘盒、水箱和机器通风维护。",
        "manual_checkpoints": ["是否贴合潮湿场景", "是否覆盖拖布/水箱清洁", "来源是否来自维护保养知识"],
    },
    {
        "id": "maintenance_02",
        "query": "拖布异味和水箱清洁一般怎么处理？",
        "category": "维护保养",
        "expected_focus": "应该包含拖布清洗晾干、水箱换水和定期清洁建议。",
        "manual_checkpoints": ["是否给出可执行步骤", "是否提到异味根因", "来源是否可信"],
    },
    {
        "id": "maintenance_03",
        "query": "扫地机器人滤网多久清理一次比较合适？",
        "category": "维护保养",
        "expected_focus": "应回答滤网清洁频率，并提醒根据使用强度调整。",
        "manual_checkpoints": ["是否回答频率", "是否有按使用场景调整", "是否需要补知识库"],
    },
    {
        "id": "troubleshooting_01",
        "query": "扫地机器人出现迷路或者反复打转时怎么排查？",
        "category": "故障排查",
        "subcategory": "troubleshooting_navigation",
        "expected_focus": "应提到传感器、地图、障碍物和重启重建地图等排查点。",
        "manual_checkpoints": ["是否覆盖核心排查步骤", "是否按先易后难组织", "来源是否贴题"],
    },
    {
        "id": "troubleshooting_02",
        "query": "扫地机器人吸力明显变弱，一般先检查什么？",
        "category": "故障排查",
        "subcategory": "troubleshooting_suction",
        "expected_focus": "应提到尘盒、滤网、滚刷堵塞和吸入口清理。",
        "manual_checkpoints": ["是否先给出高频原因", "是否包含清洁部件", "是否需要补知识库"],
    },
    {
        "id": "troubleshooting_03",
        "query": "机器频繁提示悬崖传感器异常，可能是什么原因？",
        "category": "故障排查",
        "subcategory": "troubleshooting_navigation",
        "expected_focus": "应提到传感器脏污、地面反光或环境影响。",
        "manual_checkpoints": ["是否说明异常原因", "是否有环境因素", "来源是否合理"],
    },
    {
        "id": "troubleshooting_navigation_04",
        "query": "机器人建图总是错乱，重新建图前我应该先检查什么？",
        "category": "故障排查",
        "subcategory": "troubleshooting_navigation",
        "expected_focus": "应提到反光环境、传感器灰尘、移动障碍物和删除旧地图重建。",
        "manual_checkpoints": ["是否命中导航建图场景", "是否包含重建地图建议", "来源是否集中在故障排除知识"],
    },
    {
        "id": "troubleshooting_suction_05",
        "query": "机器人最近吸尘效果差很多，风道和滤网应该怎么检查？",
        "category": "故障排查",
        "subcategory": "troubleshooting_suction",
        "expected_focus": "应提到尘盒、滤网、风道、吸口和刷组堵塞检查。",
        "manual_checkpoints": ["是否覆盖风道/滤网", "是否优先高频部件", "来源是否贴合吸力问题"],
    },
    {
        "id": "troubleshooting_mopping_01",
        "query": "拖地时突然不出水了，应该优先排查哪些地方？",
        "category": "故障排查",
        "subcategory": "troubleshooting_mopping",
        "expected_focus": "应提到水箱、出水口、出水管、出水量设置和堵塞问题。",
        "manual_checkpoints": ["是否命中拖地水箱场景", "是否覆盖出水链路", "来源是否合理"],
    },
    {
        "id": "troubleshooting_mopping_02",
        "query": "机器人拖地时漏水，水箱和密封圈一般怎么检查？",
        "category": "故障排查",
        "subcategory": "troubleshooting_mopping",
        "expected_focus": "应提到水箱盖、密封圈、出水阀和机身接口检查。",
        "manual_checkpoints": ["是否覆盖密封圈/水箱", "是否给出检查顺序", "来源是否集中在拖地故障知识"],
    },
    {
        "id": "troubleshooting_power_01",
        "query": "机器人最近充不进电，充电触点和电池应该怎么排查？",
        "category": "故障排查",
        "subcategory": "troubleshooting_power",
        "expected_focus": "应提到充电触点、适配器、电池状态和充电座供电检查。",
        "manual_checkpoints": ["是否命中充电电池场景", "是否包含触点/电池检查", "来源是否贴题"],
    },
    {
        "id": "troubleshooting_power_02",
        "query": "续航突然掉得很厉害，一般先判断是电池老化还是参数设置问题？",
        "category": "故障排查",
        "subcategory": "troubleshooting_power",
        "expected_focus": "应提到吸力档位、出水量、电池老化和回充频率等因素。",
        "manual_checkpoints": ["是否区分参数问题和电池问题", "是否有判断顺序", "来源是否合理"],
    },
    {
        "id": "buying_01",
        "query": "小户型更适合什么类型的扫地机器人？",
        "category": "选购建议",
        "expected_focus": "应提到机身尺寸、避障、集尘方式和性价比取舍。",
        "manual_checkpoints": ["是否真正面向小户型", "是否包含功能取舍", "答案是否具体"],
    },
    {
        "id": "buying_02",
        "query": "有宠物毛发的家庭选购扫地机器人要注意什么？",
        "category": "选购建议",
        "expected_focus": "应提到防缠绕、吸力、尘盒容量和毛发清理能力。",
        "manual_checkpoints": ["是否覆盖宠物毛发核心痛点", "是否提到防缠绕", "来源是否可信"],
    },
    {
        "id": "buying_03",
        "query": "预算有限时，扫拖一体和单扫地机器人应该怎么选？",
        "category": "选购建议",
        "expected_focus": "应对比预算、清洁诉求和维护成本。",
        "manual_checkpoints": ["是否有对比逻辑", "是否回答预算约束", "是否需要补知识库"],
    },
    {
        "id": "environment_01",
        "query": "地毯较多的家庭应该关注哪些功能？",
        "category": "环境适配",
        "expected_focus": "应提到地毯识别、自动增压、越障和拖地避让。",
        "manual_checkpoints": ["是否命中地毯场景", "是否提到增压或避让", "答案是否完整"],
    },
    {
        "id": "environment_02",
        "query": "家里门槛比较多，选扫地机器人时重点看什么？",
        "category": "环境适配",
        "expected_focus": "应提到越障高度、底盘设计和地图连续性。",
        "manual_checkpoints": ["是否关注门槛场景", "是否给出关键指标", "来源是否贴题"],
    },
    {
        "id": "environment_03",
        "query": "采光强、地面反光多的房间，会影响扫地机器人吗？",
        "category": "环境适配",
        "expected_focus": "应提到反光对传感器或识别的影响，以及规避建议。",
        "manual_checkpoints": ["是否解释影响原因", "是否给出规避建议", "是否需要补知识库"],
    },
    {
        "id": "report_01",
        "query": "给我生成这个月的使用报告",
        "category": "报告型问题",
        "expected_focus": "应能够触发报告模式，围绕本月使用情况生成结构化总结。",
        "manual_checkpoints": ["是否适合报告模式", "是否和月度数据相关", "输出是否清晰"],
    },
    {
        "id": "report_02",
        "query": "结合本月表现，给我一些保养建议",
        "category": "报告型问题",
        "expected_focus": "应结合月度使用情况输出针对性的保养建议。",
        "manual_checkpoints": ["是否结合月度上下文", "建议是否具体", "来源是否合理"],
    },
    {
        "id": "report_03",
        "query": "如果本月清扫次数明显下降，我该怎么理解这份报告？",
        "category": "报告型问题",
        "expected_focus": "应解释指标变化，并给出合理原因或进一步观察建议。",
        "manual_checkpoints": ["是否解释指标变化", "是否给出合理原因", "答案是否具备分析感"],
    },
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


def normalize_query_entry(query_entry: str | dict[str, Any], index: int | None = None) -> dict[str, Any]:
    if isinstance(query_entry, str):
        fallback_id = f"query_{index or 1:02d}"
        return {
            "id": fallback_id,
            "query": query_entry,
            "category": "未分类",
            "subcategory": "",
            "expected_focus": "",
            "manual_checkpoints": [],
        }

    query = query_entry.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Each query entry must contain a non-empty 'query' field.")

    fallback_id = f"query_{index or 1:02d}"
    manual_checkpoints = query_entry.get("manual_checkpoints", [])
    if not isinstance(manual_checkpoints, list) or not all(isinstance(item, str) for item in manual_checkpoints):
        raise ValueError("'manual_checkpoints' must be a list of strings.")

    return {
        "id": str(query_entry.get("id") or fallback_id),
        "query": query,
        "category": str(query_entry.get("category") or "未分类"),
        "subcategory": str(query_entry.get("subcategory") or ""),
        "expected_focus": str(query_entry.get("expected_focus") or ""),
        "manual_checkpoints": manual_checkpoints,
    }


def _build_manual_review_template(query_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed": False,
        "better_result": "",
        "source_reliability": "",
        "needs_knowledge_base_update": "",
        "notes": "",
        "checkpoints": [
            {"item": checkpoint, "status": "", "notes": ""}
            for checkpoint in query_entry["manual_checkpoints"]
        ],
    }


def compare_query(service: VectorStoreService, query: str | dict[str, Any]) -> dict[str, Any]:
    query_entry = normalize_query_entry(query)
    query_text = query_entry["query"]
    try:
        vector_docs = service.vector_search(query_text)
        hybrid_result = service.hybrid_search(query_text)
        bm25_docs = hybrid_result.get("bm25_docs")
        if bm25_docs is None:
            bm25_docs = service.bm25_search(query_text)
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
            "query_id": query_entry["id"],
            "query": query_text,
            "category": query_entry["category"],
            "subcategory": query_entry["subcategory"],
            "expected_focus": query_entry["expected_focus"],
            "status": "success",
            "mode": "full",
            "decision": decision,
            "query_type": hybrid_result.get("query_type", ""),
            "query_subtype": hybrid_result.get("query_subtype", ""),
            "retrieval_query": hybrid_result.get("retrieval_query", query_text),
            "vector": docs_to_rows(vector_docs),
            "bm25": docs_to_rows(bm25_docs),
            "hybrid": docs_to_rows(hybrid_docs),
            "notes": notes,
            "manual_review": _build_manual_review_template(query_entry),
        }
    except Exception as exc:
        logger.warning("[compare_retrieval] full comparison failed, fallback to bm25-only: %s", query_text)

        try:
            fallback_result = service.safe_hybrid_search(query_text)
            bm25_docs = fallback_result.get("bm25_docs") or fallback_result.get("docs", [])
            return {
                "query_id": query_entry["id"],
                "query": query_text,
                "category": query_entry["category"],
                "subcategory": query_entry["subcategory"],
                "expected_focus": query_entry["expected_focus"],
                "status": "fallback",
                "mode": "bm25-only",
                "decision": "bm25_only",
                "query_type": fallback_result.get("query_type", "fallback"),
                "query_subtype": fallback_result.get("query_subtype", query_entry["subcategory"]),
                "retrieval_query": fallback_result.get("retrieval_query", query_text),
                "vector": [],
                "bm25": docs_to_rows(bm25_docs),
                "hybrid": docs_to_rows(bm25_docs),
                "notes": [
                    "vector embedding unavailable in this environment; used bm25-only fallback",
                ],
                "error": str(exc),
                "manual_review": _build_manual_review_template(query_entry),
            }
        except Exception as fallback_exc:
            logger.exception("[compare_retrieval] bm25 fallback also failed: %s", query_text)
            return {
                "query_id": query_entry["id"],
                "query": query_text,
                "category": query_entry["category"],
                "subcategory": query_entry["subcategory"],
                "expected_focus": query_entry["expected_focus"],
                "status": "error",
                "mode": "failed",
                "decision": "error",
                "query_type": "failed",
                "query_subtype": query_entry["subcategory"],
                "retrieval_query": query_text,
                "vector": [],
                "bm25": [],
                "hybrid": [],
                "notes": ["comparison failed before retrieval results could be collected"],
                "error": str(fallback_exc),
                "manual_review": _build_manual_review_template(query_entry),
            }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = {"hybrid": 0, "vector": 0, "tie": 0, "bm25_only": 0, "error": 0}
    mode_counts = {"full": 0, "bm25-only": 0, "failed": 0}
    category_summary: dict[str, dict[str, Any]] = {}
    subcategory_summary: dict[str, dict[str, Any]] = {}
    success_count = 0
    fallback_count = 0
    manual_review_completed = 0
    for item in results:
        decision = item.get("decision", "error")
        mode = item.get("mode", "failed")
        category = item.get("category", "未分类")
        subcategory = item.get("subcategory") or item.get("query_subtype") or "未细分"
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if item.get("status") in {"success", "fallback"}:
            success_count += 1
        if item.get("status") == "fallback":
            fallback_count += 1
        if item.get("manual_review", {}).get("completed"):
            manual_review_completed += 1

        if category not in category_summary:
            category_summary[category] = {
                "total_queries": 0,
                "fallback_queries": 0,
                "decision_counts": {"hybrid": 0, "vector": 0, "tie": 0, "bm25_only": 0, "error": 0},
            }
        category_summary[category]["total_queries"] += 1
        category_summary[category]["decision_counts"][decision] += 1
        if item.get("status") == "fallback":
            category_summary[category]["fallback_queries"] += 1

        if subcategory not in subcategory_summary:
            subcategory_summary[subcategory] = {
                "total_queries": 0,
                "fallback_queries": 0,
                "decision_counts": {"hybrid": 0, "vector": 0, "tie": 0, "bm25_only": 0, "error": 0},
            }
        subcategory_summary[subcategory]["total_queries"] += 1
        subcategory_summary[subcategory]["decision_counts"][decision] += 1
        if item.get("status") == "fallback":
            subcategory_summary[subcategory]["fallback_queries"] += 1

    return {
        "total_queries": len(results),
        "successful_queries": success_count,
        "fallback_queries": fallback_count,
        "failed_queries": len(results) - success_count,
        "decision_counts": decision_counts,
        "mode_counts": mode_counts,
        "categories": category_summary,
        "subcategories": subcategory_summary,
        "manual_review_completed": manual_review_completed,
        "manual_review_pending": len(results) - manual_review_completed,
        "recommended_retrieval": _recommend_retrieval(decision_counts, mode_counts),
    }


def _recommend_retrieval(decision_counts: dict[str, int], mode_counts: dict[str, int]) -> str:
    if mode_counts.get("bm25-only", 0) and not mode_counts.get("full", 0):
        return "bm25_only"
    if decision_counts["hybrid"] >= decision_counts["vector"]:
        return "hybrid"
    return "vector"


def load_queries(query_file: str | None) -> list[dict[str, Any]]:
    if not query_file:
        return [normalize_query_entry(item, index) for index, item in enumerate(DEFAULT_QUERY_SET, start=1)]

    path = Path(query_file)
    if not path.is_absolute():
        path = Path(get_abs_path(query_file))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Query file must be a JSON array.")

    return [normalize_query_entry(item, index) for index, item in enumerate(data, start=1)]


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
