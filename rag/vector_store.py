import os
import re
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from core.retrieval import reciprocal_rank_fusion
from model.factory import get_embed_model
from services.exceptions import RetrievalError
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

FAQ_SOURCE_NAMES = {
    "扫地机器人100问.pdf",
    "扫地机器人100问2.txt",
    "扫拖一体机器人100问.txt",
}

PRIMARY_SOURCE_GROUPS = {
    "maintenance": {"维护保养.txt"},
    "troubleshooting": {"故障排除.txt"},
    "buying": {"选购指南.txt"},
}

QUERY_TYPE_KEYWORDS = {
    "report": ["本月", "报告", "月度", "表现", "数据", "记录", "建议"],
    "troubleshooting": ["故障", "排查", "迷路", "打转", "异常", "不转", "变弱", "漏水", "报警", "不出水", "吸尘效果差", "续航突然掉", "充不进电"],
    "maintenance": ["维护", "保养", "清洁", "异味", "滤网", "拖布", "水箱", "回南天", "潮湿", "除味"],
    "buying": ["选购", "怎么选", "适合", "功能", "参数", "预算", "门槛", "小户型", "地毯", "宠物"],
}

TROUBLESHOOTING_SUBTYPE_KEYWORDS = {
    "troubleshooting_navigation": ["迷路", "打转", "地图", "建图", "定位", "路线", "漏扫", "传感器", "避障", "悬崖", "防跌落", "反光"],
    "troubleshooting_suction": ["吸力", "风道", "尘盒", "滤网", "吸口", "主刷", "边刷", "滚刷", "堵塞"],
    "troubleshooting_mopping": ["拖地", "拖布", "水箱", "出水", "漏水", "清洁液", "污水", "水垢", "抬升"],
    "troubleshooting_power": ["充电", "电池", "续航", "亏电", "电量", "触点", "适配器", "回充", "发热"],
}

TROUBLESHOOTING_SUBTYPE_REWRITES = {
    "troubleshooting_navigation": "故障排查 导航建图 迷路 打转 地图错乱 定位异常 传感器 悬崖传感器 防跌落 反光 强光 镜面 避障",
    "troubleshooting_suction": "故障排查 吸力风道 吸力下降 尘盒 滤网 风道 吸口 主刷 边刷 滚刷 堵塞 吸尘效果差",
    "troubleshooting_mopping": "故障排查 拖地水箱 拖布 水箱 出水 不出水 出水管 出水口 漏水 清洁液 污水 水垢 抬升 水泵",
    "troubleshooting_power": "故障排查 充电电池 充不进电 续航衰减 亏电 回充异常 触点 适配器 发热 电池老化 吸力档位 出水量 参数设置",
}

TROUBLESHOOTING_SUBTYPE_CONTENT_HINTS = {
    "troubleshooting_navigation": {
        "悬崖": 3.0,
        "防跌落": 2.8,
        "反光": 2.2,
        "强光": 2.0,
        "镜面": 1.8,
        "地图错乱": 1.8,
        "建图": 1.3,
        "定位": 1.1,
        "传感器": 0.35,
    },
    "troubleshooting_suction": {
        "吸力": 1.8,
        "风道": 1.8,
        "滤网": 1.7,
        "吸口": 1.5,
        "尘盒": 1.5,
        "主刷": 1.2,
        "边刷": 1.0,
        "堵塞": 1.4,
    },
    "troubleshooting_mopping": {
        "不出水": 2.2,
        "出水量": 1.8,
        "出水管": 1.8,
        "出水口": 1.6,
        "水箱": 1.5,
        "水泵": 1.5,
        "漏水": 1.4,
    },
    "troubleshooting_power": {
        "续航": 1.8,
        "电池": 1.8,
        "回充": 1.5,
        "触点": 1.6,
        "适配器": 1.5,
        "充电": 1.5,
        "出水量": 1.0,
        "吸力档位": 1.2,
        "参数": 0.9,
    },
}

NAVIGATION_STRONG_QUERY_HINTS = ("悬崖", "防跌落", "反光", "强光", "镜面")


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return []

    tokens: list[str] = []

    latin_parts = re.findall(r"[a-z0-9_+\-./]+", normalized)
    tokens.extend(latin_parts)

    chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for sequence in chinese_sequences:
        tokens.append(sequence)
        if len(sequence) == 1:
            continue
        for size in (2, 3):
            if len(sequence) < size:
                continue
            tokens.extend(sequence[index : index + size] for index in range(len(sequence) - size + 1))

    return tokens


def classify_query_type(query: str) -> str:
    # Troubleshooting should win when a query combines maintenance-like nouns
    # (such as 滤网/水箱) with obvious failure symptoms.
    if any(keyword in query for keyword in QUERY_TYPE_KEYWORDS["troubleshooting"]):
        return "troubleshooting"

    for query_type in ("report", "troubleshooting", "maintenance", "buying"):
        if any(keyword in query for keyword in QUERY_TYPE_KEYWORDS[query_type]):
            return query_type
    return "general"


def classify_troubleshooting_subtype(query: str) -> str | None:
    if classify_query_type(query) != "troubleshooting":
        return None

    scores: dict[str, int] = {}
    for subtype, keywords in TROUBLESHOOTING_SUBTYPE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in query)
        if score > 0:
            scores[subtype] = score

    if not scores:
        return "troubleshooting_navigation"

    return max(scores.items(), key=lambda item: item[1])[0]


def rewrite_query_for_retrieval(query: str, query_type: str, query_subtype: str | None = None) -> str:
    if query_type == "report":
        if "保养建议" in query:
            return "月度使用后的机器人维护保养建议 拖布 水箱 滤网 尘盒 风道 清洁 周期 建议"
        return f"月度使用报告相关的维护建议与清洁重点 {query}"
    if query_type == "troubleshooting":
        prefix = TROUBLESHOOTING_SUBTYPE_REWRITES.get(
            query_subtype or "troubleshooting_navigation",
            "故障排查 检测 修复",
        )
        return f"{prefix} {query}"
    if query_type == "maintenance":
        return f"维护保养 清洁 周期 建议 潮湿 回南天 拖布 水箱 滤网 除味 晾干 {query}"
    if query_type == "buying":
        return f"选购建议 功能参数 场景适配 宠物 地毯 小户型 门槛 预算 {query}"
    return query


def build_bm25_text(doc: Document) -> str:
    parts = [
        str(doc.metadata.get("source_name", "")),
        str(doc.metadata.get("section_title", "")),
        str(doc.metadata.get("subsection_title", "")),
        str(doc.metadata.get("item_number", "")),
        doc.page_content,
    ]
    return "\n".join(part for part in parts if part).strip()


def get_source_weight(doc: Document, query_type: str, query_subtype: str | None = None) -> float:
    source_name = str(doc.metadata.get("source_name", ""))
    section_title = str(doc.metadata.get("section_title", ""))
    is_faq = source_name in FAQ_SOURCE_NAMES
    is_primary = source_name in (
        PRIMARY_SOURCE_GROUPS.get("maintenance", set())
        | PRIMARY_SOURCE_GROUPS.get("troubleshooting", set())
        | PRIMARY_SOURCE_GROUPS.get("buying", set())
    )

    base_weight = 1.0
    if is_faq:
        base_weight = 0.7
    elif is_primary:
        base_weight = 1.1

    if query_type == "report":
        if source_name in PRIMARY_SOURCE_GROUPS["maintenance"]:
            return base_weight * 1.9
        if source_name in PRIMARY_SOURCE_GROUPS["buying"]:
            return base_weight * 0.45
        if is_faq:
            return base_weight * 0.45
        return base_weight

    if query_type == "maintenance":
        if source_name in PRIMARY_SOURCE_GROUPS["maintenance"]:
            return base_weight * 1.7
        if source_name in PRIMARY_SOURCE_GROUPS["buying"]:
            return base_weight * 0.55
        if is_faq:
            return base_weight * 0.55
        return base_weight

    if query_type == "troubleshooting":
        if source_name in PRIMARY_SOURCE_GROUPS["troubleshooting"]:
            weight = base_weight * 1.9
            if query_subtype == "troubleshooting_navigation" and "导航建图" in section_title:
                return weight * 1.45
            if query_subtype == "troubleshooting_suction" and "基础通电" in section_title:
                return weight * 1.35
            if query_subtype == "troubleshooting_mopping" and ("集尘拖地" in section_title or "续航充电" in section_title):
                return weight * 1.35
            if query_subtype == "troubleshooting_power" and "续航充电" in section_title:
                return weight * 1.5
            return weight
        if source_name in PRIMARY_SOURCE_GROUPS["maintenance"]:
            weight = base_weight * 1.2
            if query_subtype == "troubleshooting_suction" and "扫地功能专属维护" in section_title:
                return weight * 1.2
            if query_subtype == "troubleshooting_mopping" and "扫拖一体拖地功能专属维护" in section_title:
                return weight * 1.25
            if query_subtype == "troubleshooting_power" and ("长期存放维护" in section_title or "故障预防维护" in section_title):
                return weight * 1.15
            return weight
        if source_name in PRIMARY_SOURCE_GROUPS["buying"]:
            if query_subtype == "troubleshooting_power":
                return base_weight * 0.28
            return base_weight * 0.5
        if is_faq:
            if query_subtype == "troubleshooting_navigation":
                return base_weight * 0.38
            if query_subtype == "troubleshooting_suction":
                return base_weight * 0.4
            if query_subtype == "troubleshooting_mopping":
                return base_weight * 0.39
            if query_subtype == "troubleshooting_power":
                return base_weight * 0.32
            return base_weight * 0.5
        return base_weight

    if query_type == "buying":
        if source_name in PRIMARY_SOURCE_GROUPS["buying"]:
            return base_weight * 1.6
        if source_name in PRIMARY_SOURCE_GROUPS["maintenance"]:
            return base_weight * 0.8
        if source_name in PRIMARY_SOURCE_GROUPS["troubleshooting"]:
            return base_weight * 0.65
        if is_faq:
            return base_weight * 0.9
        return base_weight

    return base_weight


def get_subtype_content_bonus(doc: Document, query_subtype: str | None, query_text: str | None = None) -> float:
    if not query_subtype:
        return 0.0

    hints = TROUBLESHOOTING_SUBTYPE_CONTENT_HINTS.get(query_subtype, {})
    if not hints:
        return 0.0

    text = str(doc.metadata.get("item_text") or doc.page_content)
    if query_subtype == "troubleshooting_navigation" and query_text:
        strong_query_hints = [hint for hint in NAVIGATION_STRONG_QUERY_HINTS if hint in query_text]
        if strong_query_hints:
            bonus = sum(hints.get(hint, 0.0) for hint in strong_query_hints if hint in text)
            if bonus <= 0:
                return 0.0
            return min(bonus, 4.5)

    bonus = sum(weight for hint, weight in hints.items() if hint in text)
    if bonus <= 0:
        return 0.0

    # A light additive bonus to pull subtype-specific chunks upward
    # without overwhelming the base BM25 score.
    return min(bonus, 4.5)


def get_navigation_strong_match_bonus(doc: Document, query_text: str) -> float:
    strong_query_hints = [hint for hint in NAVIGATION_STRONG_QUERY_HINTS if hint in query_text]
    if not strong_query_hints:
        return 0.0

    item_text = str(doc.metadata.get("item_text") or "")
    if not item_text:
        return 0.0

    matched_hints = [hint for hint in strong_query_hints if hint in item_text]
    if not matched_hints:
        return 0.0

    # Hard rerank bonus for the narrow troubleshooting_03-style queries.
    # This is intentionally much stronger than the generic subtype bonus so
    # true悬崖/反光/防跌落条目 can outrank generic sensor mentions.
    return 10.0 + len(matched_hints) * 2.0


def split_txt_document_by_structure(
    splitter: RecursiveCharacterTextSplitter,
    doc: Document,
) -> list[Document]:
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$")
    item_pattern = re.compile(r"^(\d+)\.\s*(.+)$")

    title = ""
    section = ""
    subsection = ""
    section_context_lines: list[str] = []
    current_item_number: int | None = None
    current_item_lines: list[str] = []
    structured_docs: list[Document] = []

    def flush_item() -> None:
        nonlocal current_item_number, current_item_lines
        if not current_item_lines:
            return

        metadata = dict(doc.metadata)
        if section:
            metadata["section_title"] = section
        if subsection:
            metadata["subsection_title"] = subsection
        if current_item_number is not None:
            metadata["item_number"] = current_item_number
        metadata["item_text"] = "\n".join(current_item_lines).strip()

        text_parts = [title, section, subsection, *section_context_lines, *current_item_lines]
        content = "\n".join(part for part in text_parts if part).strip()
        structured_docs.extend(splitter.split_documents([Document(page_content=content, metadata=metadata)]))

        current_item_number = None
        current_item_lines = []

    for raw_line in doc.page_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = heading_pattern.match(line)
        if heading_match:
            flush_item()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level == 1:
                title = heading_text
                section = ""
                subsection = ""
                section_context_lines = []
            elif level == 2:
                section = heading_text
                subsection = ""
                section_context_lines = []
            else:
                subsection = heading_text
                section_context_lines = []
            continue

        item_match = item_pattern.match(line)
        if item_match:
            flush_item()
            current_item_number = int(item_match.group(1))
            current_item_lines = [line]
            continue

        if current_item_lines:
            current_item_lines.append(line)
        else:
            section_context_lines.append(line)

    flush_item()

    if structured_docs:
        return structured_docs

    return split_documents_with_structure(splitter, [doc])


def split_documents_with_structure(splitter: RecursiveCharacterTextSplitter, documents: list[Document]) -> list[Document]:
    structured_docs: list[Document] = []
    heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)

    for doc in documents:
        source_name = str(doc.metadata.get("source_name", ""))
        if source_name.endswith(".txt"):
            structured_docs.extend(split_txt_document_by_structure(splitter, doc))
            continue

        text = doc.page_content
        matches = list(heading_pattern.finditer(text))

        if not matches:
            structured_docs.extend(splitter.split_documents([doc]))
            continue

        for index, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue
            section_doc = Document(page_content=section_text, metadata={**doc.metadata, "section_title": heading})
            structured_docs.extend(splitter.split_documents([section_doc]))

    return structured_docs


class VectorStoreService:
    def __init__(self):
        persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=persist_directory,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self._documents_cache: list[Document] | None = None
        self._bm25_docs_cache: list[Document] | None = None
        self._bm25_vectorizer: BM25Okapi | None = None

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def vector_search(self, query: str) -> list[Document]:
        try:
            return self.get_retriever().invoke(query)
        except Exception as exc:
            logger.warning("[vector_search] vector retrieval failed for query=%s: %s", query, exc)
            raise RetrievalError(
                "Vector retrieval is unavailable.",
                code="vector_retrieval_unavailable",
                details=[str(exc)],
            ) from exc

    def get_documents_for_bm25(self) -> list[Document]:
        if self._documents_cache is not None:
            return self._documents_cache

        documents: list[Document] = []
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            if path.endswith("txt"):
                loaded = txt_loader(path)
            elif path.endswith("pdf"):
                loaded = pdf_loader(path)
            else:
                loaded = []

            if not loaded:
                continue

            split_documents = split_documents_with_structure(self.spliter, loaded)
            for index, doc in enumerate(split_documents, start=1):
                doc.metadata["chunk_id"] = index
                doc.metadata["source"] = doc.metadata.get("source", path)
                doc.metadata["source_name"] = doc.metadata.get("source_name", os.path.basename(path))
            documents.extend(split_documents)

        self._documents_cache = documents
        return documents

    def get_bm25_index(self) -> tuple[BM25Okapi, list[Document]]:
        if self._bm25_vectorizer is not None and self._bm25_docs_cache is not None:
            return self._bm25_vectorizer, self._bm25_docs_cache

        documents = self.get_documents_for_bm25()
        processed_corpus = [tokenize_for_bm25(build_bm25_text(doc)) for doc in documents]
        self._bm25_vectorizer = BM25Okapi(processed_corpus)
        self._bm25_docs_cache = documents
        return self._bm25_vectorizer, self._bm25_docs_cache

    def bm25_search(
        self,
        query: str,
        query_type: str | None = None,
        query_subtype: str | None = None,
        retrieval_query: str | None = None,
    ) -> list[Document]:
        try:
            vectorizer, docs = self.get_bm25_index()
            effective_query_type = query_type or classify_query_type(query)
            effective_query_subtype = query_subtype or classify_troubleshooting_subtype(query)
            effective_query = retrieval_query or rewrite_query_for_retrieval(query, effective_query_type, effective_query_subtype)
            query_tokens = tokenize_for_bm25(effective_query)
            if not query_tokens:
                return []
            scored_indexes = sorted(
                (
                    (
                        index,
                        float(score) * get_source_weight(docs[index], effective_query_type, effective_query_subtype)
                        + get_subtype_content_bonus(docs[index], effective_query_subtype, query)
                        + (
                            get_navigation_strong_match_bonus(docs[index], query)
                            if effective_query_subtype == "troubleshooting_navigation"
                            else 0.0
                        ),
                    )
                    for index, score in enumerate(vectorizer.get_scores(query_tokens))
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            top_indexes = [index for index, score in scored_indexes if score > 0][: chroma_conf["bm25_k"]]
            return [docs[index] for index in top_indexes]
        except Exception as exc:
            logger.exception("[bm25_search] bm25 retrieval failed for query=%s", query)
            raise RetrievalError(
                "BM25 retrieval failed.",
                code="bm25_retrieval_failed",
                details=[str(exc)],
            ) from exc

    def safe_hybrid_search(self, query: str) -> dict[str, Any]:
        query_type = classify_query_type(query)
        query_subtype = classify_troubleshooting_subtype(query)
        retrieval_query = rewrite_query_for_retrieval(query, query_type, query_subtype)
        vector_docs: list[Document] = []
        vector_error: str | None = None

        try:
            vector_docs = self.vector_search(query)
        except RetrievalError as exc:
            vector_error = exc.message

        bm25_docs = self.bm25_search(
            query,
            query_type=query_type,
            query_subtype=query_subtype,
            retrieval_query=retrieval_query,
        )
        if vector_docs:
            fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs], k=chroma_conf["rrf_k"])
            final_docs = fused_docs[: chroma_conf["k"]]
            mode = "hybrid"
        else:
            final_docs = bm25_docs[: chroma_conf["k"]]
            mode = "bm25-only"

        logger.info(
            "[safe_hybrid_search] query=%s mode=%s vector_hits=%s bm25_hits=%s final_hits=%s",
            query,
            mode,
            len(vector_docs),
            len(bm25_docs),
            len(final_docs),
        )

        return {
            "mode": mode,
            "query_type": query_type,
            "query_subtype": query_subtype,
            "retrieval_query": retrieval_query,
            "vector_docs": vector_docs,
            "bm25_docs": bm25_docs,
            "docs": final_docs,
            "vector_error": vector_error,
        }

    def hybrid_search(self, query: str) -> dict:
        return self.safe_hybrid_search(query)

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = split_documents_with_structure(self.spliter, documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                for index, doc in enumerate(split_document, start=1):
                    doc.metadata["chunk_id"] = index
                    doc.metadata["source"] = doc.metadata.get("source", path)
                    doc.metadata["source_name"] = doc.metadata.get("source_name", os.path.basename(path))

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


