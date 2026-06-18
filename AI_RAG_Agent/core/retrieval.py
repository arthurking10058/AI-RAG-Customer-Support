from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from langchain_core.documents import Document


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def document_identity(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    chunk_id = doc.metadata.get("chunk_id", "")
    content = normalize_text(doc.page_content)
    return f"{source}::{chunk_id}::{content}"


def reciprocal_rank_fusion(result_sets: Iterable[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for docs in result_sets:
        for rank, doc in enumerate(docs, start=1):
            key = document_identity(doc)
            doc_map[key] = doc
            scores[key] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def format_source_reference(doc: Document) -> str:
    source = doc.metadata.get("source_name") or doc.metadata.get("source") or "未知来源"
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id is None:
        return str(source)
    return f"{source}#chunk-{chunk_id}"
