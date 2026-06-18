
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.retrieval import format_source_reference
from model.factory import get_chat_model
from rag.vector_store import VectorStoreService
from services.exceptions import RetrievalError
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    logger.debug("[rag_summarize] prompt prepared")
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retrieve_context(self, query: str) -> dict:
        retrieval_result = self.vector_store.safe_hybrid_search(query)
        docs = retrieval_result["docs"]
        if not docs:
            raise RetrievalError(
                "No relevant knowledge snippets were found.",
                code="no_retrieval_docs",
                status_code=404,
            )
        return retrieval_result

    def get_sources(self, docs: list[Document]) -> list[str]:
        sources: list[str] = []
        for doc in docs:
            ref = format_source_reference(doc)
            if ref not in sources:
                sources.append(ref)
        return sources

    def rag_summarize(self, query: str) -> str:
        try:
            retrieval_result = self.retrieve_context(query)
        except RetrievalError as exc:
            logger.warning("[rag_summarize] retrieval stage failed for query=%s code=%s", query, exc.code)
            if exc.code == "no_retrieval_docs":
                return "未检索到相关知识资料，请尝试换一种问法，或补充更具体的产品/场景信息。"
            return "当前知识检索服务暂时不可用，已自动尝试降级处理，请稍后重试。"

        context_docs = retrieval_result["docs"]
        retrieval_mode = retrieval_result.get("mode", "hybrid")

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += (
                f"【参考资料{counter}】\n"
                f"来源：{format_source_reference(doc)}\n"
                f"正文：{doc.page_content}\n\n"
            )

        try:
            answer = self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            ).strip()
        except Exception as exc:
            logger.exception("[rag_summarize] summarization failed for query=%s", query)
            raise RetrievalError(
                "RAG summarization failed.",
                code="rag_summarization_failed",
                details=[str(exc)],
            ) from exc

        sources = self.get_sources(context_docs)
        source_text = "、".join(sources)
        mode_text = "BM25 fallback" if retrieval_mode == "bm25-only" else "hybrid retrieval"
        return f"{answer}\n\n参考来源：{source_text}\n检索模式：{mode_text}"


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
