
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.retrieval import format_source_reference
from model.factory import chat_model
from rag.vector_store import VectorStoreService
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
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        retrieval_result = self.vector_store.hybrid_search(query)
        return retrieval_result["docs"]

    def get_sources(self, docs: list[Document]) -> list[str]:
        sources: list[str] = []
        for doc in docs:
            ref = format_source_reference(doc)
            if ref not in sources:
                sources.append(ref)
        return sources

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)
        if not context_docs:
            logger.warning("[rag_summarize] 未找到与问题相关的知识片段: %s", query)
            return "未检索到相关知识资料，请尝试换一种问法，或补充更具体的产品/场景信息。"

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += (
                f"【参考资料{counter}】\n"
                f"来源：{format_source_reference(doc)}\n"
                f"正文：{doc.page_content}\n\n"
            )

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        ).strip()

        sources = self.get_sources(context_docs)
        source_text = "、".join(sources)
        return f"{answer}\n\n参考来源：{source_text}"


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
