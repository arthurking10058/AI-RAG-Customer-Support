import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.retrieval import reciprocal_rank_fusion
from model.factory import get_embed_model
from services.exceptions import RetrievalError
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


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

            split_documents = self.spliter.split_documents(loaded)
            for index, doc in enumerate(split_documents, start=1):
                doc.metadata["chunk_id"] = index
                doc.metadata["source"] = doc.metadata.get("source", path)
                doc.metadata["source_name"] = doc.metadata.get("source_name", os.path.basename(path))
            documents.extend(split_documents)

        self._documents_cache = documents
        return documents

    def get_bm25_retriever(self) -> BM25Retriever:
        documents = self.get_documents_for_bm25()
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = chroma_conf["bm25_k"]
        return retriever

    def bm25_search(self, query: str) -> list[Document]:
        try:
            return self.get_bm25_retriever().invoke(query)
        except Exception as exc:
            logger.exception("[bm25_search] bm25 retrieval failed for query=%s", query)
            raise RetrievalError(
                "BM25 retrieval failed.",
                code="bm25_retrieval_failed",
                details=[str(exc)],
            ) from exc

    def safe_hybrid_search(self, query: str) -> dict[str, Any]:
        vector_docs: list[Document] = []
        vector_error: str | None = None

        try:
            vector_docs = self.vector_search(query)
        except RetrievalError as exc:
            vector_error = exc.message

        bm25_docs = self.bm25_search(query)
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

                split_document: list[Document] = self.spliter.split_documents(documents)

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


