"""
统一读取项目配置。
"""
from __future__ import annotations

import os

import yaml

from utils.path_tool import get_abs_path


def _load_yaml_config(config_path: str, encoding: str = "utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f) or {}


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    config = _load_yaml_config(config_path, encoding)
    config["chat_model_name"] = os.getenv("AI_CHAT_MODEL_NAME", config.get("chat_model_name", "qwen3-max"))
    config["embedding_model_name"] = os.getenv(
        "AI_EMBEDDING_MODEL_NAME",
        config.get("embedding_model_name", "text-embedding-v4"),
    )
    return config


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    config = _load_yaml_config(config_path, encoding)
    config["persist_directory"] = os.getenv("CHROMA_PERSIST_DIRECTORY", config.get("persist_directory", "chroma_db"))
    config["collection_name"] = os.getenv("CHROMA_COLLECTION_NAME", config.get("collection_name", "agent"))
    config["k"] = int(os.getenv("RAG_TOP_K", config.get("k", 4)))
    config["bm25_k"] = int(os.getenv("RAG_BM25_TOP_K", config.get("bm25_k", config["k"])))
    config["rrf_k"] = int(os.getenv("RAG_RRF_K", config.get("rrf_k", 60)))
    config["chunk_size"] = int(os.getenv("RAG_CHUNK_SIZE", config.get("chunk_size", 200)))
    config["chunk_overlap"] = int(os.getenv("RAG_CHUNK_OVERLAP", config.get("chunk_overlap", 20)))
    return config


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    config = _load_yaml_config(config_path, encoding)
    config["external_data_path"] = os.getenv(
        "AGENT_EXTERNAL_DATA_PATH",
        config.get("external_data_path", "data/external/records.csv"),
    )
    return config


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
