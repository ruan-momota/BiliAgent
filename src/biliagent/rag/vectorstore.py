"""ChromaDB 向量存储封装 — 初始化、存储、检索、删除"""

import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from biliagent.config import settings

logger = logging.getLogger("biliagent.rag.vectorstore")

# 模块级单例，惰性初始化
_embedding_fn: HuggingFaceEmbeddings | None = None
_chroma_client: chromadb.ClientAPI | None = None

COLLECTION_NAME = "video_subtitles"


def _get_embedding_fn() -> HuggingFaceEmbeddings:
    """获取 Embedding 函数（单例）"""
    global _embedding_fn
    if _embedding_fn is None:
        model_name = settings.rag.embedding_model
        logger.info("Loading embedding model: %s (first time may download ~400MB)", model_name)
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embedding_fn


def _get_chroma_client() -> chromadb.ClientAPI:
    """获取 ChromaDB 持久化客户端（单例）"""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = settings.rag.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized at %s", persist_dir)
    return _chroma_client


def get_vectorstore() -> Chroma:
    """获取 LangChain Chroma 向量存储实例"""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
    )


def is_video_indexed(platform: str, video_id: str) -> bool:
    """检查某个视频是否已经被索引"""
    client = _get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.get(
        where={"$and": [{"video_id": video_id}, {"platform": platform}]},
        limit=1,
    )
    return len(results["ids"]) > 0


def add_chunks(
    platform: str,
    video_id: str,
    video_title: str,
    chunks: list[str],
) -> int:
    """将字幕分块存入 ChromaDB。

    使用 "{platform}_{video_id}_{chunk_index}" 作为 document ID，天然幂等。

    Returns:
        存入的分块数量。
    """
    if not chunks:
        return 0

    total = len(chunks)
    ids = [f"{platform}_{video_id}_{i}" for i in range(total)]
    metadatas = [
        {
            "video_id": video_id,
            "platform": platform,
            "chunk_index": i,
            "total_chunks": total,
            "video_title": video_title or "",
        }
        for i in range(total)
    ]

    store = get_vectorstore()
    store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    logger.info(
        "Indexed %d chunks for video %s/%s", total, platform, video_id,
    )
    return total


def get_all_chunks_ordered(platform: str, video_id: str) -> list[str]:
    """按 chunk_index 顺序获取某个视频的全部分块文本。

    用于 Map-Reduce 摘要（需要按顺序处理所有分块）。
    """
    client = _get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.get(
        where={"$and": [{"video_id": video_id}, {"platform": platform}]},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        return []

    # 按 chunk_index 排序
    pairs = list(zip(results["metadatas"], results["documents"]))
    pairs.sort(key=lambda x: x[0].get("chunk_index", 0))
    return [doc for _, doc in pairs]


def similarity_search(query: str, platform: str, video_id: str, k: int | None = None) -> list[str]:
    """对指定视频的分块做相似度检索，返回 top-k 相关片段。

    用于 Q&A 场景（Phase 11）。
    """
    if k is None:
        k = settings.rag.qa_top_k

    store = get_vectorstore()
    docs = store.similarity_search(
        query=query,
        k=k,
        filter={"$and": [{"video_id": video_id}, {"platform": platform}]},
    )
    return [doc.page_content for doc in docs]


def delete_video_chunks(platform: str, video_id: str) -> None:
    """删除某个视频的全部分块索引"""
    client = _get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    # 先查出所有 IDs 再删除
    results = collection.get(
        where={"$and": [{"video_id": video_id}, {"platform": platform}]},
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info("Deleted %d chunks for video %s/%s", len(results["ids"]), platform, video_id)
