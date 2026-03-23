"""字幕分块模块 — 将长字幕文本拆分为语义完整的片段"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from biliagent.config import settings

logger = logging.getLogger("biliagent.rag.chunker")


def chunk_subtitles(text: str) -> list[str]:
    """将字幕文本按配置的 chunk_size / overlap 拆分为片段列表。

    Returns:
        按原文顺序排列的片段列表。
    """
    cfg = settings.rag
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        # 中文文本优先按段落 → 句号/问号/感叹号 → 逗号 → 字符拆分
        separators=["\n\n", "\n", "。", "？", "！", "，", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info("Chunked subtitle text (%d chars) into %d chunks", len(text), len(chunks))
    return chunks
