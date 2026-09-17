"""高层索引接口 — 字幕 → 分块 → embedding → 存入 ChromaDB"""

import logging

from biliagent.rag.chunker import chunk_subtitles
from biliagent.rag.vectorstore import add_chunks, is_video_indexed

logger = logging.getLogger("biliagent.rag.indexer")


def index_subtitles(
    platform: str,
    video_id: str,
    video_title: str,
    subtitles: str,
) -> int:
    """索引字幕到 ChromaDB（幂等：已索引则跳过）。

    Returns:
        分块数量（已索引时返回 0）。
    """
    if is_video_indexed(platform, video_id):
        logger.info("Video %s/%s already indexed, skipping", platform, video_id)
        return 0

    chunks = chunk_subtitles(subtitles)
    count = add_chunks(platform, video_id, video_title, chunks)
    return count
