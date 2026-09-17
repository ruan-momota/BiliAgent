"""Summarizer Agent — 基于字幕生成结构化摘要（含 Map-Reduce 模式）"""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.config import settings
from biliagent.rag.vectorstore import get_all_chunks_ordered

logger = logging.getLogger("biliagent.agent.summarizer")


class SummarizerAgent:
    """生成 Agent：根据视频标题、简介、字幕生成摘要

    支持两种模式：
    - 直接模式：短视频，字幕直接送 LLM
    - Map-Reduce 模式：长视频，分块提取要点后合并
    """

    def __init__(self) -> None:
        self._llm = create_llm("summarizer", temperature=1)
        self._prompt_template = load_prompt("summarizer")
        self._map_template = load_prompt("summarizer_map")
        self._reduce_template = load_prompt("summarizer_reduce")
        self._max_length = settings.app.summary_max_length

    async def run(
        self,
        title: str,
        description: str,
        subtitles: str,
        is_long_video: bool = False,
        video_id: str | None = None,
        platform: str = "bilibili",
    ) -> dict:
        """生成视频摘要

        Args:
            is_long_video: 是否使用 Map-Reduce 模式
            video_id: 长视频时用于从 ChromaDB 取分块
            platform: 平台标识

        Returns:
            {"summary": str}
        """
        if is_long_video and video_id:
            summary = await self._run_map_reduce(title, video_id, platform)
        else:
            summary = await self._run_direct(title, description, subtitles)

        # 后处理：硬截断兜底
        if len(summary) > self._max_length:
            summary = summary[: self._max_length - 3] + "..."
            logger.warning("Summary truncated to %d chars", self._max_length)

        logger.info("Summary generated, length=%d", len(summary))
        return {"summary": summary}

    async def _run_direct(self, title: str, description: str, subtitles: str) -> str:
        """直接模式：字幕全文送 LLM"""
        prompt = self._prompt_template.format(
            title=title,
            description=description,
            subtitles=subtitles,
            max_length=self._max_length,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请生成视频摘要。"),
        ]
        return await invoke_llm_with_retry(self._llm, messages, "summarizer")

    async def _run_map_reduce(self, title: str, video_id: str, platform: str) -> str:
        """Map-Reduce 模式：分块提取要点 → 合并最终摘要"""
        chunks = get_all_chunks_ordered(platform, video_id)
        if not chunks:
            logger.warning("No chunks found for %s/%s, falling back to empty summary", platform, video_id)
            return "无法获取视频分块内容。"

        total = len(chunks)
        logger.info("Map-Reduce: processing %d chunks for video %s", total, video_id)

        # Map 阶段：并发提取每块要点
        async def _map_chunk(index: int, chunk_text: str) -> str:
            prompt = self._map_template.format(
                title=title,
                chunk_index=index + 1,
                total_chunks=total,
                chunk_text=chunk_text,
            )
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="请提取这部分内容的关键要点。"),
            ]
            return await invoke_llm_with_retry(self._llm, messages, f"summarizer_map[{index + 1}/{total}]")

        map_results = await asyncio.gather(
            *[_map_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        )
        logger.info("Map phase complete: %d chunk summaries generated", len(map_results))

        # Reduce 阶段：合并所有要点
        all_chunk_summaries = "\n\n".join(
            f"【第{i + 1}部分】\n{result}" for i, result in enumerate(map_results)
        )
        reduce_prompt = self._reduce_template.format(
            title=title,
            all_chunk_summaries=all_chunk_summaries,
            max_length=self._max_length,
        )
        messages = [
            SystemMessage(content=reduce_prompt),
            HumanMessage(content="请合并为最终摘要。"),
        ]
        summary = await invoke_llm_with_retry(self._llm, messages, "summarizer_reduce")
        logger.info("Reduce phase complete")
        return summary
