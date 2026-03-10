"""Summarizer Agent — 基于字幕生成结构化摘要"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.config import settings

logger = logging.getLogger("biliagent.agent.summarizer")


class SummarizerAgent:
    """生成 Agent：根据视频标题、简介、字幕生成摘要"""

    def __init__(self) -> None:
        self._llm = create_llm("summarizer", temperature=1)
        self._prompt_template = load_prompt("summarizer")
        self._max_length = settings.app.summary_max_length

    async def run(
        self,
        title: str,
        description: str,
        subtitles: str,
    ) -> dict:
        """生成视频摘要

        Returns:
            {
                "summary": str,  # 生成的摘要文本
            }
        """
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

        summary = await invoke_llm_with_retry(self._llm, messages, "summarizer")

        # 后处理：硬截断兜底
        if len(summary) > self._max_length:
            summary = summary[: self._max_length - 3] + "..."
            logger.warning("Summary truncated to %d chars", self._max_length)

        logger.info("Summary generated, length=%d", len(summary))
        return {"summary": summary}
