"""Analyzer Agent — 获取视频信息、提取字幕、评估可总结性"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.models.schemas import VideoInfo
from biliagent.platforms.base import PlatformBase

logger = logging.getLogger("biliagent.agent.analyzer")


class AnalyzerAgent:
    """分析 Agent：获取视频元数据和字幕，用 LLM 评估是否可生成摘要"""

    def __init__(self, platform: PlatformBase) -> None:
        self._llm = create_llm("analyzer", temperature=1)
        self._prompt_template = load_prompt("analyzer")
        self._platform = platform

    async def run(self, video_id: str) -> dict:
        """执行分析

        Returns:
            {
                "can_summarize": bool,
                "video_info": VideoInfo | None,
                "subtitles": str | None,
                "reason": str | None,   # 不可总结时的原因
            }
        """
        # 1. 获取视频信息
        try:
            video_info = await self._platform.get_video_info(video_id)
        except Exception:
            logger.exception("Failed to get video info for %s", video_id)
            return {
                "can_summarize": False,
                "video_info": None,
                "subtitles": None,
                "reason": "Unable to retrieve video information.",
            }

        # 2. 获取字幕
        subtitles = await self._platform.get_subtitles(video_id)
        has_subtitles = subtitles is not None and len(subtitles.strip()) > 0

        # 3. 用 LLM 评估
        subtitle_preview = subtitles[:200] if subtitles else ""
        prompt = self._prompt_template.format(
            title=video_info.title,
            description=video_info.description,
            has_subtitles=has_subtitles,
            subtitle_preview=subtitle_preview,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请评估这个视频是否可以生成摘要。"),
        ]

        text = await invoke_llm_with_retry(self._llm, messages, "analyzer")
        result = self._parse_response(text)

        verdict = result.get("result", "")

        if verdict == "can_summarize":
            logger.info("Video %s is summarizable", video_id)
            return {
                "can_summarize": True,
                "video_info": video_info,
                "subtitles": subtitles,
                "reason": None,
            }

        reason = result.get("reason", "该视频暂无字幕，无法生成摘要。")
        logger.info("Video %s not summarizable: %s", video_id, reason)
        return {
            "can_summarize": False,
            "video_info": video_info,
            "subtitles": None,
            "reason": reason,
        }

    @staticmethod
    def _parse_response(text: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        try:
            text = text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse analyzer response: %s", text)
        return {"result": "no_subtitles", "reason": "该视频暂无字幕，无法生成摘要。"}
