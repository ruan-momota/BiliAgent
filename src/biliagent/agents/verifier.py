"""Verifier Agent — 联网搜索 + 视频内容鉴别 + 观点生成

绕过 langchain，直接使用 OpenAI client 调用 Kimi $web_search builtin_function。
"""

import json
import logging
from datetime import datetime

from openai import AsyncOpenAI

from biliagent.agents import load_prompt
from biliagent.config import settings
from biliagent.models.schemas import VideoInfo
from biliagent.platforms.base import PlatformBase

logger = logging.getLogger("biliagent.agent.verifier")


class VerifierAgent:
    """鉴别 Agent：独立获取视频信息，联网搜索事实，生成鉴别评论"""

    def __init__(self, platform: PlatformBase) -> None:
        cfg = settings.get_agent_llm("verifier")
        self._client = AsyncOpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )
        self._model = cfg["model"]
        self._prompt_template = load_prompt("verifier")
        self._platform = platform
        self._max_length = settings.app.verify_max_length

    async def run(self, video_id: str, question: str) -> dict:
        """执行鉴别

        Returns:
            {
                "video_info": VideoInfo | None,
                "verification": str,
                "opinion": str,
                "sources": list[str],
            }
        """
        # 1. 独立获取视频信息
        try:
            video_info = await self._platform.get_video_info(video_id)
        except Exception:
            logger.exception("Failed to get video info for %s", video_id)
            video_info = VideoInfo(video_id=video_id, title="未知", description="")

        # 2. 获取字幕
        subtitles = await self._platform.get_subtitles(video_id)
        subtitles_text = subtitles or "（该视频无字幕）"

        # 3. 构造提示词
        prompt = self._prompt_template.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            title=video_info.title,
            description=video_info.description,
            subtitles=subtitles_text[:settings.app.subtitle_max_length],
            question=question,
            max_length=self._max_length,
        )

        # 4. 调用 Kimi API（带 $web_search）
        text = await self._call_kimi_with_search(prompt, question)
        result = self._parse_response(text)

        logger.info(
            "Verification for %s: opinion=%s", video_id, result.get("opinion")
        )

        return {
            "video_info": video_info,
            "verification": result.get("verification", "鉴别失败，请稍后重试。"),
            "opinion": result.get("opinion", "neutral"),
            "sources": result.get("sources", []),
        }

    async def _call_kimi_with_search(self, system_prompt: str, question: str) -> str:
        """调用 Kimi API，启用 $web_search builtin_function"""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请对这个视频进行鉴别。用户提问：{question}"},
                ],
                tools=[{
                    "type": "builtin_function",
                    "function": {"name": "$web_search"},
                }],
                temperature=1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Kimi API call with web_search failed: %s", e)
            # 降级：不带搜索重试
            return await self._call_kimi_fallback(system_prompt, question)

    async def _call_kimi_fallback(self, system_prompt: str, question: str) -> str:
        """降级调用：不带 $web_search"""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请对这个视频进行鉴别。用户提问：{question}"},
                ],
                temperature=1,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("Kimi fallback call also failed")
            return '{"opinion": "neutral", "verification": "抱歉，鉴别服务暂时不可用。", "sources": []}'

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
            logger.warning("Failed to parse verifier response: %s", text)
        return {
            "opinion": "neutral",
            "verification": text if text else "鉴别失败，请稍后重试。",
            "sources": [],
        }
