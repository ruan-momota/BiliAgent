"""Supervisor Agent — 解析@消息意图、查缓存、路由分发"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.storage.cache import get_cached_summary

logger = logging.getLogger("biliagent.agent.supervisor")


class SupervisorAgent:
    """调度 Agent：判断@消息是否为有效总结请求，查缓存决定路由"""

    def __init__(self) -> None:
        self._llm = create_llm("supervisor", temperature=1)
        self._prompt_template = load_prompt("supervisor")

    async def run(
        self,
        content: str,
        user_name: str,
        video_id: str,
        platform: str,
    ) -> dict:
        """执行 Supervisor 决策

        Returns:
            {
                "route": "use_cache" | "analyze" | "ignore",
                "cached_summary": str | None,  # use_cache 时有值
                "reason": str | None,           # ignore 时有值
            }
        """
        # 1. 先查缓存
        cached = await get_cached_summary(platform, video_id)
        if cached is not None:
            logger.info("Cache hit for %s:%s", platform, video_id)
            return {
                "route": "use_cache",
                "cached_summary": cached.summary_text,
                "reason": None,
            }

        # 2. 用 LLM 判断意图
        prompt = self._prompt_template.format(
            content=content,
            user_name=user_name,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"用户 @消息原文：{content}"),
        ]

        text = await invoke_llm_with_retry(self._llm, messages, "supervisor")
        result = self._parse_response(text)

        if result.get("action") == "ignore":
            logger.info("Mention ignored: %s", result.get("reason"))
            return {
                "route": "ignore",
                "cached_summary": None,
                "reason": result.get("reason", "Not a summary request"),
            }

        # 默认走 analyze
        logger.info("Routing to analyzer for %s:%s", platform, video_id)
        return {
            "route": "analyze",
            "cached_summary": None,
            "reason": None,
        }

    @staticmethod
    def _parse_response(text: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        try:
            # 尝试提取 JSON 部分
            text = text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse supervisor response: %s", text)
        # 解析失败时默认 summarize
        return {"action": "summarize"}
