"""VerifyCacheJudge Agent — 查询历史鉴别，LLM 判断提问相似度决定缓存复用"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.storage.verify_cache import get_cached_verifications, get_verification_by_id

logger = logging.getLogger("biliagent.agent.verify_cache_judge")


class VerifyCacheJudgeAgent:
    """缓存判断 Agent：查询同视频历史鉴别，用 LLM 判断新提问是否可复用缓存"""

    def __init__(self) -> None:
        self._llm = create_llm("verify_cache_judge", temperature=1)
        self._prompt_template = load_prompt("verify_cache_judge")

    async def run(
        self,
        video_id: str,
        question: str,
        platform: str,
    ) -> dict:
        """执行缓存判断

        Returns:
            {
                "route": "use_verify_cache" | "verify",
                "cached_verification": str | None,
                "cached_opinion": str | None,
                "cached_sources": str | None,
            }
        """
        # 1. 查询同视频的历史鉴别
        cached = await get_cached_verifications(platform, video_id)

        if not cached:
            logger.info("No verification cache for %s:%s", platform, video_id)
            return {
                "route": "verify",
                "cached_verification": None,
                "cached_opinion": None,
                "cached_sources": None,
            }

        # 2. 格式化历史记录供 LLM 判断
        cached_text = "\n".join(
            f"- ID: {v.id}, 提问: \"{v.question}\", 观点: {v.opinion or 'unknown'}"
            for v in cached
        )

        prompt = self._prompt_template.format(
            video_id=video_id,
            new_question=question,
            cached_verifications=cached_text,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"用户新提问：{question}"),
        ]

        text = await invoke_llm_with_retry(self._llm, messages, "verify_cache_judge")
        result = self._parse_response(text)

        if result.get("action") == "use_cache":
            cache_id = result.get("cache_id")
            if cache_id is not None:
                record = await get_verification_by_id(int(cache_id))
                if record:
                    logger.info(
                        "Verification cache hit for %s:%s (id=%d)",
                        platform, video_id, cache_id,
                    )
                    return {
                        "route": "use_verify_cache",
                        "cached_verification": record.verification,
                        "cached_opinion": record.opinion,
                        "cached_sources": record.sources,
                    }

        logger.info("Verification cache miss for %s:%s, regenerating", platform, video_id)
        return {
            "route": "verify",
            "cached_verification": None,
            "cached_opinion": None,
            "cached_sources": None,
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
            logger.warning("Failed to parse verify_cache_judge response: %s", text)
        # 解析失败默认重新生成
        return {"action": "regenerate"}
