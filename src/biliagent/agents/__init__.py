"""Agent 模块 — 提供 LLM 工厂函数、提示词加载、带重试的 LLM 调用"""

import asyncio
import logging
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from biliagent.config import settings

logger = logging.getLogger("biliagent.agents")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# 重试配置
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2  # 秒，指数退避基数


def create_llm(agent_name: str, temperature: float = 1) -> ChatOpenAI:
    """为指定 Agent 创建 LLM 实例（含超时配置）"""
    cfg = settings.get_agent_llm(agent_name)
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=temperature,
        request_timeout=60,  # 单次请求超时 60 秒
    )


async def invoke_llm_with_retry(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    agent_name: str = "unknown",
) -> str:
    """带重试的 LLM 调用（最多 3 次，指数退避）

    处理网络超时、API 限流(429)、服务端错误(5xx) 等临时故障。
    返回 response.content.strip()
    """
    last_error: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = await llm.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # 判断是否值得重试（网络/超时/限流/服务端错误）
            retryable = any(keyword in error_str for keyword in [
                "timeout", "timed out", "429", "rate limit",
                "500", "502", "503", "504", "connection",
                "network", "reset", "eof",
            ])

            if not retryable or attempt == LLM_MAX_RETRIES:
                logger.error(
                    "[%s] LLM call failed after %d attempt(s): %s",
                    agent_name, attempt, e,
                )
                raise

            delay = LLM_RETRY_BASE_DELAY ** attempt  # 2s, 4s, 8s
            logger.warning(
                "[%s] LLM call failed (attempt %d/%d), retrying in %ds: %s",
                agent_name, attempt, LLM_MAX_RETRIES, delay, e,
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


def load_prompt(name: str) -> str:
    """加载提示词模板文件"""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
