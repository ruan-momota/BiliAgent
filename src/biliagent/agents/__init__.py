"""Agent 模块 — 提供 LLM 工厂函数和提示词加载"""

from pathlib import Path

from langchain_openai import ChatOpenAI

from biliagent.config import settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def create_llm(agent_name: str, temperature: float = 0.7) -> ChatOpenAI:
    """为指定 Agent 创建 LLM 实例"""
    cfg = settings.get_agent_llm(agent_name)
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=temperature,
    )


def load_prompt(name: str) -> str:
    """加载提示词模板文件"""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
