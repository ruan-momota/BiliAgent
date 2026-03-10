"""Summarizer Agent 单元测试"""

import pytest
from unittest.mock import patch, MagicMock

from biliagent.agents.summarizer import SummarizerAgent


class TestSummarizerRun:
    """测试 Summarizer 运行逻辑"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_normal_summary(self, mock_llm):
        """正常生成摘要"""
        mock_llm.return_value = "这是一段视频摘要，内容不超过500字。"

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {description} {subtitles} {max_length}"
        agent._max_length = 500

        result = await agent.run(
            title="测试视频",
            description="一段描述",
            subtitles="字幕内容...",
        )

        assert "summary" in result
        assert len(result["summary"]) <= 500

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_summary_truncation(self, mock_llm):
        """超长摘要会被截断"""
        mock_llm.return_value = "A" * 600  # 超过 500 字

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {description} {subtitles} {max_length}"
        agent._max_length = 500

        result = await agent.run(
            title="测试视频",
            description="一段描述",
            subtitles="字幕...",
        )

        assert len(result["summary"]) == 500
        assert result["summary"].endswith("...")
