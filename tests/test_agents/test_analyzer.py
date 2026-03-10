"""Analyzer Agent 单元测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biliagent.agents.analyzer import AnalyzerAgent
from biliagent.models.schemas import VideoInfo


class TestAnalyzerParseResponse:
    """测试 Analyzer 的 JSON 解析逻辑"""

    def test_parse_can_summarize(self):
        text = '{"result": "can_summarize", "reason": ""}'
        result = AnalyzerAgent._parse_response(text)
        assert result["result"] == "can_summarize"

    def test_parse_no_subtitles(self):
        text = '{"result": "no_subtitles", "reason": "该视频无字幕"}'
        result = AnalyzerAgent._parse_response(text)
        assert result["result"] == "no_subtitles"

    def test_parse_invalid_defaults_to_no_subtitles(self):
        result = AnalyzerAgent._parse_response("not json")
        assert result["result"] == "no_subtitles"

    def test_parse_with_markdown_wrapper(self):
        text = '```json\n{"result": "can_summarize"}\n```'
        # rfind("}") 可以找到最后的 }
        result = AnalyzerAgent._parse_response(text)
        assert result["result"] == "can_summarize"


class TestAnalyzerRun:
    """测试 Analyzer 运行逻辑"""

    @pytest.mark.asyncio
    async def test_video_info_failure_returns_cannot_summarize(self):
        """获取视频信息失败 → 不可总结"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.side_effect = Exception("API error")

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = ""
        agent._platform = mock_platform

        result = await agent.run("BV123")

        assert result["can_summarize"] is False
        assert result["video_info"] is None

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_has_subtitles_can_summarize(self, mock_llm):
        """有字幕 + LLM 判断可总结"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test Video", description="desc"
        )
        mock_platform.get_subtitles.return_value = "这是一段字幕内容，很长很长..."

        mock_llm.return_value = '{"result": "can_summarize"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {description} {has_subtitles} {subtitle_preview}"
        agent._platform = mock_platform

        result = await agent.run("BV123")

        assert result["can_summarize"] is True
        assert result["video_info"].title == "Test Video"
        assert result["subtitles"] is not None

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_no_subtitles_cannot_summarize(self, mock_llm):
        """无字幕 → LLM 判断不可总结"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test Video", description="desc"
        )
        mock_platform.get_subtitles.return_value = None

        mock_llm.return_value = '{"result": "no_subtitles", "reason": "无字幕"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {description} {has_subtitles} {subtitle_preview}"
        agent._platform = mock_platform

        result = await agent.run("BV123")

        assert result["can_summarize"] is False
        assert "字幕" in (result["reason"] or "")
