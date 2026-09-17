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
        agent._prompt_template = "test {title} {description} {has_subtitles} {subtitle_preview} {text_source}"
        agent._platform = mock_platform

        # 禁用 SenseVoice 降级，聚焦 "无字幕 → 不可总结" 本身的路径
        with patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = ""
            mock_settings.rag.long_video_threshold = 15000
            result = await agent.run("BV123")

        assert result["can_summarize"] is False
        assert "字幕" in (result["reason"] or "")


class TestAnalyzerTextSource:
    """验证 text_source 字段：subtitle / transcription / None"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_subtitles_present_sets_source_subtitle(self, mock_llm):
        """有字幕 → text_source == 'subtitle'，不触发转录"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="T", description=""
        )
        mock_platform.get_subtitles.return_value = "有字幕"
        mock_llm.return_value = '{"result": "can_summarize"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "{title} {description} {has_subtitles} {subtitle_preview} {text_source}"
        agent._platform = mock_platform

        with patch("biliagent.agents.analyzer.download_audio") as mock_dl, \
             patch("biliagent.agents.analyzer.transcribe") as mock_tr, \
             patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = "http://fake"
            mock_settings.rag.long_video_threshold = 15000
            result = await agent.run("BV123")

        assert result["can_summarize"] is True
        assert result["text_source"] == "subtitle"
        mock_dl.assert_not_called()
        mock_tr.assert_not_called()

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.transcribe")
    @patch("biliagent.agents.analyzer.download_audio")
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_no_subtitles_transcription_success(
        self, mock_llm, mock_download, mock_transcribe
    ):
        """无字幕 + 转录成功 → can_summarize=True，text_source='transcription'，临时文件被清理"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="T", description=""
        )
        mock_platform.get_subtitles.return_value = None
        mock_platform.get_audio_url.return_value = "http://fake/audio.m4a"

        fake_audio = MagicMock()
        fake_audio.unlink = MagicMock()
        mock_download.return_value = fake_audio
        mock_transcribe.return_value = "从音频转出来的文字内容"
        mock_llm.return_value = '{"result": "can_summarize"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "{title} {description} {has_subtitles} {subtitle_preview} {text_source}"
        agent._platform = mock_platform

        with patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = "http://fake"
            mock_settings.rag.long_video_threshold = 15000
            result = await agent.run("BV123")

        assert result["can_summarize"] is True
        assert result["text_source"] == "transcription"
        assert result["subtitles"] == "从音频转出来的文字内容"
        mock_transcribe.assert_awaited_once()
        fake_audio.unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.transcribe")
    @patch("biliagent.agents.analyzer.download_audio")
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_no_subtitles_transcription_failure(
        self, mock_llm, mock_download, mock_transcribe
    ):
        """无字幕 + 转录失败 → can_summarize=False，text_source=None"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="T", description=""
        )
        mock_platform.get_subtitles.return_value = None
        mock_platform.get_audio_url.return_value = "http://fake/audio.m4a"

        fake_audio = MagicMock()
        fake_audio.unlink = MagicMock()
        mock_download.return_value = fake_audio
        mock_transcribe.return_value = None  # 转录失败
        mock_llm.return_value = '{"result": "no_subtitles", "reason": "转录失败"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "{title} {description} {has_subtitles} {subtitle_preview} {text_source}"
        agent._platform = mock_platform

        with patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = "http://fake"
            mock_settings.rag.long_video_threshold = 15000
            result = await agent.run("BV123")

        assert result["can_summarize"] is False
        assert result["text_source"] is None
        assert result["subtitles"] is None
        # 即使转录失败，临时文件也应被清理
        fake_audio.unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.transcribe")
    @patch("biliagent.agents.analyzer.download_audio")
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_no_subtitles_no_audio_url_skips_transcription(
        self, mock_llm, mock_download, mock_transcribe
    ):
        """无字幕 + 拿不到音频 URL → 不触发下载/转录"""
        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="T", description=""
        )
        mock_platform.get_subtitles.return_value = None
        mock_platform.get_audio_url.return_value = None

        mock_llm.return_value = '{"result": "no_subtitles", "reason": "无字幕且无音频"}'

        agent = AnalyzerAgent.__new__(AnalyzerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "{title} {description} {has_subtitles} {subtitle_preview} {text_source}"
        agent._platform = mock_platform

        with patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = "http://fake"
            mock_settings.rag.long_video_threshold = 15000
            result = await agent.run("BV123")

        assert result["can_summarize"] is False
        assert result["text_source"] is None
        mock_download.assert_not_called()
        mock_transcribe.assert_not_called()
