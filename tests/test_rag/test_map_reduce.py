"""Map-Reduce 摘要流程集成测试"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from biliagent.agents.summarizer import SummarizerAgent


class TestMapReduceSummarizer:
    """测试 Summarizer 的 Map-Reduce 模式"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.get_all_chunks_ordered")
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_map_reduce_flow(self, mock_llm, mock_get_chunks):
        """长视频触发 Map-Reduce：3 个 chunk → 3 次 map + 1 次 reduce"""
        mock_get_chunks.return_value = ["chunk1_text", "chunk2_text", "chunk3_text"]

        # Map 阶段返回每块要点，Reduce 阶段返回最终摘要
        call_count = 0

        async def mock_invoke(llm, messages, agent_name=""):
            nonlocal call_count
            call_count += 1
            if "summarizer_map" in agent_name:
                return f"· 要点{call_count}"
            return "这是一份合并后的最终摘要，涵盖了视频的核心内容。"

        mock_llm.side_effect = mock_invoke

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "unused"
        agent._map_template = "Map: {title} {chunk_index}/{total_chunks}\n{chunk_text}"
        agent._reduce_template = "Reduce: {title}\n{all_chunk_summaries}\n{max_length}"
        agent._max_length = 500

        result = await agent.run(
            title="长视频测试",
            description="描述",
            subtitles="",  # Map-Reduce 不用原始字幕
            is_long_video=True,
            video_id="BV_LONG",
            platform="bilibili",
        )

        assert "summary" in result
        assert len(result["summary"]) <= 500
        # 3 次 map + 1 次 reduce = 4 次 LLM 调用
        assert mock_llm.call_count == 4
        mock_get_chunks.assert_called_once_with("bilibili", "BV_LONG")

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_direct_mode_unchanged(self, mock_llm):
        """短视频走直接模式，不调用 ChromaDB"""
        mock_llm.return_value = "这是一段直接生成的视频摘要。"

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "{title} {description} {subtitles} {max_length}"
        agent._map_template = "unused"
        agent._reduce_template = "unused"
        agent._max_length = 500

        result = await agent.run(
            title="短视频",
            description="描述",
            subtitles="短字幕内容",
            is_long_video=False,
        )

        assert "summary" in result
        assert mock_llm.call_count == 1  # 仅 1 次直接调用

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.get_all_chunks_ordered")
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_map_reduce_no_chunks_fallback(self, mock_llm, mock_get_chunks):
        """Map-Reduce 找不到分块时返回兜底文案"""
        mock_get_chunks.return_value = []

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "unused"
        agent._map_template = "unused"
        agent._reduce_template = "unused"
        agent._max_length = 500

        result = await agent.run(
            title="测试",
            description="",
            subtitles="",
            is_long_video=True,
            video_id="BV_EMPTY",
            platform="bilibili",
        )

        assert "无法获取" in result["summary"]
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.get_all_chunks_ordered")
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    async def test_map_reduce_truncation(self, mock_llm, mock_get_chunks):
        """Map-Reduce 结果超长时会被截断"""
        mock_get_chunks.return_value = ["chunk1"]
        mock_llm.return_value = "A" * 600  # 超过 500 字

        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "unused"
        agent._map_template = "{title} {chunk_index}/{total_chunks}\n{chunk_text}"
        agent._reduce_template = "{title}\n{all_chunk_summaries}\n{max_length}"
        agent._max_length = 500

        result = await agent.run(
            title="测试",
            description="",
            subtitles="",
            is_long_video=True,
            video_id="BV_TRUNC",
            platform="bilibili",
        )

        assert len(result["summary"]) == 500
        assert result["summary"].endswith("...")
