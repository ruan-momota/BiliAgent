"""QA Agent 单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from biliagent.agents.qa import QAAgent
from biliagent.models.schemas import VideoInfo


class TestQAParseResponse:
    """测试 QA 的 JSON 解析逻辑"""

    def test_parse_found_true(self):
        text = '{"found": true, "answer": "视频里说 A 等于 B。"}'
        result = QAAgent._parse_response(text)
        assert result["found"] is True
        assert "A 等于 B" in result["answer"]

    def test_parse_found_false(self):
        text = '{"found": false, "answer": "视频中未详细提及"}'
        result = QAAgent._parse_response(text)
        assert result["found"] is False

    def test_parse_with_surrounding_text(self):
        text = '思考中...\n{"found": true, "answer": "xxx"}\n结束。'
        result = QAAgent._parse_response(text)
        assert result["found"] is True

    def test_parse_invalid_json_defaults_to_not_found(self):
        result = QAAgent._parse_response("not json at all")
        assert result["found"] is False


def _make_agent(top_k: int = 3, max_length: int = 500, long_threshold: int = 15000):
    agent = QAAgent.__new__(QAAgent)
    agent._llm = MagicMock()
    agent._prompt_rag = (
        "title={title} question={question} chunks={chunks} max_length={max_length}"
    )
    agent._prompt_direct = (
        "title={title} question={question} subtitles={subtitles} max_length={max_length}"
    )
    agent._platform = AsyncMock()
    agent._top_k = top_k
    agent._max_length = max_length
    agent._long_threshold = long_threshold
    return agent


class TestQARun:
    """测试 QA Agent 运行逻辑（分流：RAG vs Direct）"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.index_subtitles")
    @patch("biliagent.agents.qa.is_video_indexed")
    @patch("biliagent.agents.qa.invoke_llm_with_retry")
    async def test_already_indexed_uses_rag(
        self, mock_llm, mock_indexed, mock_index, mock_search
    ):
        """视频已索引 → 直接走 RAG，不拉字幕、不重新索引"""
        mock_indexed.return_value = True
        mock_search.return_value = [
            "片段1 这里讲到 XX 策略的核心是 YY",
            "片段2 具体步骤是 ABC",
        ]
        mock_llm.return_value = '{"found": true, "answer": "视频里说 XX 策略的核心是 YY。"}'

        agent = _make_agent()
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )

        result = await agent.run(video_id="BV123", question="XX 策略是什么")

        assert result["mode"] == "rag"
        assert result["found"] is True
        assert "YY" in result["answer"]
        assert len(result["chunks"]) == 2
        mock_index.assert_not_called()
        agent._platform.get_subtitles.assert_not_called()

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.index_subtitles")
    @patch("biliagent.agents.qa.is_video_indexed")
    @patch("biliagent.agents.qa.invoke_llm_with_retry")
    async def test_short_unindexed_uses_direct_mode(
        self, mock_llm, mock_indexed, mock_index, mock_search
    ):
        """短视频未索引 → Direct 模式，不入库、不检索"""
        mock_indexed.return_value = False
        mock_llm.return_value = '{"found": true, "answer": "字幕里说了 XYZ。"}'

        agent = _make_agent(long_threshold=15000)
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )
        agent._platform.get_subtitles.return_value = "短字幕内容，不到阈值。" * 10  # 远小于 15000

        result = await agent.run(video_id="BV123", question="XYZ?")

        assert result["mode"] == "direct"
        assert result["found"] is True
        assert result["chunks"] == []
        # Direct 模式不应索引、不应检索
        mock_index.assert_not_called()
        mock_search.assert_not_called()
        agent._platform.get_subtitles.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.index_subtitles")
    @patch("biliagent.agents.qa.is_video_indexed")
    @patch("biliagent.agents.qa.invoke_llm_with_retry")
    async def test_long_unindexed_indexes_and_uses_rag(
        self, mock_llm, mock_indexed, mock_index, mock_search
    ):
        """长视频未索引 → 先索引再 RAG"""
        mock_indexed.return_value = False
        mock_search.return_value = ["相关片段1", "相关片段2"]
        mock_llm.return_value = '{"found": true, "answer": "视频里讲到..."}'

        agent = _make_agent(long_threshold=100)  # 降低阈值以便用小字幕模拟长视频
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )
        agent._platform.get_subtitles.return_value = "A" * 500  # 超过阈值 100

        result = await agent.run(video_id="BV123", question="XX?")

        assert result["mode"] == "rag"
        assert result["found"] is True
        assert len(result["chunks"]) == 2
        mock_index.assert_called_once()
        agent._platform.get_subtitles.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.is_video_indexed")
    async def test_no_subtitles_returns_not_found(
        self, mock_indexed, mock_search
    ):
        """未索引且视频无字幕 → 直接返回未找到（不走 LLM、不检索）"""
        mock_indexed.return_value = False

        agent = _make_agent()
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )
        agent._platform.get_subtitles.return_value = None

        result = await agent.run(video_id="BV123", question="xxx")

        assert result["found"] is False
        assert result["chunks"] == []
        assert result["mode"] == "none"
        mock_search.assert_not_called()

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.is_video_indexed")
    async def test_empty_retrieval_returns_not_found(
        self, mock_indexed, mock_search
    ):
        """已索引但检索结果为空 → not found，不调用 LLM"""
        mock_indexed.return_value = True
        mock_search.return_value = []

        agent = _make_agent()
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )

        result = await agent.run(video_id="BV123", question="xxx")

        assert result["found"] is False
        assert result["chunks"] == []
        assert result["mode"] == "rag"

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.similarity_search")
    @patch("biliagent.agents.qa.is_video_indexed")
    @patch("biliagent.agents.qa.invoke_llm_with_retry")
    async def test_answer_truncated_when_too_long(
        self, mock_llm, mock_indexed, mock_search
    ):
        """LLM 返回超长回答会被截断到 max_length（RAG 路径）"""
        mock_indexed.return_value = True
        mock_search.return_value = ["片段"]
        long_answer = "很长的回答" * 200  # 1200 字
        mock_llm.return_value = f'{{"found": true, "answer": "{long_answer}"}}'

        agent = _make_agent(max_length=100)
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )

        result = await agent.run(video_id="BV123", question="xxx")

        assert len(result["answer"]) == 100
        assert result["answer"].endswith("...")

    @pytest.mark.asyncio
    @patch("biliagent.agents.qa.is_video_indexed")
    @patch("biliagent.agents.qa.invoke_llm_with_retry")
    async def test_direct_mode_answer_truncated(self, mock_llm, mock_indexed):
        """Direct 模式同样应用 max_length 截断"""
        mock_indexed.return_value = False
        long_answer = "直答" * 200
        mock_llm.return_value = f'{{"found": true, "answer": "{long_answer}"}}'

        agent = _make_agent(max_length=50, long_threshold=15000)
        agent._platform.get_video_info.return_value = VideoInfo(
            video_id="BV123", title="Test", description=""
        )
        agent._platform.get_subtitles.return_value = "短字幕"

        result = await agent.run(video_id="BV123", question="xxx")

        assert result["mode"] == "direct"
        assert len(result["answer"]) == 50
        assert result["answer"].endswith("...")
