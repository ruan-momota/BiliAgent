"""Supervisor Agent 单元测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biliagent.agents.supervisor import SupervisorAgent


class TestSupervisorParseResponse:
    """测试 Supervisor 的 JSON 解析逻辑"""

    def test_parse_valid_summarize(self):
        text = '{"action": "summarize", "video_id": "BV123"}'
        result = SupervisorAgent._parse_response(text)
        assert result["action"] == "summarize"

    def test_parse_valid_ignore(self):
        text = '{"action": "ignore", "reason": "not a request"}'
        result = SupervisorAgent._parse_response(text)
        assert result["action"] == "ignore"
        assert result["reason"] == "not a request"

    def test_parse_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"action": "summarize"}\nDone.'
        result = SupervisorAgent._parse_response(text)
        assert result["action"] == "summarize"

    def test_parse_invalid_json_defaults_to_summarize(self):
        text = "This is not JSON at all"
        result = SupervisorAgent._parse_response(text)
        assert result["action"] == "summarize"

    def test_parse_empty_string_defaults_to_summarize(self):
        result = SupervisorAgent._parse_response("")
        assert result["action"] == "summarize"


class TestSupervisorRun:
    """测试 Supervisor 运行逻辑"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.supervisor.invoke_llm_with_retry")
    @patch("biliagent.agents.supervisor.get_cached_summary")
    async def test_cache_hit_returns_use_cache(self, mock_cache, mock_llm):
        """LLM 判断 summarize + 缓存命中 → use_cache 路由"""
        mock_llm.return_value = '{"action": "summarize"}'
        mock_summary = MagicMock()
        mock_summary.summary_text = "Cached summary content"
        mock_cache.return_value = mock_summary

        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test prompt {content} {user_name}"

        result = await agent.run(
            content="帮我总结这个视频",
            user_name="test_user",
            video_id="BV123",
            platform="bilibili",
        )

        assert result["route"] == "use_cache"
        assert result["cached_summary"] == "Cached summary content"

    @pytest.mark.asyncio
    @patch("biliagent.agents.supervisor.invoke_llm_with_retry")
    @patch("biliagent.agents.supervisor.get_cached_summary")
    async def test_no_cache_summarize_intent(self, mock_cache, mock_llm):
        """无缓存 + LLM 判断为 summarize → analyze 路由"""
        mock_cache.return_value = None
        mock_llm.return_value = '{"action": "summarize"}'

        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test prompt {content} {user_name}"

        result = await agent.run(
            content="帮我总结这个视频",
            user_name="test_user",
            video_id="BV123",
            platform="bilibili",
        )

        assert result["route"] == "analyze"

    @pytest.mark.asyncio
    @patch("biliagent.agents.supervisor.invoke_llm_with_retry")
    @patch("biliagent.agents.supervisor.get_cached_summary")
    async def test_no_cache_ignore_intent(self, mock_cache, mock_llm):
        """无缓存 + LLM 判断为 ignore → ignore 路由"""
        mock_cache.return_value = None
        mock_llm.return_value = '{"action": "ignore", "reason": "闲聊"}'

        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test prompt {content} {user_name}"

        result = await agent.run(
            content="你好呀",
            user_name="test_user",
            video_id="BV123",
            platform="bilibili",
        )

        assert result["route"] == "ignore"
        assert result["reason"] == "闲聊"
