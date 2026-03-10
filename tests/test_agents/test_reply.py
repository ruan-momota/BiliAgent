"""Reply Agent 单元测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biliagent.agents.reply import ReplyAgent


class TestReplySplitComment:
    """测试评论拆分（盖楼）逻辑"""

    def _make_agent(self, max_length=500):
        agent = ReplyAgent.__new__(ReplyAgent)
        agent._max_length = max_length
        return agent

    def test_short_text_no_split(self):
        """短文本不拆分"""
        agent = self._make_agent()
        parts = agent._split_comment("这是一条短评论")
        assert len(parts) == 1
        assert parts[0] == "这是一条短评论"

    def test_exact_limit_no_split(self):
        """刚好等于字数限制不拆分"""
        agent = self._make_agent(max_length=10)
        parts = agent._split_comment("1234567890")
        assert len(parts) == 1

    def test_long_text_splits(self):
        """超长文本会被拆分"""
        agent = self._make_agent(max_length=50)
        text = "段落一\n" * 20  # 总共超过 50 字
        parts = agent._split_comment(text)
        assert len(parts) > 1

    def test_split_adds_markers(self):
        """拆分后有「续 ↓」和「完」标记"""
        agent = self._make_agent(max_length=30)
        text = "第一段\n第二段\n第三段\n第四段\n第五段\n第六段\n第七段\n第八段"
        parts = agent._split_comment(text)
        if len(parts) > 1:
            assert "续 ↓" in parts[0]
            assert "完" in parts[-1]

    def test_single_long_paragraph_force_split(self):
        """单段超长会强制按字数切"""
        agent = self._make_agent(max_length=50)
        text = "A" * 200
        parts = agent._split_comment(text)
        assert len(parts) > 1


class TestReplyRun:
    """测试 Reply Agent 运行逻辑"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.reply.invoke_llm_with_retry")
    async def test_single_comment_success(self, mock_llm):
        """单条评论成功发布"""
        mock_llm.return_value = "这是格式化后的评论"

        mock_platform = AsyncMock()
        mock_platform.post_comment.return_value = "rpid_123"

        agent = ReplyAgent.__new__(ReplyAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {summary} {is_error} {error_reason} {max_length}"
        agent._platform = mock_platform
        agent._max_length = 500
        agent._send_interval = 0  # 测试中无需等待

        result = await agent.run(
            video_id="BV123",
            title="Test Video",
            summary="这是摘要",
        )

        assert result["success"] is True
        assert len(result["reply_parts"]) == 1
        assert result["comment_ids"][0] == "rpid_123"

    @pytest.mark.asyncio
    @patch("biliagent.agents.reply.invoke_llm_with_retry")
    async def test_comment_failure(self, mock_llm):
        """评论发布失败"""
        mock_llm.return_value = "评论内容"

        mock_platform = AsyncMock()
        mock_platform.post_comment.return_value = None

        agent = ReplyAgent.__new__(ReplyAgent)
        agent._llm = MagicMock()
        agent._prompt_template = "test {title} {summary} {is_error} {error_reason} {max_length}"
        agent._platform = mock_platform
        agent._max_length = 500
        agent._send_interval = 0

        result = await agent.run(
            video_id="BV123",
            title="Test",
            summary="摘要",
        )

        assert result["success"] is False
