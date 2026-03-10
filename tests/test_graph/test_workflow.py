"""工作流集成测试 — Mock 所有外部依赖（LLM + B站 API）"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biliagent.graph.state import AgentState
from biliagent.models.schemas import MentionInfo, VideoInfo


def _make_mention(**kwargs) -> MentionInfo:
    defaults = dict(
        mention_id="test_001",
        video_id="BV1test123",
        user_id="uid_001",
        user_name="test_user",
        content="帮我总结一下这个视频",
        platform="bilibili",
    )
    defaults.update(kwargs)
    return MentionInfo(**defaults)


class TestWorkflowNodes:
    """测试工作流各节点"""

    @pytest.mark.asyncio
    @patch("biliagent.agents.supervisor.invoke_llm_with_retry")
    @patch("biliagent.agents.supervisor.get_cached_summary")
    async def test_supervisor_node_analyze_route(self, mock_cache, mock_llm):
        """Supervisor 节点：无缓存 → analyze 路由"""
        from biliagent.graph.workflow import supervisor_node

        mock_cache.return_value = None
        mock_llm.return_value = '{"action": "summarize"}'

        state: AgentState = {
            "mention": _make_mention(),
            "traces": [],
        }

        result = await supervisor_node(state)
        assert result["route"] == "analyze"
        assert result["video_id"] == "BV1test123"
        assert len(result["traces"]) == 1
        assert result["traces"][0].agent_name == "supervisor"

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_analyzer_node_with_subtitles(self, mock_llm):
        """Analyzer 节点：有字幕 → can_summarize"""
        from biliagent.graph.workflow import analyzer_node

        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV1test123", title="Test Title", description="desc"
        )
        mock_platform.get_subtitles.return_value = "这是视频字幕内容..."

        mock_llm.return_value = '{"result": "can_summarize"}'

        state: AgentState = {
            "mention": _make_mention(),
            "video_id": "BV1test123",
            "traces": [],
        }

        result = await analyzer_node(state, mock_platform)
        assert result["can_summarize"] is True
        assert result["has_subtitles"] is True
        assert result["video_info"].title == "Test Title"

    @pytest.mark.asyncio
    @patch("biliagent.agents.summarizer.invoke_llm_with_retry")
    @patch("biliagent.graph.workflow.save_summary")
    async def test_summarizer_node(self, mock_save, mock_llm):
        """Summarizer 节点：生成摘要并保存"""
        from biliagent.graph.workflow import summarizer_node

        mock_llm.return_value = "这是一段测试摘要"
        mock_save.return_value = MagicMock()

        state: AgentState = {
            "mention": _make_mention(),
            "video_id": "BV1test123",
            "video_info": VideoInfo(
                video_id="BV1test123", title="Test", description=""
            ),
            "subtitles": "字幕内容",
            "traces": [],
        }

        result = await summarizer_node(state)
        assert result["summary"] == "这是一段测试摘要"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("biliagent.agents.reply.invoke_llm_with_retry")
    async def test_reply_node_success(self, mock_llm):
        """Reply 节点：成功发布评论"""
        from biliagent.graph.workflow import reply_node

        mock_llm.return_value = "格式化后的评论"

        mock_platform = AsyncMock()
        mock_platform.post_comment.return_value = "rpid_999"

        state: AgentState = {
            "mention": _make_mention(),
            "video_id": "BV1test123",
            "video_info": VideoInfo(
                video_id="BV1test123", title="Test", description=""
            ),
            "summary": "测试摘要",
            "traces": [],
        }

        result = await reply_node(state, mock_platform)
        assert result["success"] is True
        assert "rpid_999" in result["comment_ids"]


class TestWorkflowRouting:
    """测试条件路由逻辑"""

    def test_route_after_supervisor_use_cache(self):
        from biliagent.graph.workflow import route_after_supervisor
        state = {"route": "use_cache"}
        assert route_after_supervisor(state) == "reply"

    def test_route_after_supervisor_analyze(self):
        from biliagent.graph.workflow import route_after_supervisor
        state = {"route": "analyze"}
        assert route_after_supervisor(state) == "analyzer"

    def test_route_after_supervisor_ignore(self):
        from biliagent.graph.workflow import route_after_supervisor, END
        state = {"route": "ignore"}
        assert route_after_supervisor(state) == END

    def test_route_after_analyzer_can_summarize(self):
        from biliagent.graph.workflow import route_after_analyzer
        state = {"can_summarize": True}
        assert route_after_analyzer(state) == "summarizer"

    def test_route_after_analyzer_cannot_summarize(self):
        from biliagent.graph.workflow import route_after_analyzer
        state = {"can_summarize": False}
        assert route_after_analyzer(state) == "reply"
