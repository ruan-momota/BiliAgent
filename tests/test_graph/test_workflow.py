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
        assert result["text_source"] == "subtitle"

    @pytest.mark.asyncio
    @patch("biliagent.agents.analyzer.transcribe")
    @patch("biliagent.agents.analyzer.download_audio")
    @patch("biliagent.agents.analyzer.invoke_llm_with_retry")
    async def test_analyzer_node_no_subtitles_transcription_fallback(
        self, mock_llm, mock_download, mock_transcribe
    ):
        """集成：无字幕 → 降级转录成功 → can_summarize=True，trace 中 text_source='transcription'"""
        from biliagent.graph.workflow import analyzer_node

        mock_platform = AsyncMock()
        mock_platform.get_video_info.return_value = VideoInfo(
            video_id="BV1test123", title="T", description=""
        )
        mock_platform.get_subtitles.return_value = None
        mock_platform.get_audio_url.return_value = "http://fake/audio.m4a"

        fake_audio = MagicMock()
        fake_audio.unlink = MagicMock()
        mock_download.return_value = fake_audio
        mock_transcribe.return_value = "音频转录出的正文内容"
        mock_llm.return_value = '{"result": "can_summarize"}'

        state: AgentState = {
            "mention": _make_mention(),
            "video_id": "BV1test123",
            "traces": [],
        }

        with patch("biliagent.agents.analyzer.settings") as mock_settings:
            mock_settings.sensevoice.api_url = "http://fake"
            mock_settings.rag.long_video_threshold = 15000
            result = await analyzer_node(state, mock_platform)

        assert result["can_summarize"] is True
        assert result["has_subtitles"] is True
        assert result["text_source"] == "transcription"
        assert result["subtitles"] == "音频转录出的正文内容"
        analyzer_trace = next(t for t in result["traces"] if t.agent_name == "analyzer")
        assert analyzer_trace.output_data is not None
        assert '"text_source": "transcription"' in analyzer_trace.output_data

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

    def test_route_after_supervisor_ask(self):
        from biliagent.graph.workflow import route_after_supervisor
        state = {"route": "ask"}
        assert route_after_supervisor(state) == "qa"


class TestQANode:
    """测试 QA 节点"""

    @pytest.mark.asyncio
    async def test_qa_node_populates_state(self):
        """QA 节点：结果写入 state 并生成一条 trace"""
        from biliagent.graph.workflow import qa_node

        mock_platform = AsyncMock()

        state: AgentState = {
            "mention": _make_mention(content="视频里 X 是什么"),
            "video_id": "BV1test123",
            "question": "视频里 X 是什么",
            "traces": [],
        }

        with patch("biliagent.graph.workflow.QAAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value={
                "video_info": VideoInfo(video_id="BV1test123", title="T", description=""),
                "found": True,
                "answer": "基于视频片段的回答",
                "chunks": ["片段A", "片段B"],
            })
            mock_agent_cls.return_value = mock_agent

            result = await qa_node(state, mock_platform)

        assert result["qa_found"] is True
        assert result["qa_answer"] == "基于视频片段的回答"
        assert result["qa_chunks"] == ["片段A", "片段B"]
        assert any(t.agent_name == "qa" for t in result["traces"])

    @pytest.mark.asyncio
    async def test_qa_node_failure_sets_defaults(self):
        """QA Agent 抛异常 → qa_found=False，记录 failed trace"""
        from biliagent.graph.workflow import qa_node

        mock_platform = AsyncMock()

        state: AgentState = {
            "mention": _make_mention(content="xxx"),
            "video_id": "BV1test123",
            "question": "xxx",
            "traces": [],
        }

        with patch("biliagent.graph.workflow.QAAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
            mock_agent_cls.return_value = mock_agent

            result = await qa_node(state, mock_platform)

        assert result["qa_found"] is False
        assert result["qa_answer"] is None
        qa_trace = next(t for t in result["traces"] if t.agent_name == "qa")
        assert qa_trace.status == "failed"


class TestReplyNodeQA:
    """测试 Reply 节点的 QA 分支"""

    @pytest.mark.asyncio
    async def test_reply_node_qa_found(self):
        """route=ask + qa_found=True → Reply 走 QA 分支，评论含 QA header"""
        from biliagent.graph.workflow import reply_node

        mock_platform = AsyncMock()
        mock_platform.post_comment.return_value = "rpid_qa_1"

        state: AgentState = {
            "mention": _make_mention(user_name="ruan"),
            "video_id": "BV1test123",
            "video_info": VideoInfo(video_id="BV1test123", title="T", description=""),
            "route": "ask",
            "qa_answer": "视频里说过 XYZ。",
            "qa_found": True,
            "qa_chunks": ["片段1"],
            "traces": [],
        }

        result = await reply_node(state, mock_platform)

        assert result["success"] is True
        # 发出的评论应包含 QA 找到答案的头部、@用户名、回答正文
        posted_text = mock_platform.post_comment.await_args.args[1]
        assert "@ruan" in posted_text
        assert "XYZ" in posted_text

    @pytest.mark.asyncio
    async def test_reply_node_qa_missing(self):
        """route=ask + qa_found=False → 使用 "未找到" 话术"""
        from biliagent.graph.workflow import reply_node

        mock_platform = AsyncMock()
        mock_platform.post_comment.return_value = "rpid_qa_2"

        state: AgentState = {
            "mention": _make_mention(user_name="ruan"),
            "video_id": "BV1test123",
            "video_info": VideoInfo(video_id="BV1test123", title="T", description=""),
            "route": "ask",
            "qa_answer": "视频中未详细提及该内容。",
            "qa_found": False,
            "qa_chunks": [],
            "traces": [],
        }

        result = await reply_node(state, mock_platform)

        assert result["success"] is True
        posted_text = mock_platform.post_comment.await_args.args[1]
        assert "没怎么提到" in posted_text or "未详细提及" in posted_text
