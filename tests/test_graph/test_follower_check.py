"""关注检查集成测试 — 验证 handle_mention 中的关注拦截逻辑"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biliagent.models.schemas import MentionInfo


def _make_mention(**kwargs) -> MentionInfo:
    defaults = dict(
        mention_id="test_follower_001",
        video_id="BV1test123",
        user_id="uid_001",
        user_name="test_user",
        content="帮我总结一下这个视频",
        platform="bilibili",
    )
    defaults.update(kwargs)
    return MentionInfo(**defaults)


class TestFollowerCheckInHandleMention:
    """测试 handle_mention 中的关注检查拦截"""

    @pytest.mark.asyncio
    @patch("biliagent.main.workflow")
    @patch("biliagent.main.platform")
    @patch("biliagent.main.async_session")
    @patch("biliagent.main.settings")
    async def test_not_follower_skips_workflow(
        self, mock_settings, mock_session_factory, mock_platform, mock_workflow
    ):
        """未关注用户 → 不执行工作流，直接回复提示"""
        from biliagent.main import handle_mention

        # 配置：关注检查开启
        mock_settings.app.follower_check_enabled = True
        mock_settings.app.not_follower_reply = "请先关注哦"

        # 模拟未关注
        mock_platform.check_is_follower = AsyncMock(return_value=False)
        mock_platform.post_comment = AsyncMock(return_value="rpid_100")

        # 模拟数据库会话
        mock_task = MagicMock()
        mock_task.id = 1
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_task
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session

        mention = _make_mention()
        should_retry = await handle_mention(mention)

        # 验证：调用了关注检查
        mock_platform.check_is_follower.assert_called_once_with("uid_001")
        # 验证：发布了提示评论（含 @用户名 前缀）
        mock_platform.post_comment.assert_called_once_with("BV1test123", "@test_user 请先关注哦")
        # 验证：工作流未被调用
        mock_workflow.ainvoke.assert_not_called()
        # 验证：任务状态设为 not_follower
        assert mock_task.status == "not_follower"
        # 验证：返回 True 表示需要重试
        assert should_retry is True

    @pytest.mark.asyncio
    @patch("biliagent.main.workflow")
    @patch("biliagent.main.platform")
    @patch("biliagent.main.async_session")
    @patch("biliagent.main.settings")
    async def test_follower_proceeds_to_workflow(
        self, mock_settings, mock_session_factory, mock_platform, mock_workflow
    ):
        """已关注用户 → 正常进入工作流"""
        from biliagent.main import handle_mention

        mock_settings.app.follower_check_enabled = True

        # 模拟已关注
        mock_platform.check_is_follower = AsyncMock(return_value=True)

        # 模拟数据库会话
        mock_task = MagicMock()
        mock_task.id = 1
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_task
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session

        # 模拟工作流结果
        mock_workflow.ainvoke = AsyncMock(return_value={
            "route": "ignore",
            "traces": [],
        })

        mention = _make_mention()
        should_retry = await handle_mention(mention)

        # 验证：工作流被调用
        mock_workflow.ainvoke.assert_called_once()
        # 验证：返回 False 表示已处理完毕
        assert should_retry is False

    @pytest.mark.asyncio
    @patch("biliagent.main.workflow")
    @patch("biliagent.main.platform")
    @patch("biliagent.main.async_session")
    @patch("biliagent.main.settings")
    async def test_follower_check_disabled_skips_check(
        self, mock_settings, mock_session_factory, mock_platform, mock_workflow
    ):
        """关注检查关闭 → 直接进入工作流，不调用 check_is_follower"""
        from biliagent.main import handle_mention

        mock_settings.app.follower_check_enabled = False

        mock_platform.check_is_follower = AsyncMock()

        # 模拟数据库会话
        mock_task = MagicMock()
        mock_task.id = 1
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_task
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session

        mock_workflow.ainvoke = AsyncMock(return_value={
            "route": "ignore",
            "traces": [],
        })

        mention = _make_mention()
        should_retry = await handle_mention(mention)

        # 验证：未调用关注检查
        mock_platform.check_is_follower.assert_not_called()
        # 验证：工作流被调用
        mock_workflow.ainvoke.assert_called_once()
        # 验证：返回 False 表示已处理完毕
        assert should_retry is False
