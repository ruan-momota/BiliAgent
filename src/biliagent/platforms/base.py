"""平台抽象基类 — 所有平台实现此接口，支持平台可扩展"""

from abc import ABC, abstractmethod

from biliagent.models.schemas import MentionInfo, VideoInfo


class PlatformBase(ABC):
    """平台抽象接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """平台名称标识，如 'bilibili', 'xiaohongshu'"""
        ...

    @abstractmethod
    async def get_mentions(self, last_id: str | None = None) -> list[MentionInfo]:
        """获取@消息通知列表

        Args:
            last_id: 上次拉取的最新消息ID，用于增量拉取
        Returns:
            新的@消息列表
        """
        ...

    @abstractmethod
    async def get_video_info(self, video_id: str) -> VideoInfo:
        """获取视频元数据（标题、简介）

        Args:
            video_id: 视频唯一标识（如B站BV号）
        """
        ...

    @abstractmethod
    async def get_subtitles(self, video_id: str) -> str | None:
        """获取视频字幕文本

        Args:
            video_id: 视频唯一标识
        Returns:
            字幕文本（纯文本拼接），无字幕时返回 None
        """
        ...

    @abstractmethod
    async def get_audio_url(self, video_id: str) -> str | None:
        """获取视频音频流 URL（供语音转文字降级使用）

        Args:
            video_id: 视频唯一标识
        Returns:
            音频流 URL，无法获取时返回 None
        """
        ...

    @abstractmethod
    async def post_comment(self, video_id: str, text: str) -> str | None:
        """在视频下发布一级评论

        Args:
            video_id: 视频唯一标识
            text: 评论内容
        Returns:
            评论ID（平台返回），失败返回 None
        """
        ...

    @abstractmethod
    async def reply_comment(
        self, video_id: str, root_comment_id: str, text: str
    ) -> str | None:
        """回复指定评论（盖楼）

        Args:
            video_id: 视频唯一标识
            root_comment_id: 被回复的根评论ID
            text: 回复内容
        Returns:
            回复评论ID，失败返回 None
        """
        ...

    @abstractmethod
    async def check_is_follower(self, user_id: str) -> bool:
        """检查指定用户是否关注了本账号

        Args:
            user_id: 待检查的用户ID
        Returns:
            True 表示该用户已关注本账号
        """
        ...
