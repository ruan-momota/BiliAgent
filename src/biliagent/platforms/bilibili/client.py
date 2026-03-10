"""B站平台实现 — 封装 bilibili-api-python"""

import logging

import httpx
from bilibili_api import Credential
from bilibili_api.comment import CommentResourceType, send_comment
from bilibili_api.session import get_at
from bilibili_api.video import Video

from biliagent.config import settings
from biliagent.models.schemas import MentionInfo, VideoInfo
from biliagent.platforms.base import PlatformBase

logger = logging.getLogger("biliagent.bilibili")


class BilibiliPlatform(PlatformBase):
    """B站平台接口实现"""

    def __init__(self) -> None:
        self._credential = Credential(
            sessdata=settings.bili.sessdata,
            bili_jct=settings.bili.bili_jct,
            buvid3=settings.bili.buvid3,
        )

    @property
    def name(self) -> str:
        return "bilibili"

    # ---- @消息 ----
    async def get_mentions(self, last_id: str | None = None) -> list[MentionInfo]:
        """获取@我的消息列表"""
        try:
            params: dict = {}
            if last_id is not None:
                params["last_id"] = int(last_id)

            result = await get_at(credential=self._credential, **params)
            items = result.get("items", [])

            mentions: list[MentionInfo] = []
            for item in items:
                mention = self._parse_mention(item)
                if mention is not None:
                    mentions.append(mention)
            return mentions
        except Exception:
            logger.exception("Failed to fetch mentions")
            return []

    def _parse_mention(self, item: dict) -> MentionInfo | None:
        """解析单条@消息，提取视频ID和用户信息"""
        try:
            item_id = str(item.get("id", ""))
            user = item.get("user", {})
            user_id = str(user.get("mid", ""))
            user_name = user.get("nickname", "")

            # 从 item 中提取被@的视频信息
            # at 消息的 item.source_content 包含原始评论内容
            # item.uri 或 item.native_uri 包含视频链接
            source_id = str(item.get("source_id", ""))
            subject_id = str(item.get("subject_id", ""))

            # 尝试从 item 中获取视频 BV 号
            # at 消息中 item_type 为 "video" 时 subject_id 是 aid
            uri = item.get("uri", "")
            native_uri = item.get("native_uri", "")

            # 从 URI 中提取 BV 号（如 //www.bilibili.com/video/BV1xxxxx）
            video_id = self._extract_bvid(uri) or self._extract_bvid(native_uri)

            # 如果无法从 URI 获取 BV 号，尝试用 subject_id 作为 aid
            if not video_id and subject_id:
                video_id = f"aid:{subject_id}"

            if not video_id:
                logger.warning("Cannot extract video_id from mention %s", item_id)
                return None

            content = item.get("source_content", "")

            return MentionInfo(
                mention_id=item_id,
                video_id=video_id,
                user_id=user_id,
                user_name=user_name,
                content=content,
                platform="bilibili",
            )
        except Exception:
            logger.exception("Failed to parse mention item")
            return None

    @staticmethod
    def _extract_bvid(uri: str) -> str | None:
        """从 URL 中提取 BV 号"""
        if not uri:
            return None
        # 匹配 /video/BVxxxxxx 格式
        parts = uri.split("/video/")
        if len(parts) >= 2:
            bvid = parts[1].split("/")[0].split("?")[0]
            if bvid.startswith("BV"):
                return bvid
        return None

    # ---- 视频信息 ----
    async def get_video_info(self, video_id: str) -> VideoInfo:
        """获取视频标题和简介"""
        video = self._make_video(video_id)
        info = await video.get_info()
        return VideoInfo(
            video_id=video_id,
            title=info.get("title", ""),
            description=info.get("desc", ""),
            platform="bilibili",
        )

    # ---- 字幕 ----
    async def get_subtitles(self, video_id: str) -> str | None:
        """获取视频字幕文本，无字幕返回 None"""
        try:
            video = self._make_video(video_id)
            info = await video.get_info()
            pages = info.get("pages", [])
            if not pages:
                logger.info("No pages found for video %s", video_id)
                return None

            cid = pages[0]["cid"]
            subtitle_data = await video.get_subtitle(cid=cid)

            subtitle_list = subtitle_data.get("subtitles", [])
            if not subtitle_list:
                logger.info("No subtitles available for video %s", video_id)
                return None

            # 优先选中文字幕，否则取第一个
            subtitle_url = self._pick_subtitle_url(subtitle_list)
            if not subtitle_url:
                return None

            # 下载字幕 JSON 并拼接文本
            text = await self._fetch_subtitle_text(subtitle_url)

            # 截断超长字幕
            max_len = settings.app.subtitle_max_length
            if text and len(text) > max_len:
                text = text[:max_len] + "...(truncated)"
                logger.info("Subtitle truncated to %d chars for video %s", max_len, video_id)

            return text
        except Exception:
            logger.exception("Failed to get subtitles for video %s", video_id)
            return None

    @staticmethod
    def _pick_subtitle_url(subtitle_list: list[dict]) -> str | None:
        """从字幕列表中选择最合适的字幕 URL（优先中文）"""
        # 优先中文
        for sub in subtitle_list:
            lang = sub.get("lan", "")
            if "zh" in lang or "cn" in lang:
                url = sub.get("subtitle_url", "")
                return f"https:{url}" if url.startswith("//") else url

        # 没有中文则取第一个
        if subtitle_list:
            url = subtitle_list[0].get("subtitle_url", "")
            return f"https:{url}" if url.startswith("//") else url
        return None

    @staticmethod
    async def _fetch_subtitle_text(url: str) -> str | None:
        """下载字幕 JSON，拼接成纯文本"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

            # B站字幕 JSON 格式: {"body": [{"content": "xxx", "from": 0.0, "to": 1.0}, ...]}
            body = data.get("body", [])
            lines = [item.get("content", "") for item in body if item.get("content")]
            return "\n".join(lines) if lines else None
        except Exception:
            logger.exception("Failed to fetch subtitle from %s", url)
            return None

    # ---- 评论 ----
    async def post_comment(self, video_id: str, text: str) -> str | None:
        """发布一级评论"""
        try:
            video = self._make_video(video_id)
            aid = video.get_aid()

            # aid 可能在初始化时未解析，需要先获取 info
            if not aid:
                info = await video.get_info()
                aid = info.get("aid")

            result = await send_comment(
                text=text,
                oid=aid,
                type_=CommentResourceType.VIDEO,
                credential=self._credential,
            )
            rpid = result.get("rpid")
            logger.info("Comment posted on video %s, rpid=%s", video_id, rpid)
            return str(rpid) if rpid else None
        except Exception:
            logger.exception("Failed to post comment on video %s", video_id)
            return None

    async def reply_comment(
        self, video_id: str, root_comment_id: str, text: str
    ) -> str | None:
        """回复评论（盖楼）"""
        try:
            video = self._make_video(video_id)
            aid = video.get_aid()

            if not aid:
                info = await video.get_info()
                aid = info.get("aid")

            result = await send_comment(
                text=text,
                oid=aid,
                type_=CommentResourceType.VIDEO,
                root=int(root_comment_id),
                credential=self._credential,
            )
            rpid = result.get("rpid")
            logger.info("Reply posted on video %s under %s, rpid=%s", video_id, root_comment_id, rpid)
            return str(rpid) if rpid else None
        except Exception:
            logger.exception("Failed to reply comment on video %s", video_id)
            return None

    # ---- 内部工具 ----
    def _make_video(self, video_id: str) -> Video:
        """根据视频ID创建 Video 对象（支持 BV 号和 aid）"""
        if video_id.startswith("BV"):
            return Video(bvid=video_id, credential=self._credential)
        elif video_id.startswith("aid:"):
            return Video(aid=int(video_id[4:]), credential=self._credential)
        else:
            # 尝试当作 BV 号处理
            return Video(bvid=video_id, credential=self._credential)
