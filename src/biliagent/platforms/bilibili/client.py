"""B站平台实现 — 封装 bilibili-api-python"""

import logging

import httpx
from bilibili_api import Credential
from bilibili_api.comment import CommentResourceType, send_comment
from bilibili_api.session import get_at
from bilibili_api.user import User
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
        # Cookie 有效性状态（供健康检查使用）
        self._credential_valid: bool | None = None  # None = 未检测

    @property
    def name(self) -> str:
        return "bilibili"

    @property
    def credential_valid(self) -> bool | None:
        """Cookie 是否有效（None 表示尚未检测）"""
        return self._credential_valid

    # ---- Cookie 有效性检测 ----
    async def check_credential(self) -> bool:
        """检测 Cookie 是否仍然有效（调用B站导航接口）"""
        try:
            result = await self._credential.check_valid()
            self._credential_valid = result
            if not result:
                logger.warning("Bilibili credential is INVALID (expired or revoked)")
            else:
                logger.debug("Bilibili credential check passed")
            return result
        except Exception:
            # check_valid 方法不可用时，尝试用 get_at 做轻量检测
            try:
                await get_at(credential=self._credential)
                self._credential_valid = True
                return True
            except Exception as e:
                error_str = str(e).lower()
                if "login" in error_str or "expire" in error_str or "-101" in error_str:
                    self._credential_valid = False
                    logger.warning("Bilibili credential expired: %s", e)
                    return False
                # 其它错误（如网络问题）不改变状态
                logger.warning("Credential check inconclusive: %s", e)
                return self._credential_valid is not False

    # ---- @消息 ----
    async def get_mentions(self, last_id: str | None = None) -> list[MentionInfo]:
        """获取@我的消息列表"""
        try:
            params: dict = {}
            if last_id is not None:
                params["last_id"] = int(last_id)

            result = await get_at(credential=self._credential, **params)
            items = result.get("items", [])

            # 能拿到结果说明 Cookie 有效
            self._credential_valid = True

            mentions: list[MentionInfo] = []
            for item in items:
                mention = self._parse_mention(item)
                if mention is not None:
                    mentions.append(mention)
            return mentions
        except Exception as e:
            self._detect_credential_error(e)
            logger.exception("Failed to fetch mentions")
            return []

    def _parse_mention(self, item: dict) -> MentionInfo | None:
        """解析单条@消息，提取视频ID和用户信息

        B站 get_at() 返回结构：
        {
            "id": 123, "user": {"mid": ..., "nickname": ...},
            "item": {"uri": "...", "subject_id": ..., "source_content": ...}
        }
        注意：uri/subject_id/source_content 等字段嵌套在 item.item 子对象中。
        """
        try:
            item_id = str(item.get("id", ""))
            user = item.get("user", {})
            user_id = str(user.get("mid", ""))
            user_name = user.get("nickname", "")

            # B站 at 通知的详情字段在嵌套的 "item" 子对象中
            detail = item.get("item", {})

            logger.debug(
                "Parsing mention %s: detail keys=%s", item_id, list(detail.keys())
            )

            subject_id = str(detail.get("subject_id", ""))
            uri = detail.get("uri", "")
            native_uri = detail.get("native_uri", "")

            # 从 URI 中提取 BV 号（如 //www.bilibili.com/video/BV1xxxxx）
            video_id = self._extract_bvid(uri) or self._extract_bvid(native_uri)

            # 如果无法从 URI 获取 BV 号，尝试用 subject_id 作为 aid
            if not video_id and subject_id and subject_id != "0":
                video_id = f"aid:{subject_id}"

            if not video_id:
                logger.warning(
                    "Cannot extract video_id from mention %s, uri=%s, native_uri=%s, subject_id=%s",
                    item_id, uri, native_uri, subject_id,
                )
                return None

            content = detail.get("source_content", "")

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
                resp = await client.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()

            # B站字幕 JSON 格式: {"body": [{"content": "xxx", "from": 0.0, "to": 1.0}, ...]}
            body = data.get("body", [])
            lines = [item.get("content", "") for item in body if item.get("content")]
            return "\n".join(lines) if lines else None
        except httpx.TimeoutException:
            logger.error("Timeout fetching subtitle from %s", url)
            return None
        except Exception:
            logger.exception("Failed to fetch subtitle from %s", url)
            return None

    # ---- 音频 ----
    async def get_audio_url(self, video_id: str) -> str | None:
        """获取视频音频流 URL（DASH 格式最高码率音频）"""
        try:
            video = self._make_video(video_id)
            download_info = await video.get_download_url(page_index=0)
            dash = download_info.get("dash")
            if not dash:
                logger.info("No DASH info for video %s", video_id)
                return None

            audio_list = dash.get("audio") or []
            if not audio_list:
                logger.info("No audio streams for video %s", video_id)
                return None

            # 按码率降序，取最高码率的音频流
            best = max(audio_list, key=lambda a: a.get("bandwidth", 0))
            url = best.get("baseUrl") or best.get("base_url", "")
            if url:
                logger.info("Audio URL obtained for video %s, bandwidth=%d", video_id, best.get("bandwidth", 0))
            return url or None

        except Exception:
            logger.exception("Failed to get audio URL for video %s", video_id)
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
        except Exception as e:
            self._detect_credential_error(e)
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
        except Exception as e:
            self._detect_credential_error(e)
            logger.exception("Failed to reply comment on video %s", video_id)
            return None

    # ---- 关注关系检查 ----
    async def check_is_follower(self, user_id: str) -> bool:
        """检查指定用户是否关注了本账号

        通过 B站用户关系 API 查询，be_relation.attribute 含义：
        0=未关注, 1=悄悄关注, 2=已关注, 6=互相关注, 128=拉黑
        attribute 为 1/2/6 时视为已关注。
        """
        try:
            user = User(uid=int(user_id), credential=self._credential)
            relation = await user.get_relation()
            be_relation = relation.get("be_relation", {})
            attribute = be_relation.get("attribute", 0)
            is_follower = attribute in (1, 2, 6)
            logger.info(
                "Follower check: user=%s, attribute=%s, is_follower=%s",
                user_id, attribute, is_follower,
            )
            return is_follower
        except Exception as e:
            self._detect_credential_error(e)
            logger.exception("Failed to check follower status for user %s", user_id)
            # 查询失败时默认放行，避免误拦截
            return True

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

    def _detect_credential_error(self, e: Exception) -> None:
        """从异常信息中检测 Cookie 是否已过期"""
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["login", "expire", "-101", "credential", "csrf"]):
            self._credential_valid = False
            logger.error("Credential appears expired or invalid: %s", e)
