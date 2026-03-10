"""B站 @消息轮询服务 — 基于 asyncio 循环"""

import asyncio
import logging

from biliagent.config import settings
from biliagent.models.schemas import MentionInfo
from biliagent.platforms.bilibili.client import BilibiliPlatform

logger = logging.getLogger("biliagent.monitor")


class MentionMonitor:
    """轮询 B站@消息，检测到新消息时调用回调"""

    def __init__(
        self,
        platform: BilibiliPlatform,
        on_mention: "asyncio.coroutines | None" = None,
    ) -> None:
        self._platform = platform
        self._on_mention = on_mention  # 回调: async def callback(mention: MentionInfo)
        self._interval = settings.app.monitor_interval
        self._last_id: str | None = None  # 上次拉取的最新消息ID
        self._running = False
        self._task: asyncio.Task | None = None
        # 已处理的 mention_id 集合（防重复触发）
        self._processed_ids: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """启动轮询（在当前事件循环中创建后台任务）"""
        if self._running:
            logger.warning("Monitor is already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Monitor started, polling every %ds", self._interval)

    def stop(self) -> None:
        """停止轮询"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Monitor stopped")

    def set_callback(self, callback) -> None:
        """设置新消息回调"""
        self._on_mention = callback

    async def _poll_loop(self) -> None:
        """轮询主循环"""
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in monitor poll loop")
            await asyncio.sleep(self._interval)

    async def _poll_once(self) -> None:
        """单次轮询"""
        mentions = await self._platform.get_mentions(last_id=self._last_id)

        if not mentions:
            logger.debug("No new mentions found")
            return

        logger.info("Found %d new mention(s)", len(mentions))

        for mention in mentions:
            # 跳过已处理的
            if mention.mention_id in self._processed_ids:
                continue

            self._processed_ids.add(mention.mention_id)
            logger.info(
                "New mention: id=%s, video=%s, user=%s",
                mention.mention_id,
                mention.video_id,
                mention.user_name,
            )

            # 触发回调
            if self._on_mention is not None:
                try:
                    await self._on_mention(mention)
                except Exception:
                    logger.exception(
                        "Error processing mention %s", mention.mention_id
                    )

        # 更新 last_id 为最新的一条
        self._last_id = mentions[0].mention_id
