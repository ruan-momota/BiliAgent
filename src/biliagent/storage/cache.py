"""摘要缓存操作 — 增删查"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from biliagent.storage.database import Summary, async_session

logger = logging.getLogger("biliagent.cache")


async def get_cached_summary(
    platform: str, video_id: str
) -> Summary | None:
    """查询缓存：同平台同视频是否已有摘要"""
    async with async_session() as session:
        stmt = select(Summary).where(
            Summary.platform == platform,
            Summary.video_id == video_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def save_summary(
    platform: str,
    video_id: str,
    video_title: str | None,
    summary_text: str,
    has_subtitles: bool = True,
) -> Summary:
    """保存摘要到缓存"""
    async with async_session() as session:
        summary = Summary(
            platform=platform,
            video_id=video_id,
            video_title=video_title,
            summary_text=summary_text,
            has_subtitles=has_subtitles,
        )
        session.add(summary)
        await session.commit()
        await session.refresh(summary)
        logger.info("Summary cached for %s:%s", platform, video_id)
        return summary


async def delete_summary(summary_id: int) -> bool:
    """删除指定缓存"""
    async with async_session() as session:
        stmt = select(Summary).where(Summary.id == summary_id)
        result = await session.execute(stmt)
        summary = result.scalar_one_or_none()
        if summary:
            await session.delete(summary)
            await session.commit()
            logger.info("Summary %d deleted", summary_id)
            return True
        return False


async def list_summaries(
    platform: str | None = None, limit: int = 50, offset: int = 0
) -> list[Summary]:
    """列出缓存摘要"""
    async with async_session() as session:
        stmt = select(Summary).order_by(Summary.created_at.desc())
        if platform:
            stmt = stmt.where(Summary.platform == platform)
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        return list(result.scalars().all())
