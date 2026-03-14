"""鉴别缓存操作 — 增删查"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from biliagent.storage.database import Verification, async_session

logger = logging.getLogger("biliagent.verify_cache")


async def get_cached_verifications(
    platform: str, video_id: str
) -> list[Verification]:
    """查询同平台同视频的所有历史鉴别记录"""
    async with async_session() as session:
        stmt = (
            select(Verification)
            .where(
                Verification.platform == platform,
                Verification.video_id == video_id,
            )
            .order_by(Verification.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def get_verification_by_id(verification_id: int) -> Verification | None:
    """按 ID 查询单条鉴别记录"""
    async with async_session() as session:
        return await session.get(Verification, verification_id)


async def save_verification(
    platform: str,
    video_id: str,
    video_title: str | None,
    question: str,
    verification: str,
    opinion: str | None = None,
    sources: str | None = None,
) -> Verification:
    """保存鉴别结果"""
    async with async_session() as session:
        record = Verification(
            platform=platform,
            video_id=video_id,
            video_title=video_title,
            question=question,
            verification=verification,
            opinion=opinion,
            sources=sources,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info("Verification cached for %s:%s", platform, video_id)
        return record


async def delete_verification(verification_id: int) -> bool:
    """删除指定鉴别缓存"""
    async with async_session() as session:
        stmt = select(Verification).where(Verification.id == verification_id)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if record:
            await session.delete(record)
            await session.commit()
            logger.info("Verification %d deleted", verification_id)
            return True
        return False


async def list_verifications(
    platform: str | None = None, limit: int = 50, offset: int = 0
) -> list[Verification]:
    """列出鉴别缓存"""
    async with async_session() as session:
        stmt = select(Verification).order_by(Verification.created_at.desc())
        if platform:
            stmt = stmt.where(Verification.platform == platform)
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        return list(result.scalars().all())
