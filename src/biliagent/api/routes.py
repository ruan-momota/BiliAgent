"""FastAPI 路由 — 任务管理、统计、摘要缓存、手动触发"""

import datetime
import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from biliagent.models.schemas import (
    AgentTraceInfo,
    CommentResponse,
    MentionInfo,
    StatsResponse,
    SummaryResponse,
    TaskDetailResponse,
    TaskResponse,
)
from biliagent.storage.cache import delete_summary, list_summaries
from biliagent.storage.database import (
    AgentTrace,
    Comment,
    Summary,
    Task,
    async_session,
)

logger = logging.getLogger("biliagent.api")
router = APIRouter()


# ---- 任务列表 ----
@router.get("/tasks", response_model=list[TaskResponse])
async def get_tasks(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
):
    """获取任务列表"""
    async with async_session() as session:
        stmt = select(Task).order_by(Task.created_at.desc())
        if status:
            stmt = stmt.where(Task.status == status)
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        tasks = result.scalars().all()
        return [TaskResponse.model_validate(t) for t in tasks]


# ---- 任务详情（含完整追溯链路） ----
@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task_detail(task_id: int):
    """获取任务详情：含 Agent 执行追溯、评论记录、摘要"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # 加载关联的 traces
        traces_stmt = (
            select(AgentTrace)
            .where(AgentTrace.task_id == task_id)
            .order_by(AgentTrace.created_at)
        )
        traces_result = await session.execute(traces_stmt)
        traces = [
            AgentTraceInfo(
                agent_name=t.agent_name,
                input_data=t.input_data,
                output_data=t.output_data,
                duration_ms=t.duration_ms,
                status=t.status,
                error_message=t.error_message,
            )
            for t in traces_result.scalars().all()
        ]

        # 加载关联的评论
        comments_stmt = (
            select(Comment)
            .where(Comment.task_id == task_id)
            .order_by(Comment.floor_number)
        )
        comments_result = await session.execute(comments_stmt)
        comments = [
            CommentResponse.model_validate(c) for c in comments_result.scalars().all()
        ]

        # 查找对应的摘要缓存
        summary_stmt = select(Summary).where(
            Summary.platform == task.platform,
            Summary.video_id == task.video_id,
        )
        summary_result = await session.execute(summary_stmt)
        summary_row = summary_result.scalar_one_or_none()
        summary = SummaryResponse.model_validate(summary_row) if summary_row else None

        return TaskDetailResponse(
            id=task.id,
            platform=task.platform,
            video_id=task.video_id,
            user_name=task.user_name,
            status=task.status,
            error_message=task.error_message,
            created_at=task.created_at,
            updated_at=task.updated_at,
            traces=traces,
            comments=comments,
            summary=summary,
        )


# ---- 统计概览 ----
@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """统计概览"""
    async with async_session() as session:
        total = (await session.execute(select(func.count(Task.id)))).scalar() or 0
        completed = (
            await session.execute(
                select(func.count(Task.id)).where(Task.status == "completed")
            )
        ).scalar() or 0
        failed = (
            await session.execute(
                select(func.count(Task.id)).where(Task.status == "failed")
            )
        ).scalar() or 0
        summaries = (
            await session.execute(select(func.count(Summary.id)))
        ).scalar() or 0

        not_follower = (
            await session.execute(
                select(func.count(Task.id)).where(Task.status == "not_follower")
            )
        ).scalar() or 0

        today_start = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today = (
            await session.execute(
                select(func.count(Task.id)).where(Task.created_at >= today_start)
            )
        ).scalar() or 0

        # 获取 Cookie 状态（延迟导入避免循环依赖）
        from biliagent.main import platform as bili_platform
        credential_valid = bili_platform.credential_valid

        return StatsResponse(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            not_follower_tasks=not_follower,
            success_rate=round(completed / total, 4) if total > 0 else 0.0,
            total_summaries=summaries,
            today_tasks=today,
            credential_valid=credential_valid,
        )


# ---- 摘要缓存管理 ----
@router.get("/summaries", response_model=list[SummaryResponse])
async def get_summaries(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """列出摘要缓存"""
    rows = await list_summaries(limit=limit, offset=offset)
    return [SummaryResponse.model_validate(r) for r in rows]


@router.delete("/summaries/{summary_id}")
async def remove_summary(summary_id: int):
    """删除指定摘要缓存"""
    ok = await delete_summary(summary_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"deleted": True}


# ---- 手动触发（开发测试用） ----
class TestTriggerRequest(BaseModel):
    video_id: str
    content: str = "请帮我总结一下这个视频"
    user_name: str = "test_user"


@router.post("/test/trigger")
async def test_trigger(req: TestTriggerRequest):
    """手动触发工作流（不依赖 B站@消息，用于开发测试）"""
    # 延迟导入避免循环依赖
    from biliagent.main import handle_mention

    mention = MentionInfo(
        mention_id=f"test_{uuid.uuid4().hex[:8]}",
        video_id=req.video_id,
        user_id="0",
        user_name=req.user_name,
        content=req.content,
        platform="bilibili",
    )

    logger.info("Manual trigger: video=%s", req.video_id)
    await handle_mention(mention)
    return {"triggered": True, "mention_id": mention.mention_id}
