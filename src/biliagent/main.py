"""BiliAgent — FastAPI 应用入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError

from biliagent.api.routes import router
from biliagent.config import settings
from biliagent.graph.workflow import build_workflow
from biliagent.models.schemas import MentionInfo
from biliagent.platforms.bilibili.client import BilibiliPlatform
from biliagent.platforms.bilibili.monitor import MentionMonitor
from sqlalchemy import select

from biliagent.storage.database import (
    AgentTrace,
    Comment,
    Task,
    async_session,
    init_db,
)

# 日志配置（输出英文）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("biliagent")

# 全局实例
platform = BilibiliPlatform()
monitor = MentionMonitor(platform)
workflow = build_workflow(platform)


async def handle_mention(mention: MentionInfo) -> bool:
    """Monitor 检测到新@消息时的回调：创建任务记录 → 执行工作流 → 保存结果

    Returns:
        True 表示需要稍后重试（如用户未关注），False 表示已处理完毕。
    """
    logger.info(
        "Processing mention: id=%s, video=%s, user=%s",
        mention.mention_id, mention.video_id, mention.user_name,
    )

    # 1. 创建任务记录（重复 mention_id → 检查是否为 not_follower 可重试）
    is_retry = False
    try:
        async with async_session() as session:
            task = Task(
                platform=mention.platform,
                mention_id=mention.mention_id,
                video_id=mention.video_id,
                user_id=mention.user_id,
                user_name=mention.user_name,
                status="processing",
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            task_id = task.id
    except IntegrityError:
        # 如果之前因未关注被拒绝，允许重新检查
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.mention_id == mention.mention_id)
            )
            existing_task = result.scalar_one_or_none()
            if existing_task and existing_task.status == "not_follower":
                logger.info(
                    "Re-evaluating mention %s (previously not_follower)",
                    mention.mention_id,
                )
                task_id = existing_task.id
                is_retry = True
            else:
                logger.info("Mention %s already processed, skipping", mention.mention_id)
                return False

    # 2. 关注检查：未关注用户直接回复提示，不进入工作流
    if settings.app.follower_check_enabled:
        is_follower = await platform.check_is_follower(mention.user_id)
        if not is_follower:
            if not is_retry:
                # 首次：发送求关注提示
                logger.info(
                    "User %s (%s) is not a follower, skipping workflow",
                    mention.user_id, mention.user_name,
                )
                at_prefix = f"@{mention.user_name} " if mention.user_name else ""
                reply_text = f"{at_prefix}{settings.app.not_follower_reply}"
                comment_id = await platform.post_comment(mention.video_id, reply_text)
                async with async_session() as session:
                    task = await session.get(Task, task_id)
                    if task:
                        task.status = "not_follower"
                        if comment_id:
                            session.add(Comment(
                                task_id=task_id,
                                platform=mention.platform,
                                comment_id=comment_id,
                                content=reply_text,
                                floor_number=1,
                            ))
                        await session.commit()
            else:
                # 重试但仍未关注，静默等待下次轮询
                logger.debug(
                    "User %s still not a follower, will retry later",
                    mention.user_id,
                )
            return True  # 告知 Monitor 需要重试

        # 如果是重试且用户已关注，更新任务状态为 processing
        if is_retry:
            logger.info(
                "User %s has followed! Re-processing mention %s",
                mention.user_id, mention.mention_id,
            )
            async with async_session() as session:
                task = await session.get(Task, task_id)
                if task:
                    task.status = "processing"
                    await session.commit()

    # 3. 执行 LangGraph 工作流
    try:
        initial_state = {"mention": mention, "traces": []}
        result = await workflow.ainvoke(initial_state)

        route = result.get("route", "")
        if route == "ignore":
            # 非总结请求，标记为忽略（不算失败）
            status = "completed"
            error = result.get("error")
        else:
            success = result.get("success", False)
            error = result.get("error")
            status = "completed" if success else "failed"

        logger.info("Workflow finished for task %d: status=%s", task_id, status)
    except Exception as e:
        result = {"traces": []}
        status = "failed"
        error = str(e)
        logger.exception("Workflow failed for task %d", task_id)

    # 4. 更新任务状态 + 保存 traces
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task:
            task.status = status
            task.error_message = error

            # 保存 Agent 执行追溯
            for trace_info in result.get("traces", []):
                trace = AgentTrace(
                    task_id=task_id,
                    agent_name=trace_info.agent_name,
                    input_data=trace_info.input_data,
                    output_data=trace_info.output_data,
                    duration_ms=trace_info.duration_ms,
                    status=trace_info.status,
                    error_message=trace_info.error_message,
                )
                session.add(trace)

            # 保存评论记录
            reply_parts = result.get("reply_parts", [])
            comment_ids = result.get("comment_ids", [])
            for i, part in enumerate(reply_parts):
                comment = Comment(
                    task_id=task_id,
                    platform=mention.platform,
                    comment_id=comment_ids[i] if i < len(comment_ids) else None,
                    content=part,
                    floor_number=i + 1,
                )
                session.add(comment)

            await session.commit()

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表 + 启动 Monitor"""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")

    # 启动时检测 Cookie 有效性
    credential_ok = await platform.check_credential()
    if credential_ok:
        logger.info("Bilibili credential check passed.")
    else:
        logger.warning("Bilibili credential may be invalid! Check your Cookie.")

    # 从数据库加载已处理的 mention_id，防止重启后重复处理
    # 排除 not_follower 状态的任务，使其可被重新检查
    async with async_session() as session:
        result = await session.execute(
            select(Task.mention_id).where(Task.status != "not_follower")
        )
        existing_ids = {row[0] for row in result.all()}
    if existing_ids:
        monitor.mark_processed(existing_ids)
        logger.info("Loaded %d processed mention IDs from database.", len(existing_ids))

    # 启动 Monitor 轮询
    monitor.set_callback(handle_mention)
    monitor.start()
    logger.info("BiliAgent is ready.")

    yield

    # 关闭 Monitor
    monitor.stop()
    logger.info("BiliAgent shutting down.")


app = FastAPI(
    title="BiliAgent",
    description="Multi-agent Bilibili video summarization bot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.app.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    credential_status = platform.credential_valid
    return {
        "status": "ok",
        "service": "biliagent",
        "monitor_running": monitor.is_running,
        "credential_valid": credential_status,
    }
