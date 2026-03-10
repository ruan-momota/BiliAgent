"""BiliAgent — FastAPI 应用入口"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from biliagent.api.routes import router
from biliagent.graph.workflow import build_workflow
from biliagent.models.schemas import MentionInfo
from biliagent.platforms.bilibili.client import BilibiliPlatform
from biliagent.platforms.bilibili.monitor import MentionMonitor
from biliagent.storage.database import (
    AgentTrace,
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


async def handle_mention(mention: MentionInfo) -> None:
    """Monitor 检测到新@消息时的回调：创建任务记录 → 执行工作流 → 保存结果"""
    logger.info(
        "Processing mention: id=%s, video=%s, user=%s",
        mention.mention_id, mention.video_id, mention.user_name,
    )

    # 1. 创建任务记录
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

    # 2. 执行 LangGraph 工作流
    try:
        initial_state = {"mention": mention, "traces": []}
        result = await workflow.ainvoke(initial_state)

        success = result.get("success", False)
        error = result.get("error")
        status = "completed" if success else "failed"

        logger.info("Workflow finished for task %d: status=%s", task_id, status)
    except Exception as e:
        result = {"traces": []}
        status = "failed"
        error = str(e)
        logger.exception("Workflow failed for task %d", task_id)

    # 3. 更新任务状态 + 保存 traces
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

            await session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建表 + 启动 Monitor"""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")

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

app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "biliagent",
        "monitor_running": monitor.is_running,
    }
