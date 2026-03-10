"""FastAPI 路由 — Phase 1 先提供基础骨架"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/tasks")
async def list_tasks():
    """任务列表（Phase 5 实现）"""
    return []


@router.get("/stats")
async def get_stats():
    """统计概览（Phase 5 实现）"""
    return {
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "success_rate": 0.0,
        "total_summaries": 0,
        "today_tasks": 0,
    }
