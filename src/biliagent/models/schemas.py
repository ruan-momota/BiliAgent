"""Pydantic 数据模型 — 用于 Agent 间传递数据和 API 响应"""

import datetime

from pydantic import BaseModel


# ---- @消息相关 ----
class MentionInfo(BaseModel):
    """从平台获取的 @消息"""
    mention_id: str
    video_id: str
    user_id: str
    user_name: str | None = None
    content: str  # @消息原文
    platform: str = "bilibili"


# ---- 视频相关 ----
class VideoInfo(BaseModel):
    """视频元数据"""
    video_id: str
    title: str
    description: str = ""
    platform: str = "bilibili"


# ---- Agent 追溯相关 ----
class AgentTraceInfo(BaseModel):
    """单个 Agent 节点执行记录"""
    agent_name: str
    input_data: str | None = None  # JSON 字符串
    output_data: str | None = None  # JSON 字符串
    duration_ms: int | None = None
    status: str = "success"
    error_message: str | None = None


# ---- 鉴别相关 ----
class VerificationInfo(BaseModel):
    """鉴别结果数据"""
    verification: str       # 鉴别评论内容
    opinion: str = "neutral"  # agree/disagree/doubt/neutral
    sources: list[str] = []   # 联网搜索引用来源


# ---- API 响应模型 ----
class TaskResponse(BaseModel):
    """任务列表 API 响应"""
    id: int
    platform: str
    video_id: str
    user_name: str | None
    status: str
    error_message: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


class TaskDetailResponse(TaskResponse):
    """任务详情 API 响应（含追溯信息）"""
    traces: list[AgentTraceInfo] = []
    comments: list["CommentResponse"] = []
    summary: "SummaryResponse | None" = None


class CommentResponse(BaseModel):
    """评论记录 API 响应"""
    id: int
    content: str
    floor_number: int
    posted_at: datetime.datetime

    model_config = {"from_attributes": True}


class SummaryResponse(BaseModel):
    """摘要缓存 API 响应"""
    id: int
    video_id: str
    video_title: str | None
    summary_text: str
    has_subtitles: bool
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class VerificationResponse(BaseModel):
    """鉴别缓存 API 响应"""
    id: int
    video_id: str
    video_title: str | None
    question: str
    verification: str
    opinion: str | None
    sources: str | None  # JSON string
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class StatsResponse(BaseModel):
    """统计概览 API 响应"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    not_follower_tasks: int = 0
    success_rate: float
    total_summaries: int
    total_verifications: int = 0
    today_tasks: int
    credential_valid: bool | None = None  # B站 Cookie 状态：True/False/None(未检测)
