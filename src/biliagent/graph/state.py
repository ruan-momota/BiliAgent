"""LangGraph 工作流状态定义"""

from typing import TypedDict

from biliagent.models.schemas import AgentTraceInfo, MentionInfo, VideoInfo


class AgentState(TypedDict, total=False):
    """工作流在各节点间传递的状态"""

    # ---- 输入 ----
    mention: MentionInfo          # @消息信息

    # ---- Supervisor 输出 ----
    video_id: str                 # 视频ID
    route: str                    # 路由决策: use_cache / analyze / ignore
    cached_summary: str | None    # 缓存命中时的摘要

    # ---- Analyzer 输出 ----
    video_info: VideoInfo | None  # 视频元数据
    subtitles: str | None         # 字幕文本
    has_subtitles: bool           # 是否有字幕
    can_summarize: bool           # 是否可总结

    # ---- Summarizer 输出 ----
    summary: str | None           # 生成的摘要

    # ---- Reply 输出 ----
    reply_parts: list[str]        # 评论内容（可能多条/盖楼）
    comment_ids: list[str | None] # 平台返回的评论ID
    success: bool                 # 发布是否成功

    # ---- 通用 ----
    error: str | None             # 错误信息
    traces: list[AgentTraceInfo]  # 各 Agent 节点的执行记录（供追溯）
