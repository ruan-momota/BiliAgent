"""LangGraph 工作流状态定义"""

from typing import TypedDict

from biliagent.models.schemas import AgentTraceInfo, MentionInfo, VideoInfo


class AgentState(TypedDict, total=False):
    """工作流在各节点间传递的状态"""

    # ---- 输入 ----
    mention: MentionInfo          # @消息信息

    # ---- Supervisor 输出 ----
    video_id: str                 # 视频ID
    route: str                    # 路由决策: use_cache / analyze / verify / ignore
    cached_summary: str | None    # 缓存命中时的摘要
    question: str | None          # verify 意图时的用户提问

    # ---- Analyzer 输出 ----
    video_info: VideoInfo | None  # 视频元数据
    subtitles: str | None         # 字幕文本
    has_subtitles: bool           # 是否有字幕
    can_summarize: bool           # 是否可总结

    # ---- Summarizer 输出 ----
    summary: str | None           # 生成的摘要

    # ---- VerifyCacheJudge 输出 ----
    verify_route: str | None      # use_verify_cache / verify
    cached_verification: str | None  # 缓存命中时的鉴别内容
    cached_opinion: str | None    # 缓存命中时的观点
    cached_sources: str | None    # 缓存命中时的来源

    # ---- Verifier 输出 ----
    verification: str | None      # 生成的鉴别评论
    opinion: str | None           # 观点标签: agree/disagree/doubt/neutral
    sources: list[str]            # 引用来源

    # ---- Reply 输出 ----
    reply_parts: list[str]        # 评论内容（可能多条/盖楼）
    comment_ids: list[str | None] # 平台返回的评论ID
    success: bool                 # 发布是否成功

    # ---- 通用 ----
    error: str | None             # 错误信息
    traces: list[AgentTraceInfo]  # 各 Agent 节点的执行记录（供追溯）
