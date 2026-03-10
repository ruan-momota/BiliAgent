"""LangGraph 工作流编排 — 把 4 个 Agent 串成完整的处理链"""

import json
import logging
import time

from langgraph.graph import END, START, StateGraph

from biliagent.agents.analyzer import AnalyzerAgent
from biliagent.agents.reply import ReplyAgent
from biliagent.agents.summarizer import SummarizerAgent
from biliagent.agents.supervisor import SupervisorAgent
from biliagent.graph.state import AgentState
from biliagent.models.schemas import AgentTraceInfo
from biliagent.platforms.base import PlatformBase
from biliagent.storage.cache import save_summary

logger = logging.getLogger("biliagent.graph")


def _add_trace(
    state: AgentState,
    agent_name: str,
    input_data: dict | None,
    output_data: dict | None,
    duration_ms: int,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    """往 state.traces 追加一条执行记录"""
    traces = state.get("traces", [])
    traces.append(AgentTraceInfo(
        agent_name=agent_name,
        input_data=json.dumps(input_data, ensure_ascii=False) if input_data else None,
        output_data=json.dumps(output_data, ensure_ascii=False) if output_data else None,
        duration_ms=duration_ms,
        status=status,
        error_message=error_message,
    ))
    state["traces"] = traces


# ---- 节点函数 ----

async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor 节点：判断意图、查缓存、路由"""
    mention = state["mention"]
    t0 = time.perf_counter()

    try:
        agent = SupervisorAgent()
        result = await agent.run(
            content=mention.content,
            user_name=mention.user_name or "",
            video_id=mention.video_id,
            platform=mention.platform,
        )

        state["video_id"] = mention.video_id
        state["route"] = result["route"]
        state["cached_summary"] = result.get("cached_summary")
        if result.get("reason"):
            state["error"] = result["reason"]

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "supervisor",
            {"content": mention.content, "video_id": mention.video_id},
            {"route": result["route"]},
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["route"] = "ignore"
        state["error"] = str(e)
        _add_trace(state, "supervisor", None, None, duration, "failed", str(e))
        logger.exception("Supervisor node failed")

    return state


async def analyzer_node(state: AgentState, platform: PlatformBase) -> AgentState:
    """Analyzer 节点：拉取视频信息和字幕，评估可总结性"""
    video_id = state["video_id"]
    t0 = time.perf_counter()

    try:
        agent = AnalyzerAgent(platform)
        result = await agent.run(video_id)

        state["video_info"] = result["video_info"]
        state["subtitles"] = result.get("subtitles")
        state["has_subtitles"] = result.get("subtitles") is not None
        state["can_summarize"] = result["can_summarize"]
        if not result["can_summarize"]:
            state["error"] = result.get("reason")

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "analyzer",
            {"video_id": video_id},
            {
                "can_summarize": result["can_summarize"],
                "has_subtitles": state["has_subtitles"],
                "title": result["video_info"].title if result["video_info"] else None,
            },
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["can_summarize"] = False
        state["error"] = str(e)
        _add_trace(state, "analyzer", None, None, duration, "failed", str(e))
        logger.exception("Analyzer node failed")

    return state


async def summarizer_node(state: AgentState) -> AgentState:
    """Summarizer 节点：生成摘要"""
    t0 = time.perf_counter()

    try:
        video_info = state.get("video_info")
        title = video_info.title if video_info else ""
        description = video_info.description if video_info else ""
        subtitles = state.get("subtitles", "")

        agent = SummarizerAgent()
        result = await agent.run(title=title, description=description, subtitles=subtitles)

        summary = result["summary"]
        state["summary"] = summary

        # 保存到缓存
        mention = state["mention"]
        await save_summary(
            platform=mention.platform,
            video_id=state["video_id"],
            video_title=title,
            summary_text=summary,
            has_subtitles=True,
        )

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "summarizer",
            {"title": title, "subtitle_length": len(subtitles) if subtitles else 0},
            {"summary_length": len(summary)},
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["summary"] = None
        state["error"] = str(e)
        _add_trace(state, "summarizer", None, None, duration, "failed", str(e))
        logger.exception("Summarizer node failed")

    return state


async def reply_node(state: AgentState, platform: PlatformBase) -> AgentState:
    """Reply 节点：格式化并发布评论"""
    t0 = time.perf_counter()

    try:
        video_id = state["video_id"]
        video_info = state.get("video_info")
        title = video_info.title if video_info else ""

        # 确定回复内容：缓存摘要 / 新摘要 / 错误信息
        summary = state.get("cached_summary") or state.get("summary")
        is_error = summary is None
        error_reason = state.get("error") if is_error else None

        agent = ReplyAgent(platform)
        result = await agent.run(
            video_id=video_id,
            title=title,
            summary=summary,
            is_error=is_error,
            error_reason=error_reason,
        )

        state["reply_parts"] = result["reply_parts"]
        state["comment_ids"] = result["comment_ids"]
        state["success"] = result["success"]

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "reply",
            {"is_error": is_error, "summary_length": len(summary) if summary else 0},
            {"parts_count": len(result["reply_parts"]), "success": result["success"]},
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["success"] = False
        state["error"] = str(e)
        _add_trace(state, "reply", None, None, duration, "failed", str(e))
        logger.exception("Reply node failed")

    return state


# ---- 路由函数 ----

def route_after_supervisor(state: AgentState) -> str:
    """Supervisor 之后的条件路由"""
    route = state.get("route", "ignore")
    if route == "use_cache":
        return "reply"
    elif route == "analyze":
        return "analyzer"
    else:
        return END


def route_after_analyzer(state: AgentState) -> str:
    """Analyzer 之后的条件路由"""
    if state.get("can_summarize"):
        return "summarizer"
    else:
        # 无法总结也要回复（告知原因）
        return "reply"


# ---- 构建图 ----

def build_workflow(platform: PlatformBase) -> StateGraph:
    """构建并编译 LangGraph 工作流

    工作流结构：
        START → supervisor → [use_cache] → reply → END
                           → [analyze]  → analyzer → [can_summarize] → summarizer → reply → END
                                                    → [cannot]       → reply → END
                           → [ignore]   → END
    """
    graph = StateGraph(AgentState)

    # 注册节点（用 async 闭包注入 platform 依赖）
    async def _analyzer(state: AgentState) -> AgentState:
        return await analyzer_node(state, platform)

    async def _reply(state: AgentState) -> AgentState:
        return await reply_node(state, platform)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyzer", _analyzer)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("reply", _reply)

    # 边：START → supervisor
    graph.add_edge(START, "supervisor")

    # 条件边：supervisor → reply / analyzer / END
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "reply": "reply",
        "analyzer": "analyzer",
        END: END,
    })

    # 条件边：analyzer → summarizer / reply
    graph.add_conditional_edges("analyzer", route_after_analyzer, {
        "summarizer": "summarizer",
        "reply": "reply",
    })

    # 边：summarizer → reply
    graph.add_edge("summarizer", "reply")

    # 边：reply → END
    graph.add_edge("reply", END)

    return graph.compile()
