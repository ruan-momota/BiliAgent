"""LangGraph 工作流编排 — 把 Agent 串成完整的处理链

工作流结构：
    START → supervisor → [use_cache]     → reply → END
                       → [analyze]       → analyzer → [can] → summarizer → reply → END
                                                    → [no]  → reply → END
                       → [verify]        → verify_cache_judge → [hit]  → reply → END
                                                               → [miss] → verifier → reply → END
                       → [ask]           → qa → reply → END
                       → [ignore]        → END
"""

import json
import logging
import time

from langgraph.graph import END, START, StateGraph

from biliagent.agents.analyzer import AnalyzerAgent
from biliagent.agents.qa import QAAgent
from biliagent.agents.reply import ReplyAgent
from biliagent.agents.summarizer import SummarizerAgent
from biliagent.agents.supervisor import SupervisorAgent
from biliagent.agents.verify_cache_judge import VerifyCacheJudgeAgent
from biliagent.agents.verifier import VerifierAgent
from biliagent.graph.state import AgentState
from biliagent.models.schemas import AgentTraceInfo
from biliagent.platforms.base import PlatformBase
from biliagent.storage.cache import save_summary
from biliagent.storage.verify_cache import save_verification

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
        state["question"] = result.get("question")
        if result.get("reason"):
            state["error"] = result["reason"]

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "supervisor",
            {"content": mention.content, "video_id": mention.video_id},
            {"route": result["route"], "question": result.get("question")},
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
    mention = state["mention"]
    t0 = time.perf_counter()

    try:
        agent = AnalyzerAgent(platform)
        result = await agent.run(video_id, platform=mention.platform)

        state["video_info"] = result["video_info"]
        state["subtitles"] = result.get("subtitles")
        state["has_subtitles"] = result.get("subtitles") is not None
        state["can_summarize"] = result["can_summarize"]
        state["is_long_video"] = result.get("is_long_video", False)
        state["text_source"] = result.get("text_source")
        if not result["can_summarize"]:
            state["error"] = result.get("reason")

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "analyzer",
            {"video_id": video_id},
            {
                "can_summarize": result["can_summarize"],
                "has_subtitles": state["has_subtitles"],
                "is_long_video": state.get("is_long_video", False),
                "text_source": result.get("text_source"),
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

        is_long_video = state.get("is_long_video", False)

        agent = SummarizerAgent()
        result = await agent.run(
            title=title,
            description=description,
            subtitles=subtitles,
            is_long_video=is_long_video,
            video_id=state.get("video_id"),
            platform=state["mention"].platform,
        )

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
            {"title": title, "subtitle_length": len(subtitles) if subtitles else 0, "is_long_video": is_long_video},
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


async def verify_cache_judge_node(state: AgentState) -> AgentState:
    """VerifyCacheJudge 节点：查询鉴别缓存，LLM 判断相似度"""
    mention = state["mention"]
    video_id = state["video_id"]
    question = state.get("question", mention.content)
    t0 = time.perf_counter()

    try:
        agent = VerifyCacheJudgeAgent()
        result = await agent.run(
            video_id=video_id,
            question=question,
            platform=mention.platform,
        )

        state["verify_route"] = result["route"]
        state["cached_verification"] = result.get("cached_verification")
        state["cached_opinion"] = result.get("cached_opinion")
        state["cached_sources"] = result.get("cached_sources")

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "verify_cache_judge",
            {"video_id": video_id, "question": question},
            {"verify_route": result["route"]},
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["verify_route"] = "verify"  # 出错时默认重新生成
        state["error"] = str(e)
        _add_trace(state, "verify_cache_judge", None, None, duration, "failed", str(e))
        logger.exception("VerifyCacheJudge node failed")

    return state


async def verifier_node(state: AgentState, platform: PlatformBase) -> AgentState:
    """Verifier 节点：联网搜索 + 鉴别 + 观点生成"""
    mention = state["mention"]
    video_id = state["video_id"]
    question = state.get("question", mention.content)
    t0 = time.perf_counter()

    try:
        agent = VerifierAgent(platform)
        result = await agent.run(video_id=video_id, question=question)

        verification = result["verification"]
        opinion = result["opinion"]
        sources = result.get("sources", [])

        state["verification"] = verification
        state["opinion"] = opinion
        state["sources"] = sources
        state["video_info"] = result.get("video_info")

        # 保存到鉴别缓存
        video_info = result.get("video_info")
        await save_verification(
            platform=mention.platform,
            video_id=video_id,
            video_title=video_info.title if video_info else None,
            question=question,
            verification=verification,
            opinion=opinion,
            sources=json.dumps(sources, ensure_ascii=False) if sources else None,
        )

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "verifier",
            {"video_id": video_id, "question": question},
            {"opinion": opinion, "verification_length": len(verification)},
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["verification"] = None
        state["error"] = str(e)
        _add_trace(state, "verifier", None, None, duration, "failed", str(e))
        logger.exception("Verifier node failed")

    return state


async def qa_node(state: AgentState, platform: PlatformBase) -> AgentState:
    """QA 节点：RAG 检索 + LLM 生成回答"""
    mention = state["mention"]
    video_id = state["video_id"]
    question = state.get("question") or mention.content
    t0 = time.perf_counter()

    try:
        agent = QAAgent(platform)
        result = await agent.run(
            video_id=video_id,
            question=question,
            platform=mention.platform,
        )

        state["qa_answer"] = result["answer"]
        state["qa_found"] = result["found"]
        state["qa_chunks"] = result.get("chunks", [])
        # 将视频信息挂到 state 上，供 Reply 环节使用标题
        if result.get("video_info") is not None:
            state["video_info"] = result["video_info"]

        duration = int((time.perf_counter() - t0) * 1000)
        _add_trace(
            state, "qa",
            {"video_id": video_id, "question": question},
            {
                "found": result["found"],
                "chunks_count": len(result.get("chunks", [])),
                "answer_length": len(result["answer"]),
            },
            duration,
        )
    except Exception as e:
        duration = int((time.perf_counter() - t0) * 1000)
        state["qa_answer"] = None
        state["qa_found"] = False
        state["qa_chunks"] = []
        state["error"] = str(e)
        _add_trace(state, "qa", None, None, duration, "failed", str(e))
        logger.exception("QA node failed")

    return state


async def reply_node(state: AgentState, platform: PlatformBase) -> AgentState:
    """Reply 节点：格式化并发布评论（支持总结、鉴别、问答三种模式）"""
    t0 = time.perf_counter()

    try:
        video_id = state["video_id"]
        video_info = state.get("video_info")
        title = video_info.title if video_info else ""
        mention = state["mention"]
        user_name = mention.user_name or ""

        route = state.get("route")
        is_verify = route == "verify"
        is_qa = route == "ask"

        if is_verify:
            # 鉴别回复
            verification = (
                state.get("cached_verification") or state.get("verification")
            )
            opinion = state.get("cached_opinion") or state.get("opinion")

            agent = ReplyAgent(platform)
            result = await agent.run(
                video_id=video_id,
                title=title,
                user_name=user_name,
                is_verify=True,
                verification=verification,
                opinion=opinion,
            )
        elif is_qa:
            # 问答回复
            qa_answer = state.get("qa_answer")
            qa_found = state.get("qa_found", False)

            agent = ReplyAgent(platform)
            result = await agent.run(
                video_id=video_id,
                title=title,
                user_name=user_name,
                is_qa=True,
                qa_answer=qa_answer,
                qa_found=qa_found,
            )
        else:
            # 总结回复
            summary = state.get("cached_summary") or state.get("summary")
            is_error = summary is None
            error_reason = state.get("error") if is_error else None

            agent = ReplyAgent(platform)
            result = await agent.run(
                video_id=video_id,
                title=title,
                user_name=user_name,
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
            {"is_verify": is_verify, "is_qa": is_qa},
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
    elif route == "verify":
        return "verify_cache_judge"
    elif route == "ask":
        return "qa"
    else:
        return END


def route_after_analyzer(state: AgentState) -> str:
    """Analyzer 之后的条件路由"""
    if state.get("can_summarize"):
        return "summarizer"
    else:
        return "reply"


def route_after_verify_cache_judge(state: AgentState) -> str:
    """VerifyCacheJudge 之后的条件路由"""
    verify_route = state.get("verify_route", "verify")
    if verify_route == "use_verify_cache":
        return "reply"
    else:
        return "verifier"


# ---- 构建图 ----

def build_workflow(platform: PlatformBase) -> StateGraph:
    """构建并编译 LangGraph 工作流"""
    graph = StateGraph(AgentState)

    # 注册节点（用 async 闭包注入 platform 依赖）
    async def _analyzer(state: AgentState) -> AgentState:
        return await analyzer_node(state, platform)

    async def _verifier(state: AgentState) -> AgentState:
        return await verifier_node(state, platform)

    async def _qa(state: AgentState) -> AgentState:
        return await qa_node(state, platform)

    async def _reply(state: AgentState) -> AgentState:
        return await reply_node(state, platform)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyzer", _analyzer)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("verify_cache_judge", verify_cache_judge_node)
    graph.add_node("verifier", _verifier)
    graph.add_node("qa", _qa)
    graph.add_node("reply", _reply)

    # 边：START → supervisor
    graph.add_edge(START, "supervisor")

    # 条件边：supervisor → reply / analyzer / verify_cache_judge / qa / END
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "reply": "reply",
        "analyzer": "analyzer",
        "verify_cache_judge": "verify_cache_judge",
        "qa": "qa",
        END: END,
    })

    # 条件边：analyzer → summarizer / reply
    graph.add_conditional_edges("analyzer", route_after_analyzer, {
        "summarizer": "summarizer",
        "reply": "reply",
    })

    # 边：summarizer → reply
    graph.add_edge("summarizer", "reply")

    # 条件边：verify_cache_judge → reply / verifier
    graph.add_conditional_edges("verify_cache_judge", route_after_verify_cache_judge, {
        "reply": "reply",
        "verifier": "verifier",
    })

    # 边：verifier → reply
    graph.add_edge("verifier", "reply")

    # 边：qa → reply
    graph.add_edge("qa", "reply")

    # 边：reply → END
    graph.add_edge("reply", END)

    return graph.compile()
