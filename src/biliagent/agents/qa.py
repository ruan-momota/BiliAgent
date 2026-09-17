"""QA Agent — 视频问答

分流策略（与 Phase 10 的 long_video_threshold 对齐）：
- 已索引（说明是长视频，或之前已走过 RAG 路径）→ RAG 模式：向量检索 top-k + LLM 基于片段回答
- 未索引 + 字幕长度 > 阈值 → 先索引再 RAG
- 未索引 + 字幕长度 ≤ 阈值 → Direct 模式：字幕全文直接送 LLM，不入库
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.config import settings
from biliagent.models.schemas import VideoInfo
from biliagent.platforms.base import PlatformBase
from biliagent.rag.indexer import index_subtitles
from biliagent.rag.vectorstore import is_video_indexed, similarity_search

logger = logging.getLogger("biliagent.agent.qa")


class QAAgent:
    """问答 Agent：独立获取视频信息/字幕 → 按字幕长度选择 RAG / Direct"""

    def __init__(self, platform: PlatformBase) -> None:
        self._llm = create_llm("qa", temperature=1)
        self._prompt_rag = load_prompt("qa")
        self._prompt_direct = load_prompt("qa_direct")
        self._platform = platform
        self._top_k = settings.rag.qa_top_k
        self._max_length = settings.rag.qa_max_length
        self._long_threshold = settings.rag.long_video_threshold

    async def run(
        self,
        video_id: str,
        question: str,
        platform: str = "bilibili",
    ) -> dict:
        """执行问答

        Returns:
            {
                "video_info": VideoInfo | None,
                "found": bool,            # 是否从视频中找到答案
                "answer": str,            # 回答正文
                "chunks": list[str],      # RAG 检索到的片段（direct 模式返回 []）
                "mode": str,              # "rag" / "direct"
            }
        """
        # 1. 独立获取视频信息（与 Analyzer/Verifier 解耦）
        try:
            video_info = await self._platform.get_video_info(video_id)
        except Exception:
            logger.exception("Failed to get video info for %s", video_id)
            video_info = VideoInfo(video_id=video_id, title="未知", description="")

        # 2. 已索引 → 直接走 RAG（保持与 Phase 10 长视频索引的一致性）
        if is_video_indexed(platform, video_id):
            logger.info("Video %s already indexed, using RAG", video_id)
            return await self._run_rag(video_info, platform, video_id, question)

        # 3. 未索引 → 先拉字幕
        subtitles = await self._platform.get_subtitles(video_id)
        if not subtitles or not subtitles.strip():
            logger.info("Video %s has no subtitles, QA cannot proceed", video_id)
            return {
                "video_info": video_info,
                "found": False,
                "answer": "该视频暂无字幕，无法基于视频内容回答。",
                "chunks": [],
                "mode": "none",
            }

        # 4. 按字幕长度分流
        if len(subtitles) > self._long_threshold:
            logger.info(
                "Video %s has long subtitles (%d > %d), indexing + RAG",
                video_id, len(subtitles), self._long_threshold,
            )
            index_subtitles(
                platform=platform,
                video_id=video_id,
                video_title=video_info.title,
                subtitles=subtitles,
            )
            return await self._run_rag(video_info, platform, video_id, question)

        logger.info(
            "Video %s has short subtitles (%d ≤ %d), using direct mode",
            video_id, len(subtitles), self._long_threshold,
        )
        return await self._run_direct(video_info, subtitles, question)

    async def _run_rag(
        self,
        video_info: VideoInfo,
        platform: str,
        video_id: str,
        question: str,
    ) -> dict:
        """RAG 模式：向量检索 top-k 相关片段 → LLM 基于片段回答"""
        chunks = similarity_search(question, platform, video_id, k=self._top_k)
        if not chunks:
            logger.info("No chunks retrieved for %s/%s", platform, video_id)
            return {
                "video_info": video_info,
                "found": False,
                "answer": "视频中未检索到相关内容。",
                "chunks": [],
                "mode": "rag",
            }

        logger.info("Retrieved %d chunks for QA on %s", len(chunks), video_id)
        chunks_text = "\n---\n".join(
            f"片段{i + 1}：{chunk}" for i, chunk in enumerate(chunks)
        )
        prompt = self._prompt_rag.format(
            title=video_info.title,
            question=question,
            chunks=chunks_text,
            max_length=self._max_length,
        )
        text = await self._call_llm(prompt, question, tag="qa_rag")
        result = self._parse_response(text)
        answer = self._finalize_answer(result.get("answer", ""))
        found = bool(result.get("found", False))
        logger.info("QA[rag] for %s: found=%s, answer_length=%d", video_id, found, len(answer))
        return {
            "video_info": video_info,
            "found": found,
            "answer": answer,
            "chunks": chunks,
            "mode": "rag",
        }

    async def _run_direct(
        self,
        video_info: VideoInfo,
        subtitles: str,
        question: str,
    ) -> dict:
        """Direct 模式：短视频字幕全文直接送 LLM，不入库"""
        prompt = self._prompt_direct.format(
            title=video_info.title,
            question=question,
            subtitles=subtitles,
            max_length=self._max_length,
        )
        text = await self._call_llm(prompt, question, tag="qa_direct")
        result = self._parse_response(text)
        answer = self._finalize_answer(result.get("answer", ""))
        found = bool(result.get("found", False))
        logger.info(
            "QA[direct] for %s: found=%s, answer_length=%d",
            video_info.video_id, found, len(answer),
        )
        return {
            "video_info": video_info,
            "found": found,
            "answer": answer,
            "chunks": [],
            "mode": "direct",
        }

    async def _call_llm(self, prompt: str, question: str, tag: str) -> str:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"请回答用户问题：{question}"),
        ]
        return await invoke_llm_with_retry(self._llm, messages, tag)

    def _finalize_answer(self, raw: str) -> str:
        """截断到 max_length，空字符串兜底"""
        answer = (raw or "").strip() or "未能基于视频内容生成回答。"
        if len(answer) > self._max_length:
            answer = answer[: self._max_length - 3] + "..."
        return answer

    @staticmethod
    def _parse_response(text: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        try:
            text = text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse QA response: %s", text)
        return {"found": False, "answer": text or "解析回答失败，请稍后重试。"}
