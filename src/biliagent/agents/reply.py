"""Reply Agent — 格式化回复、发布评论、盖楼拆分"""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, load_prompt
from biliagent.config import settings
from biliagent.platforms.base import PlatformBase

logger = logging.getLogger("biliagent.agent.reply")


class ReplyAgent:
    """回复 Agent：格式化摘要为评论并发布，超长时盖楼"""

    def __init__(self, platform: PlatformBase) -> None:
        self._llm = create_llm("reply", temperature=0.3)
        self._prompt_template = load_prompt("reply")
        self._platform = platform
        self._max_length = settings.app.summary_max_length
        self._send_interval = settings.app.comment_send_interval

    async def run(
        self,
        video_id: str,
        title: str,
        summary: str | None = None,
        is_error: bool = False,
        error_reason: str | None = None,
    ) -> dict:
        """格式化并发布评论

        Returns:
            {
                "reply_parts": list[str],      # 实际发布的评论内容列表
                "comment_ids": list[str|None],  # 评论ID列表
                "success": bool,
            }
        """
        # 1. 用 LLM 格式化回复
        prompt = self._prompt_template.format(
            title=title,
            summary=summary or "",
            is_error=is_error,
            error_reason=error_reason or "",
            max_length=self._max_length,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请生成最终评论文本。"),
        ]

        response = await self._llm.ainvoke(messages)
        formatted_text = response.content.strip()

        # 2. 拆分（盖楼兜底）
        parts = self._split_comment(formatted_text)
        logger.info("Reply formatted into %d part(s)", len(parts))

        # 3. 发布评论
        comment_ids: list[str | None] = []
        root_id: str | None = None

        for i, part in enumerate(parts):
            if i == 0:
                # 主楼
                cid = await self._platform.post_comment(video_id, part)
                root_id = cid
            else:
                # 盖楼：回复主楼
                if root_id:
                    await asyncio.sleep(self._send_interval)
                    cid = await self._platform.reply_comment(video_id, root_id, part)
                else:
                    logger.warning("Cannot stack comment: no root comment id")
                    cid = await self._platform.post_comment(video_id, part)
            comment_ids.append(cid)

        success = any(cid is not None for cid in comment_ids)
        if success:
            logger.info("Reply posted on video %s", video_id)
        else:
            logger.error("Failed to post reply on video %s", video_id)

        return {
            "reply_parts": parts,
            "comment_ids": comment_ids,
            "success": success,
        }

    def _split_comment(self, text: str) -> list[str]:
        """按字数限制拆分评论（盖楼策略）"""
        # 预留标记位
        limit = self._max_length - 20

        if len(text) <= self._max_length:
            return [text]

        parts: list[str] = []
        paragraphs = text.split("\n")
        current = ""

        for para in paragraphs:
            # 如果单段就超限，强制按字数切
            if len(para) > limit:
                if current:
                    parts.append(current.strip())
                    current = ""
                for j in range(0, len(para), limit):
                    parts.append(para[j : j + limit])
                continue

            if len(current) + len(para) + 1 > limit:
                parts.append(current.strip())
                current = para + "\n"
            else:
                current += para + "\n"

        if current.strip():
            parts.append(current.strip())

        # 添加楼层标记
        if len(parts) > 1:
            parts[0] += "\n「续 ↓」"
            for i in range(1, len(parts) - 1):
                parts[i] = f"「第{i+1}部分」\n" + parts[i] + "\n「续 ↓」"
            parts[-1] = f"「第{len(parts)}部分」\n" + parts[-1] + "\n「完」"

        return parts
