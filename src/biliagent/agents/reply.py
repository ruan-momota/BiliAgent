"""Reply Agent — 格式化回复、发布评论、盖楼拆分"""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from biliagent.agents import create_llm, invoke_llm_with_retry, load_prompt
from biliagent.config import settings
from biliagent.platforms.base import PlatformBase

logger = logging.getLogger("biliagent.agent.reply")


class ReplyAgent:
    """回复 Agent：格式化摘要为评论并发布，超长时盖楼"""

    def __init__(self, platform: PlatformBase) -> None:
        self._llm = create_llm("reply", temperature=1)
        self._prompt_template = load_prompt("reply")
        self._platform = platform
        self._max_length = settings.app.summary_max_length
        self._send_interval = settings.app.comment_send_interval

    # 傲娇话术模板 — 总结
    _SUMMARY_HEADER = "哼，既然你都诚心诚意地召唤我了，那我就大发慈悲地给你总结一下吧！"
    _SUMMARY_FOOTER = "（拿走不谢！下次还要找我哦~ [傲娇]）"
    _ERROR_HEADER = "呜…这个视频我也没办法总结啦，不是我不想帮你！"

    # 傲娇话术模板 — 鉴别
    _VERIFY_HEADERS = {
        "agree": "难得看到一个靠谱的视频！本课代表勉为其难地认可一下 [傲娇]",
        "disagree": "哼，让本课代表来鉴别一下！这个视频嘛...说实话有点问题 [思考]",
        "doubt": "哼，让本课代表来鉴别一下！这个视频嘛...得打个问号 [思考]",
        "neutral": "让本课代表来鉴别一下这个视频吧 [思考]",
    }
    _VERIFY_FOOTERS = {
        "agree": "（不是我夸它哦，是事实就是这样！[嫌弃]）",
        "disagree": "（本课代表可是查了资料才说的，不服来辩！[得意]）",
        "doubt": "（本课代表可是查了资料才说的，不服来辩！[得意]）",
        "neutral": "（以上仅代表本课代表个人观点哦~ [傲娇]）",
    }

    # 傲娇话术模板 — 问答
    _QA_FOUND_HEADER = "哼，这种问题也要来问我？算了，看你这么好奇就告诉你吧！"
    _QA_FOUND_FOOTER = "（视频里明明说了嘛，下次认真看！[嫌弃]）"
    _QA_MISSING_HEADER = "这个嘛...视频里好像没怎么提到欸 [思考]"
    _QA_MISSING_FOOTER = "（不是我不帮你，是这个UP主确实没讲这块内容，建议去评论区问UP主本人吧~）"

    async def run(
        self,
        video_id: str,
        title: str,
        user_name: str = "",
        summary: str | None = None,
        is_error: bool = False,
        error_reason: str | None = None,
        is_verify: bool = False,
        verification: str | None = None,
        opinion: str | None = None,
        is_qa: bool = False,
        qa_answer: str | None = None,
        qa_found: bool = False,
    ) -> dict:
        """格式化并发布评论

        Returns:
            {
                "reply_parts": list[str],      # 实际发布的评论内容列表
                "comment_ids": list[str|None],  # 评论ID列表
                "success": bool,
            }
        """
        at_prefix = f"@{user_name} " if user_name else ""

        if is_verify and verification:
            # 鉴别回复：直接使用 Verifier 已生成的内容，包裹傲娇话术
            opinion_key = opinion if opinion in self._VERIFY_HEADERS else "neutral"
            header = self._VERIFY_HEADERS[opinion_key]
            footer = self._VERIFY_FOOTERS[opinion_key]
            formatted_text = (
                f"{at_prefix}{header}\n\n"
                f"{verification}\n\n"
                f"{footer}"
            )
        elif is_qa:
            # 问答回复：直接使用 QA Agent 已生成的回答，按"找到/未找到"包裹话术
            answer_body = qa_answer or "未能生成回答。"
            if qa_found:
                header = self._QA_FOUND_HEADER
                footer = self._QA_FOUND_FOOTER
            else:
                header = self._QA_MISSING_HEADER
                footer = self._QA_MISSING_FOOTER
            formatted_text = (
                f"{at_prefix}{header}\n\n"
                f"{answer_body}\n\n"
                f"{footer}"
            )
        else:
            # 总结回复：用 LLM 格式化
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

            formatted_text = await invoke_llm_with_retry(self._llm, messages, "reply")

            # 包裹 @用户名 + 傲娇话术
            if is_error:
                formatted_text = f"{at_prefix}{self._ERROR_HEADER}"
            else:
                formatted_text = (
                    f"{at_prefix}{self._SUMMARY_HEADER}\n\n"
                    f"{formatted_text}\n\n"
                    f"{self._SUMMARY_FOOTER}"
                )

        # 3. 拆分（盖楼兜底）
        parts = self._split_comment(formatted_text)
        logger.info("Reply formatted into %d part(s)", len(parts))

        # 4. 发布评论
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
