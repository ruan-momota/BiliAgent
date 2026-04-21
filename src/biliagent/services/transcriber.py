"""SenseVoice 语音转文字服务 — 下载音频 + 调用 API 转录"""

import logging
import re
import tempfile
from pathlib import Path

import httpx

from biliagent.config import settings

logger = logging.getLogger("biliagent.services.transcriber")

# SenseVoice 输出中的特殊标签（情感、事件等），转录后需清理
_TAG_PATTERN = re.compile(r"<\|[A-Z_]+\|>")


async def download_audio(url: str) -> Path | None:
    """从 URL 下载音频到临时文件。

    Args:
        url: 音频流地址（如 B站 DASH 音频 URL）

    Returns:
        临时文件路径，失败返回 None。
        调用方负责在使用完毕后删除该文件。
    """
    tmp_path: Path | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        logger.info("Downloading audio to %s", tmp_path)
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "GET",
                url,
                headers={
                    "Referer": "https://www.bilibili.com",
                    "User-Agent": "Mozilla/5.0",
                },
            ) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)

        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        logger.info("Audio downloaded: %.1f MB", size_mb)
        return tmp_path

    except httpx.TimeoutException:
        logger.error("Timeout downloading audio from %s", url)
    except Exception:
        logger.exception("Failed to download audio from %s", url)

    # 清理失败时的临时文件
    if tmp_path is not None and tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    return None


def _clean_transcription(text: str) -> str:
    """清理 SenseVoice 输出中的特殊标签"""
    return _TAG_PATTERN.sub("", text).strip()


async def transcribe(audio_path: Path) -> str | None:
    """调用 SenseVoice API 将音频转为文字。

    Args:
        audio_path: 本地音频文件路径

    Returns:
        转录纯文本，失败返回 None。
    """
    api_url = settings.sensevoice.api_url
    if not api_url:
        logger.warning("SENSEVOICE_API_URL not configured, skipping transcription")
        return None

    api_key = settings.sensevoice.api_key
    timeout = settings.sensevoice.timeout

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("Calling SenseVoice API: %s", api_url)

    last_error: Exception | None = None
    for attempt in range(1, 3):  # 最多重试 2 次
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                with open(audio_path, "rb") as f:
                    resp = await client.post(
                        api_url,
                        headers=headers,
                        files={"files": (audio_path.name, f, "audio/mp4")},
                        data={"keys": audio_path.stem, "lang": "auto"},
                    )
                resp.raise_for_status()
                data = resp.json()

            # 解析响应：兼容官方 API 格式
            results = data.get("results") or data.get("result", [])
            if not results:
                logger.warning("SenseVoice API returned empty results: %s", data)
                return None

            # 提取文本
            if isinstance(results, list):
                parts = []
                for item in results:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("clean_text") or item.get("raw_text", "")
                    else:
                        text = str(item)
                    if text:
                        parts.append(_clean_transcription(text))
                full_text = "\n".join(parts)
            else:
                full_text = _clean_transcription(str(results))

            if not full_text.strip():
                logger.warning("SenseVoice returned empty transcription")
                return None

            logger.info("Transcription complete, length=%d chars", len(full_text))
            return full_text

        except httpx.TimeoutException:
            last_error = httpx.TimeoutException(f"attempt {attempt}")
            logger.warning("SenseVoice API timeout (attempt %d/2)", attempt)
        except Exception as e:
            last_error = e
            logger.warning("SenseVoice API error (attempt %d/2): %s", attempt, e)

    logger.error("SenseVoice API failed after 2 attempts: %s", last_error)
    return None
