"""SenseVoice 转录服务单元测试"""

import pytest
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from biliagent.services import transcriber
from biliagent.services.transcriber import (
    _clean_transcription,
    download_audio,
    transcribe,
)


class TestCleanTranscription:
    """SenseVoice 输出标签清理"""

    def test_removes_emotion_tags(self):
        text = "<|HAPPY|>大家好<|NEUTRAL|>欢迎收看<|SAD|>"
        assert _clean_transcription(text) == "大家好欢迎收看"

    def test_removes_event_tags(self):
        text = "<|BGM|>开场音乐<|SPEECH|>正文内容<|APPLAUSE|>"
        assert _clean_transcription(text) == "开场音乐正文内容"

    def test_plain_text_unchanged(self):
        assert _clean_transcription("纯文本没有标签") == "纯文本没有标签"

    def test_strips_surrounding_whitespace(self):
        assert _clean_transcription("  <|NEUTRAL|>文字  ") == "文字"


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in supporting `post` and stream (via `stream`)."""

    def __init__(self, *, post_impl=None, stream_impl=None):
        self._post_impl = post_impl
        self._stream_impl = stream_impl

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return await self._post_impl(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._stream_impl(*args, **kwargs)


class _FakeStreamCtx:
    """Async context manager yielding a response-like object with aiter_bytes."""

    def __init__(self, chunks: list[bytes], status_ok: bool = True):
        self._chunks = chunks
        self._status_ok = status_ok

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if not self._status_ok:
            raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]

    async def aiter_bytes(self, chunk_size: int = 65536):
        for chunk in self._chunks:
            yield chunk


class TestTranscribe:
    """transcribe() 的行为覆盖"""

    @pytest.mark.asyncio
    async def test_no_api_url_returns_none(self, tmp_path):
        """未配置 SENSEVOICE_API_URL → 直接返回 None，不发 HTTP"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        with patch.object(transcriber.settings, "sensevoice") as mock_sv:
            mock_sv.api_url = ""
            result = await transcribe(audio)

        assert result is None

    @pytest.mark.asyncio
    async def test_success_returns_cleaned_text(self, tmp_path):
        """API 返回 results → 提取并清理标签后拼接返回"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(return_value={
            "results": [
                {"text": "<|HAPPY|>第一段"},
                {"text": "<|NEUTRAL|>第二段"},
            ]
        })

        async def post_impl(*args, **kwargs):
            return fake_response

        client = _FakeAsyncClient(post_impl=post_impl)

        with patch.object(transcriber.settings, "sensevoice") as mock_sv, \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            mock_sv.api_url = "http://fake/asr"
            mock_sv.api_key = "sk-xxx"
            mock_sv.timeout = 30

            result = await transcribe(audio)

        assert result is not None
        assert "第一段" in result
        assert "第二段" in result
        assert "<|" not in result

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self, tmp_path):
        """API 返回空 results → 返回 None"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(return_value={"results": []})

        async def post_impl(*args, **kwargs):
            return fake_response

        client = _FakeAsyncClient(post_impl=post_impl)

        with patch.object(transcriber.settings, "sensevoice") as mock_sv, \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            mock_sv.api_url = "http://fake/asr"
            mock_sv.api_key = ""
            mock_sv.timeout = 30

            result = await transcribe(audio)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_transcription_text_returns_none(self, tmp_path):
        """API 返回结构正常但 text 为空 → 返回 None"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(return_value={
            "results": [{"text": "<|NEUTRAL|>"}]
        })

        async def post_impl(*args, **kwargs):
            return fake_response

        client = _FakeAsyncClient(post_impl=post_impl)

        with patch.object(transcriber.settings, "sensevoice") as mock_sv, \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            mock_sv.api_url = "http://fake/asr"
            mock_sv.api_key = ""
            mock_sv.timeout = 30

            result = await transcribe(audio)

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_retries_then_returns_none(self, tmp_path):
        """两次都超时 → 返回 None，共尝试 2 次"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        call_count = {"n": 0}

        async def post_impl(*args, **kwargs):
            call_count["n"] += 1
            raise httpx.TimeoutException("slow")

        client = _FakeAsyncClient(post_impl=post_impl)

        with patch.object(transcriber.settings, "sensevoice") as mock_sv, \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            mock_sv.api_url = "http://fake/asr"
            mock_sv.api_key = ""
            mock_sv.timeout = 30

            result = await transcribe(audio)

        assert result is None
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_recovers_on_second_attempt(self, tmp_path):
        """第一次异常第二次成功 → 返回文本"""
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"fake")

        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        ok_response.json = MagicMock(return_value={
            "results": [{"text": "第二次成功"}]
        })

        call_count = {"n": 0}

        async def post_impl(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.TimeoutException("slow")
            return ok_response

        client = _FakeAsyncClient(post_impl=post_impl)

        with patch.object(transcriber.settings, "sensevoice") as mock_sv, \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            mock_sv.api_url = "http://fake/asr"
            mock_sv.api_key = ""
            mock_sv.timeout = 30

            result = await transcribe(audio)

        assert result == "第二次成功"
        assert call_count["n"] == 2


class TestDownloadAudio:
    """download_audio() 的行为覆盖"""

    @pytest.mark.asyncio
    async def test_success_writes_file(self):
        """正常下载 → 返回 Path，文件内容为 stream 拼接"""
        def stream_impl(method, url, headers=None):
            return _FakeStreamCtx(chunks=[b"hello", b"world"])

        client = _FakeAsyncClient(stream_impl=stream_impl)

        with patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            path = await download_audio("http://fake/audio.m4a")

        assert path is not None
        try:
            assert path.exists()
            assert path.read_bytes() == b"helloworld"
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_timeout_cleans_up_tempfile(self):
        """下载超时 → 返回 None，且临时文件被清理"""
        captured_paths: list[Path] = []

        original_named = transcriber.tempfile.NamedTemporaryFile

        def spy_named(*args, **kwargs):
            tmp = original_named(*args, **kwargs)
            captured_paths.append(Path(tmp.name))
            return tmp

        def stream_impl(method, url, headers=None):
            raise httpx.TimeoutException("slow")

        client = _FakeAsyncClient(stream_impl=stream_impl)

        with patch("biliagent.services.transcriber.tempfile.NamedTemporaryFile", side_effect=spy_named), \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            result = await download_audio("http://fake/audio.m4a")

        assert result is None
        assert captured_paths, "NamedTemporaryFile should have been called"
        assert not captured_paths[0].exists(), "temp file should be cleaned up on failure"

    @pytest.mark.asyncio
    async def test_http_error_cleans_up_tempfile(self):
        """HTTP 错误（raise_for_status 抛异常）→ 返回 None，临时文件被清理"""
        captured_paths: list[Path] = []
        original_named = transcriber.tempfile.NamedTemporaryFile

        def spy_named(*args, **kwargs):
            tmp = original_named(*args, **kwargs)
            captured_paths.append(Path(tmp.name))
            return tmp

        def stream_impl(method, url, headers=None):
            return _FakeStreamCtx(chunks=[], status_ok=False)

        client = _FakeAsyncClient(stream_impl=stream_impl)

        with patch("biliagent.services.transcriber.tempfile.NamedTemporaryFile", side_effect=spy_named), \
             patch("biliagent.services.transcriber.httpx.AsyncClient", return_value=client):
            result = await download_audio("http://fake/audio.m4a")

        assert result is None
        assert not captured_paths[0].exists()
