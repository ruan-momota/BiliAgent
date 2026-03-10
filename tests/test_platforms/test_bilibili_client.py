"""Bilibili 平台客户端单元测试 — Mock 所有外部 API 调用"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestExtractBvid:
    """测试 BV 号提取"""

    def test_extract_from_standard_url(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._extract_bvid("https://www.bilibili.com/video/BV1xx411c7XW")
        assert result == "BV1xx411c7XW"

    def test_extract_from_url_with_query(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._extract_bvid("https://www.bilibili.com/video/BV1xx411c7XW?p=1")
        assert result == "BV1xx411c7XW"

    def test_extract_from_protocol_relative(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._extract_bvid("//www.bilibili.com/video/BV1abc")
        assert result == "BV1abc"

    def test_extract_returns_none_for_invalid(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._extract_bvid("https://example.com")
        assert result is None

    def test_extract_returns_none_for_empty(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._extract_bvid("")
        assert result is None


class TestPickSubtitleUrl:
    """测试字幕 URL 选择逻辑"""

    def test_prefers_chinese(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        subs = [
            {"lan": "en", "subtitle_url": "//sub.en.json"},
            {"lan": "zh-CN", "subtitle_url": "//sub.zh.json"},
        ]
        result = BilibiliPlatform._pick_subtitle_url(subs)
        assert result == "https://sub.zh.json"

    def test_falls_back_to_first(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        subs = [
            {"lan": "en", "subtitle_url": "//sub.en.json"},
        ]
        result = BilibiliPlatform._pick_subtitle_url(subs)
        assert result == "https://sub.en.json"

    def test_empty_list_returns_none(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        result = BilibiliPlatform._pick_subtitle_url([])
        assert result is None

    def test_handles_full_url(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        subs = [{"lan": "zh-Hans", "subtitle_url": "https://sub.bilibili.com/sub.json"}]
        result = BilibiliPlatform._pick_subtitle_url(subs)
        assert result == "https://sub.bilibili.com/sub.json"


class TestMakeVideo:
    """测试视频对象创建"""

    @patch("biliagent.platforms.bilibili.client.settings")
    @patch("biliagent.platforms.bilibili.client.Credential")
    def test_bvid_video(self, mock_cred_cls, mock_settings):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        mock_settings.bili.sessdata = "test"
        mock_settings.bili.bili_jct = "test"
        mock_settings.bili.buvid3 = "test"

        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential = MagicMock()
        platform._credential_valid = None

        video = platform._make_video("BV1xx411c7XW")
        assert video is not None

    @patch("biliagent.platforms.bilibili.client.settings")
    @patch("biliagent.platforms.bilibili.client.Credential")
    def test_aid_video(self, mock_cred_cls, mock_settings):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        mock_settings.bili.sessdata = "test"
        mock_settings.bili.bili_jct = "test"
        mock_settings.bili.buvid3 = "test"

        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential = MagicMock()
        platform._credential_valid = None

        video = platform._make_video("aid:12345")
        assert video is not None


class TestParseMention:
    """测试 @消息解析"""

    @patch("biliagent.platforms.bilibili.client.settings")
    def test_parse_valid_mention(self, mock_settings):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        mock_settings.bili.sessdata = "test"
        mock_settings.bili.bili_jct = "test"
        mock_settings.bili.buvid3 = "test"

        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential = MagicMock()
        platform._credential_valid = None

        item = {
            "id": "123",
            "user": {"mid": "456", "nickname": "testuser"},
            "uri": "https://www.bilibili.com/video/BV1test",
            "native_uri": "",
            "source_id": "789",
            "subject_id": "101",
            "source_content": "帮我总结一下",
        }

        result = platform._parse_mention(item)
        assert result is not None
        assert result.mention_id == "123"
        assert result.video_id == "BV1test"
        assert result.user_name == "testuser"

    @patch("biliagent.platforms.bilibili.client.settings")
    def test_parse_mention_with_aid_fallback(self, mock_settings):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        mock_settings.bili.sessdata = "test"
        mock_settings.bili.bili_jct = "test"
        mock_settings.bili.buvid3 = "test"

        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential = MagicMock()
        platform._credential_valid = None

        item = {
            "id": "123",
            "user": {"mid": "456", "nickname": "testuser"},
            "uri": "https://example.com/no-video-here",
            "native_uri": "",
            "source_id": "789",
            "subject_id": "999",
            "source_content": "总结",
        }

        result = platform._parse_mention(item)
        assert result is not None
        assert result.video_id == "aid:999"


class TestCredentialDetection:
    """测试 Cookie 过期检测"""

    def test_detect_login_error(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential_valid = True

        platform._detect_credential_error(Exception("账号未登录 login required"))
        assert platform._credential_valid is False

    def test_detect_csrf_error(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential_valid = True

        platform._detect_credential_error(Exception("csrf token invalid"))
        assert platform._credential_valid is False

    def test_non_credential_error_unchanged(self):
        from biliagent.platforms.bilibili.client import BilibiliPlatform
        platform = BilibiliPlatform.__new__(BilibiliPlatform)
        platform._credential_valid = True

        platform._detect_credential_error(Exception("network timeout"))
        assert platform._credential_valid is True
