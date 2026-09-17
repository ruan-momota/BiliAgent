"""Indexer 单元测试"""

from unittest.mock import patch

from biliagent.rag.indexer import index_subtitles


class TestIndexSubtitles:
    """测试高层索引接口"""

    @patch("biliagent.rag.indexer.add_chunks", return_value=5)
    @patch("biliagent.rag.indexer.is_video_indexed", return_value=False)
    @patch("biliagent.rag.indexer.chunk_subtitles", return_value=["c1", "c2", "c3", "c4", "c5"])
    def test_index_new_video(self, mock_chunk, mock_indexed, mock_add):
        """新视频：执行分块和存储"""
        count = index_subtitles("bilibili", "BV001", "测试", "长字幕" * 1000)
        assert count == 5
        mock_chunk.assert_called_once()
        mock_add.assert_called_once_with("bilibili", "BV001", "测试", ["c1", "c2", "c3", "c4", "c5"])

    @patch("biliagent.rag.indexer.add_chunks")
    @patch("biliagent.rag.indexer.is_video_indexed", return_value=True)
    @patch("biliagent.rag.indexer.chunk_subtitles")
    def test_skip_already_indexed(self, mock_chunk, mock_indexed, mock_add):
        """已索引视频：跳过，返回 0"""
        count = index_subtitles("bilibili", "BV001", "测试", "字幕内容")
        assert count == 0
        mock_chunk.assert_not_called()
        mock_add.assert_not_called()
