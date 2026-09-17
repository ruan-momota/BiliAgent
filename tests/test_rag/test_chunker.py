"""Chunker 单元测试"""

from unittest.mock import patch, MagicMock

from biliagent.rag.chunker import chunk_subtitles


class TestChunkSubtitles:
    """测试字幕分块逻辑"""

    @patch("biliagent.rag.chunker.settings")
    def test_short_text_single_chunk(self, mock_settings):
        """短文本不会被拆分"""
        mock_settings.rag = MagicMock(chunk_size=1000, chunk_overlap=200)
        text = "这是一段短字幕。"
        chunks = chunk_subtitles(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    @patch("biliagent.rag.chunker.settings")
    def test_long_text_multiple_chunks(self, mock_settings):
        """长文本会被拆分为多个分块"""
        mock_settings.rag = MagicMock(chunk_size=100, chunk_overlap=20)
        # 生成一段超过 100 字符的文本
        text = "。".join([f"第{i}段内容，包含一些信息" for i in range(20)])
        chunks = chunk_subtitles(text)
        assert len(chunks) > 1
        # 每个分块不应超过 chunk_size 太多（允许一定的分隔符偏差）
        for chunk in chunks:
            assert len(chunk) <= 200  # 给一些余量

    @patch("biliagent.rag.chunker.settings")
    def test_chunks_preserve_content(self, mock_settings):
        """分块后合并应包含原文所有内容（考虑重叠）"""
        mock_settings.rag = MagicMock(chunk_size=50, chunk_overlap=10)
        text = "第一部分内容。第二部分内容。第三部分内容。第四部分内容。第五部分内容。"
        chunks = chunk_subtitles(text)
        # 所有原文字符都应出现在至少一个分块中
        combined = "".join(chunks)
        for char in text:
            assert char in combined

    @patch("biliagent.rag.chunker.settings")
    def test_empty_text(self, mock_settings):
        """空文本返回空列表"""
        mock_settings.rag = MagicMock(chunk_size=1000, chunk_overlap=200)
        chunks = chunk_subtitles("")
        assert chunks == []

    @patch("biliagent.rag.chunker.settings")
    def test_paragraph_splitting(self, mock_settings):
        """优先按段落分割"""
        mock_settings.rag = MagicMock(chunk_size=30, chunk_overlap=5)
        text = "第一段内容讲解了基础知识，这是非常重要的部分。\n\n第二段内容讲解了进阶技巧，需要多加练习。\n\n第三段内容讲解了实战案例，适合深入学习。"
        chunks = chunk_subtitles(text)
        assert len(chunks) >= 2
