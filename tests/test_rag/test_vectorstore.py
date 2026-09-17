"""VectorStore 单元测试 — 使用临时目录隔离 ChromaDB"""

import pytest
from unittest.mock import patch, MagicMock

from biliagent.rag import vectorstore


@pytest.fixture(autouse=True)
def reset_singletons():
    """每个测试前重置模块级单例"""
    vectorstore._embedding_fn = None
    vectorstore._chroma_client = None
    yield
    vectorstore._embedding_fn = None
    vectorstore._chroma_client = None


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings with a temp ChromaDB directory"""
    mock = MagicMock()
    mock.rag.chroma_persist_dir = str(tmp_path / "chroma")
    mock.rag.embedding_model = "BAAI/bge-base-zh-v1.5"
    mock.rag.qa_top_k = 3
    return mock


class TestVectorStoreOperations:
    """测试 ChromaDB 存取操作（使用假 embedding）"""

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_add_and_get_chunks(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """添加分块后能按序取回"""
        mock_settings_obj.rag = mock_settings.rag

        # 使用假 embedding（返回固定维度向量）
        fake_embed = MagicMock()
        fake_embed.embed_documents.return_value = [[0.1] * 10, [0.2] * 10, [0.3] * 10]
        fake_embed.embed_query.return_value = [0.1] * 10
        mock_embed_fn.return_value = fake_embed

        chunks = ["第一段内容", "第二段内容", "第三段内容"]
        count = vectorstore.add_chunks("bilibili", "BV123", "测试视频", chunks)
        assert count == 3

        # 按序取回
        result = vectorstore.get_all_chunks_ordered("bilibili", "BV123")
        assert len(result) == 3
        assert result[0] == "第一段内容"
        assert result[2] == "第三段内容"

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_is_video_indexed(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """检查视频是否已索引"""
        mock_settings_obj.rag = mock_settings.rag

        fake_embed = MagicMock()
        fake_embed.embed_documents.return_value = [[0.1] * 10]
        mock_embed_fn.return_value = fake_embed

        assert not vectorstore.is_video_indexed("bilibili", "BV999")

        vectorstore.add_chunks("bilibili", "BV999", "视频", ["一段内容"])
        assert vectorstore.is_video_indexed("bilibili", "BV999")

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_delete_video_chunks(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """删除视频分块"""
        mock_settings_obj.rag = mock_settings.rag

        fake_embed = MagicMock()
        fake_embed.embed_documents.return_value = [[0.1] * 10, [0.2] * 10]
        mock_embed_fn.return_value = fake_embed

        vectorstore.add_chunks("bilibili", "BVdel", "视频", ["段1", "段2"])
        assert vectorstore.is_video_indexed("bilibili", "BVdel")

        vectorstore.delete_video_chunks("bilibili", "BVdel")
        assert not vectorstore.is_video_indexed("bilibili", "BVdel")

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_add_chunks_empty(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """空分块列表返回 0"""
        mock_settings_obj.rag = mock_settings.rag
        mock_embed_fn.return_value = MagicMock()

        count = vectorstore.add_chunks("bilibili", "BVempty", "视频", [])
        assert count == 0

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_idempotent_add(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """重复添加相同 ID 的分块不会产生重复数据"""
        mock_settings_obj.rag = mock_settings.rag

        fake_embed = MagicMock()
        fake_embed.embed_documents.return_value = [[0.1] * 10, [0.2] * 10]
        mock_embed_fn.return_value = fake_embed

        chunks = ["段1", "段2"]
        vectorstore.add_chunks("bilibili", "BVidem", "视频", chunks)
        vectorstore.add_chunks("bilibili", "BVidem", "视频", chunks)

        result = vectorstore.get_all_chunks_ordered("bilibili", "BVidem")
        assert len(result) == 2

    @patch("biliagent.rag.vectorstore.settings")
    @patch("biliagent.rag.vectorstore._get_embedding_fn")
    def test_get_chunks_nonexistent_video(self, mock_embed_fn, mock_settings_obj, mock_settings, tmp_path):
        """查询不存在的视频返回空列表"""
        mock_settings_obj.rag = mock_settings.rag
        mock_embed_fn.return_value = MagicMock()

        result = vectorstore.get_all_chunks_ordered("bilibili", "BVnotfound")
        assert result == []
