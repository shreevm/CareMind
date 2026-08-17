from pathlib import Path

import pytest

from backend.caremind.config import Settings
from backend.caremind.embeddings import EmbeddingClient, EmbeddingServiceUnavailable
from backend.caremind.schemas import RetrievedChunk
from backend.caremind.store import SQLiteStore


def settings(tmp_path: Path) -> Settings:
    return Settings(
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_EMBEDDING_PROVIDER="qwen",
        CAREMIND_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B",
        CAREMIND_EMBEDDING_DIM=4,
        CAREMIND_EMBEDDING_INDEX_VERSION="qwen-test-v1-4",
        CAREMIND_EMBEDDING_BATCH_SIZE=2,
    )


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_document(self, texts, **kwargs):
        self.calls.append("document")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, texts, **kwargs):
        self.calls.append("query")
        return [[0.0, 1.0, 0.0, 0.0] for _ in texts]


def test_qwen_documents_use_encode_document(monkeypatch, tmp_path: Path) -> None:
    fake = FakeSentenceTransformer()
    monkeypatch.setattr(EmbeddingClient, "_load_sentence_transformer", lambda self: fake)
    client = EmbeddingClient(settings(tmp_path))

    vectors = client.embed_texts(["clinical chunk"])

    assert fake.calls == ["document"]
    assert vectors == [[1.0, 0.0, 0.0, 0.0]]
    assert client.last_provider == "qwen"
    assert client.last_model == "Qwen/Qwen3-Embedding-0.6B"


def test_qwen_queries_use_encode_query(monkeypatch, tmp_path: Path) -> None:
    fake = FakeSentenceTransformer()
    monkeypatch.setattr(EmbeddingClient, "_load_sentence_transformer", lambda self: fake)
    client = EmbeddingClient(settings(tmp_path))

    vector = client.embed_query("what does high hba1c mean")

    assert fake.calls == ["query"]
    assert vector == [0.0, 1.0, 0.0, 0.0]


def test_invalid_embedding_dimension_is_rejected(monkeypatch, tmp_path: Path) -> None:
    class BadModel(FakeSentenceTransformer):
        def encode_document(self, texts, **kwargs):
            return [[1.0, 0.0]]

    monkeypatch.setattr(EmbeddingClient, "_load_sentence_transformer", lambda self: BadModel())
    client = EmbeddingClient(settings(tmp_path))

    with pytest.raises(EmbeddingServiceUnavailable, match="expected 4"):
        client.embed_texts(["bad vector"])


def test_sqlite_local_vectors_filter_embedding_index_version(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "caremind.db")
    chunk = RetrievedChunk(
        chunk_id="doc-1:0",
        document_id="doc-1",
        document_name="report.txt",
        text="HbA1c is elevated.",
    )
    store.save_document(
        document_id="doc-1",
        workspace_id="default",
        filename="report.txt",
        content_type="text/plain",
        file_path=tmp_path / "report.txt",
        summary="summary",
    )
    store.replace_chunks(
        [chunk],
        [[1.0, 0.0, 0.0, 0.0]],
        "default",
        embedding_provider="qwen",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=4,
        embedding_index_version="qwen-test-v1-4",
    )

    matching = store.get_local_vectors(
        "default",
        embedding_provider="qwen",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=4,
        embedding_index_version="qwen-test-v1-4",
    )
    mismatched = store.get_local_vectors(
        "default",
        embedding_provider="nvidia",
        embedding_model="nvidia/nv-embedqa-e5-v5",
        embedding_dimension=4,
        embedding_index_version="nvidia-v1-1024",
    )

    assert len(matching) == 1
    assert mismatched == []
