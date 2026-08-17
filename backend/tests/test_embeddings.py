from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.config import Settings
from backend.caremind.embeddings import EmbeddingClient, EmbeddingServiceUnavailable


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "CAREMIND_SQLITE_PATH": tmp_path / "caremind.db",
        "CAREMIND_DATA_DIR": tmp_path,
        "CAREMIND_UPLOAD_DIR": tmp_path / "uploads",
        "CAREMIND_VECTOR_BACKEND": "sqlite",
        "PINECONE_API_KEY": None,
        "supabase_url": None,
        "supabase_secret_key": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_nvidia_embedding_failure_raises_instead_of_local_hashing(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        CAREMIND_EMBEDDING_PROVIDER="nvidia",
        NVIDIA_API_KEY="configured-key",
    )
    client = EmbeddingClient(settings)

    with patch.object(client, "_embed_with_nvidia", side_effect=RuntimeError("network")):
        with pytest.raises(EmbeddingServiceUnavailable, match="refusing local hashing fallback"):
            client.embed_query("tell me about patient venkat ramanujam")

    assert client.last_provider == ""
    assert client.degraded is False


def test_auto_without_remote_provider_uses_loud_degraded_local_hashing(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        CAREMIND_EMBEDDING_PROVIDER="auto",
        NVIDIA_API_KEY=None,
        CAREMIND_EMBEDDING_DIM=8,
    )
    client = EmbeddingClient(settings)

    vector = client.embed_query("offline demo text")

    assert len(vector) == 8
    assert client.last_provider == "local_hashing"
    assert client.last_model == "local-hashing-8d"
    assert client.degraded is True
    assert client.degraded_reason == "no_remote_embedding_provider_configured"
