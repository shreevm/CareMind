from pathlib import Path
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.config import Settings
from backend.caremind.schemas import RetrievedChunk
from backend.caremind.store import SQLiteStore
from backend.caremind.vectorstore import VectorStore, VectorStoreUnavailable


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "CAREMIND_SQLITE_PATH": tmp_path / "caremind.db",
        "CAREMIND_DATA_DIR": tmp_path,
        "CAREMIND_UPLOAD_DIR": tmp_path / "uploads",
        "CAREMIND_VECTOR_BACKEND": "supabase",
        "supabase_url": None,
        "supabase_publishable_key": None,
        "supabase_secret_key": None,
        "SUPABASE_DB_URL": None,
        "PINECONE_API_KEY": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_supabase_is_default_vector_backend(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))

    status = store.status()

    assert settings.should_use_supabase
    assert status["backend"] == "supabase"
    assert status["configured"] is False
    assert status["error"] == "SUPABASE_DB_URL or SUPABASE_URL + SUPABASE_SECRET_KEY is required for server-side vector writes"


def test_supabase_next_public_settings_are_accepted(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_publishable_key="publishable-key",
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))

    status = store.status()

    assert settings.supabase_project_ref == "example-ref"
    assert status["project_ref"] == "example-ref"
    assert status["url_configured"] is True
    assert status["publishable_key_configured"] is True
    assert status["db_url_configured"] is False


def test_supabase_secret_key_enables_rest_configuration(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_secret_key="secret-key",
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))

    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    with patch.object(store, "_supabase_rest_client") as client_factory:
        client = client_factory.return_value.__enter__.return_value
        client.get.return_value = response

        status = store.status()

    assert status["configured"] is True
    assert status["connected"] is True
    assert status["access_method"] == "rest"
    assert status["secret_key_configured"] is True


def test_supabase_status_auto_creates_missing_table_when_db_url_is_configured(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        SUPABASE_DB_URL="postgresql://user:pass@example-ref.supabase.co:5432/postgres",
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))
    cursor = Mock()
    cursor.fetchone.return_value = [None]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None
    connection.cursor.return_value.__enter__.return_value = cursor
    connection.cursor.return_value.__exit__.return_value = None

    with patch.object(store, "_supabase_connect", return_value=connection), patch.object(
        store, "ensure_supabase_schema"
    ) as ensure_schema:
        status = store.status()

    ensure_schema.assert_called_once()
    assert status["access_method"] == "postgres"
    assert status["connected"] is True
    assert status["table_exists"] is True
    assert status["auto_migration"] == "available"


def test_supabase_schema_auto_create_requires_db_url(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_secret_key="secret-key",
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))

    with pytest.raises(VectorStoreUnavailable, match="automatic table creation requires SUPABASE_DB_URL"):
        store.ensure_supabase_schema()


def test_supabase_schema_auto_create_runs_postgres_ddl(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        SUPABASE_DB_URL="postgresql://user:pass@example-ref.supabase.co:5432/postgres",
        SUPABASE_VECTOR_TABLE="public.caremind_document_chunks_v2",
        CAREMIND_EMBEDDING_DIM=1024,
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))
    cursor = Mock()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None
    connection.cursor.return_value.__enter__.return_value = cursor
    connection.cursor.return_value.__exit__.return_value = None

    with patch.object(store, "_supabase_connect", return_value=connection):
        store.ensure_supabase_schema()

    assert cursor.execute.call_count >= 6


def test_supabase_rest_upsert_uses_secret_key_path(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_secret_key="secret-key",
        CAREMIND_EMBEDDING_DIM=3,
    )
    sqlite_store = SQLiteStore(settings.sqlite_path)
    store = VectorStore(settings, sqlite_store)
    chunk = RetrievedChunk(chunk_id="doc-1:0", document_id="doc-1", document_name="one.txt", text="heart rhythm")

    delete_response = Mock()
    delete_response.raise_for_status.return_value = None
    post_response = Mock()
    post_response.raise_for_status.return_value = None
    with patch.object(store, "_supabase_rest_client") as client_factory:
        client = client_factory.return_value.__enter__.return_value
        client.delete.return_value = delete_response
        client.post.return_value = post_response

        store.upsert([chunk], [[1, 0, 0]], "default", embedding_provider="nvidia", embedding_model="nvolveqa_40k")

    client.delete.assert_called_once()
    client.post.assert_called_once()
    posted_rows = client.post.call_args.kwargs["json"]
    assert posted_rows[0]["embedding"] == "[1.0,0.0,0.0]"
    assert posted_rows[0]["embedding_provider"] == "nvidia"
    assert posted_rows[0]["embedding_model"] == "nvolveqa_40k"
    assert posted_rows[0]["embedding_dimension"] == 3
    assert posted_rows[0]["metadata"]["embedding_provider"] == "nvidia"
    assert [saved.chunk_id for saved in sqlite_store.get_document_chunks("doc-1")] == ["doc-1:0"]


def test_supabase_rest_upsert_missing_table_fails_with_setup_hint(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_secret_key="secret-key",
        CAREMIND_EMBEDDING_DIM=3,
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))
    chunk = RetrievedChunk(chunk_id="doc-1:0", document_id="doc-1", document_name="one.txt", text="heart rhythm")
    missing_table_response = Mock(status_code=404)

    with patch.object(store, "_supabase_rest_client") as client_factory:
        client = client_factory.return_value.__enter__.return_value
        client.get.return_value = missing_table_response

        with pytest.raises(VectorStoreUnavailable, match="SUPABASE_DB_URL"):
            store.upsert([chunk], [[1, 0, 0]], "default")

    client.delete.assert_not_called()
    client.post.assert_not_called()


def test_supabase_rest_search_preserves_raw_rpc_score(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        supabase_url="https://example-ref.supabase.co",
        supabase_secret_key="secret-key",
        CAREMIND_EMBEDDING_DIM=3,
    )
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {
            "chunk_id": "doc-1:0",
            "document_id": "doc-1",
            "document_name": "venkat.pdf",
            "text": "Mr. Venkat Ramanujam Sankar Ram",
            "page": 1,
            "score": 0.033,
        }
    ]
    with patch.object(store, "_supabase_rest_client") as client_factory:
        client = client_factory.return_value.__enter__.return_value
        client.post.return_value = response

        results = store.search([1, 0, 0], "default", top_k=1)

    assert results[0].score == 0.033
    assert results[0].document_name == "venkat.pdf"


def test_sqlite_vector_backend_still_searches_locally(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, CAREMIND_VECTOR_BACKEND="sqlite", CAREMIND_EMBEDDING_DIM=3)
    sqlite_store = SQLiteStore(settings.sqlite_path)
    store = VectorStore(settings, sqlite_store)
    chunks = [
        RetrievedChunk(chunk_id="doc-1:0", document_id="doc-1", document_name="one.txt", text="heart rhythm"),
        RetrievedChunk(chunk_id="doc-2:0", document_id="doc-2", document_name="two.txt", text="kidney panel"),
    ]

    sqlite_store.save_document(
        document_id="doc-1",
        workspace_id="default",
        filename="one.txt",
        content_type="text/plain",
        file_path=tmp_path / "one.txt",
        summary=None,
    )
    sqlite_store.save_document(
        document_id="doc-2",
        workspace_id="default",
        filename="two.txt",
        content_type="text/plain",
        file_path=tmp_path / "two.txt",
        summary=None,
    )
    store.upsert(chunks, [[1, 0, 0], [0, 1, 0]], "default")

    results = store.search([1, 0, 0], "default", top_k=1)

    assert [result.chunk_id for result in results] == ["doc-1:0"]
    assert results[0].score == pytest.approx(1.0)


def test_unknown_vector_backend_fails_clearly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, CAREMIND_VECTOR_BACKEND="mystery")
    store = VectorStore(settings, SQLiteStore(settings.sqlite_path))

    with pytest.raises(VectorStoreUnavailable, match="Unsupported CAREMIND_VECTOR_BACKEND"):
        store.search([1.0], "default")
