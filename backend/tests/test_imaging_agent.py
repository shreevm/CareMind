import asyncio
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agents.imaging import ImagingAgent, NO_IMAGE_EVIDENCE_ANSWER
from backend.caremind.config import Settings
from backend.caremind.ingestion import DocumentIngestionService
from backend.caremind.ocr import OCRResult
from backend.caremind.schemas import ChatRequest, RetrievedChunk
from backend.caremind.store import SQLiteStore
from backend.caremind.vectorstore import VectorStore


def test_imaging_agent_answers_from_paired_report_with_citation() -> None:
    chunk = RetrievedChunk(
        chunk_id="image:image-1",
        document_id="image-1",
        document_name="synthetic-cxr.png",
        text=(
            "Clinical image asset: synthetic-cxr.png. Modality: chest_xray. "
            "Paired report or caption for grounding: no acute infiltrate is described."
        ),
        score=1.0,
    )
    tools = Mock()
    tools.imaging_search.return_value = [chunk]
    agent = ImagingAgent(tools)

    result = agent.run(ChatRequest(message="What does this X-ray show?"), [], [])

    assert result["answer"].startswith("Based on the paired report or caption")
    assert result["citations"][0].document_id == "image-1"
    assert result["tool_calls"] == ["imaging_search"]


def test_imaging_tool_path_without_paired_report_asks_for_grounding() -> None:
    tools = Mock()
    tools.imaging_search.return_value = [
        RetrievedChunk(
            chunk_id="image:image-1",
            document_id="image-1",
            document_name="synthetic-cxr.png",
            text=(
                "Clinical image asset: synthetic-cxr.png. Modality: chest_xray. "
                "No paired report or caption was provided, so CareMind cannot ground image findings yet."
            ),
            score=0.0,
        )
    ]
    agent = ImagingAgent(tools)

    result = agent.run(ChatRequest(message="What does this X-ray show?"), [], [])

    assert result["answer"] == NO_IMAGE_EVIDENCE_ANSWER
    assert result["citations"] == []
    assert result["require_citations"] is False
    assert result["no_relevant_document_evidence"] is True


def test_document_tools_imaging_search_uses_uploaded_image_summary() -> None:
    tools = DocumentToolsShim()

    chunks = tools.imaging_search("What does this X-ray show?", workspace_id="default")

    assert len(chunks) == 1
    assert chunks[0].document_id == "image-1"
    assert "Paired report or caption" in chunks[0].text


class DocumentToolsShim:
    def __init__(self) -> None:
        from backend.caremind.tools import DocumentTools

        self._tools_cls = DocumentTools
        self.embeddings = SimpleNamespace(settings=SimpleNamespace(langsmith_enabled=False, langsmith_redact_inputs=True))
        self.store = Mock()
        self.store.list_images.return_value = [
            SimpleNamespace(
                image_id="image-1",
                filename="synthetic-cxr.png",
                modality="chest_xray",
                report_text_summary="no acute infiltrate is described.",
            )
        ]

    def imaging_search(self, query: str, workspace_id: str, top_k: int = 3):
        return self._tools_cls.imaging_search(self, query, workspace_id, top_k)


class DeterministicImageEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_provider = ""
        self.last_model = ""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_provider = "test"
        self.last_model = "image-ocr-test"
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class StubOCR:
    def extract(self, image_path: Path) -> OCRResult:
        return OCRResult(
            text="Patient: Test Person\nHemoglobin 11.2 g/dL\nImpression: mild anemia noted",
            engine="stub",
            confidence=0.91,
        )


class FailingVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def upsert(self, *args, **kwargs) -> None:
        raise RuntimeError("vector backend unavailable")


def test_image_upload_creates_linked_ocr_document_and_chunks(tmp_path: Path) -> None:
    asyncio.run(_image_upload_creates_linked_ocr_document_and_chunks(tmp_path))


def test_image_upload_keeps_asset_when_ocr_indexing_fails(tmp_path: Path) -> None:
    asyncio.run(_image_upload_keeps_asset_when_ocr_indexing_fails(tmp_path))


def test_sqlite_store_upgrades_legacy_image_assets_schema(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table image_assets (
                image_id text primary key,
                workspace_id text not null,
                filename text not null
            )
            """
        )
        conn.execute(
            "insert into image_assets (image_id, workspace_id, filename) values (?, ?, ?)",
            ("legacy-image", "default", "old.png"),
        )

    store = SQLiteStore(db_path)

    image = store.list_images("default")[0]
    assert image.image_id == "legacy-image"
    assert image.file_path == ""
    store.save_image_asset(
        image_id="new-image",
        workspace_id="default",
        filename="new.png",
        content_type="image/png",
        file_path=tmp_path / "new.png",
        modality="clinical_image",
        report_text_summary="caption",
        ocr_document_id="doc-1",
    )
    saved = {item.image_id: item for item in store.list_images("default")}
    assert saved["new-image"].file_path.endswith("new.png")
    assert saved["new-image"].ocr_document_id == "doc-1"


async def _image_upload_creates_linked_ocr_document_and_chunks(tmp_path: Path) -> None:
    from starlette.datastructures import Headers, UploadFile

    settings = Settings(
        _env_file=None,
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_VECTOR_BACKEND="sqlite",
        CAREMIND_EMBEDDING_DIM=4,
        CAREMIND_IMAGE_OCR_STRUCTURING_ENABLED=False,
        supabase_url=None,
        supabase_secret_key=None,
        PINECONE_API_KEY=None,
    )
    store = SQLiteStore(settings.sqlite_path)
    embeddings = DeterministicImageEmbeddings(settings)
    vectorstore = VectorStore(settings, store)
    ingestion = DocumentIngestionService(
        settings.upload_dir,
        store,
        embeddings,
        vectorstore,
        settings.max_upload_bytes,
        settings.allowed_upload_types,
        ocr=StubOCR(),
    )
    upload = UploadFile(
        filename="report-photo.png",
        file=BytesIO(b"not-a-real-png-but-ocr-is-stubbed"),
        headers=Headers({"content-type": "image/png"}),
    )

    image = await ingestion.ingest_upload(upload, workspace_id="default")

    assert image.ocr_document_id
    document = store.get_document(image.ocr_document_id)
    assert document is not None
    assert document.source_image_id == image.image_id
    assert document.content_type == "application/x-ocr-json"
    chunks = store.get_document_chunks(document.document_id)
    assert chunks
    assert chunks[0].metadata["source"] == "ocr"
    assert chunks[0].metadata["image_id"] == image.image_id
    assert "Hemoglobin" in chunks[0].text


async def _image_upload_keeps_asset_when_ocr_indexing_fails(tmp_path: Path) -> None:
    from starlette.datastructures import Headers, UploadFile

    settings = Settings(
        _env_file=None,
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_VECTOR_BACKEND="sqlite",
        CAREMIND_EMBEDDING_DIM=4,
        CAREMIND_IMAGE_OCR_STRUCTURING_ENABLED=False,
        supabase_url=None,
        supabase_secret_key=None,
        PINECONE_API_KEY=None,
    )
    store = SQLiteStore(settings.sqlite_path)
    embeddings = DeterministicImageEmbeddings(settings)
    ingestion = DocumentIngestionService(
        settings.upload_dir,
        store,
        embeddings,
        FailingVectorStore(settings),
        settings.max_upload_bytes,
        settings.allowed_upload_types,
        ocr=StubOCR(),
    )
    upload = UploadFile(
        filename="report-photo.png",
        file=BytesIO(b"not-a-real-png-but-ocr-is-stubbed"),
        headers=Headers({"content-type": "image/png"}),
    )

    image = await ingestion.ingest_upload(upload, workspace_id="default")

    assert image.image_id
    assert image.ocr_document_id is None
    images = store.list_images("default")
    assert [item.image_id for item in images] == [image.image_id]
    assert images[0].report_text_summary
    assert store.list_documents("default") == []
