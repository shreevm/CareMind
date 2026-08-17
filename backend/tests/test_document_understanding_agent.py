from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agents.document_understanding import DocumentUnderstandingAgent
from backend.caremind.config import Settings
from backend.caremind.ingestion import DocumentIngestionService
from backend.caremind.store import SQLiteStore
from backend.caremind.vectorstore import VectorStore


class DeterministicEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_provider = ""
        self.last_model = ""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_provider = "test"
        self.last_model = "document-understanding-test"
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def test_document_understanding_agent_extracts_structured_lab_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "blood-report.txt"
    text = "Patient: Test Person\nAge: 42\nHemoglobin 11.2 g/dL\nWBC 8.1 x10^9/L\nDiagnosis: mild anemia"
    path.write_text(text, encoding="utf-8")

    result = DocumentUnderstandingAgent(settings).understand(
        document_id="doc-1",
        workspace_id="default",
        filename="blood-report.txt",
        content_type="text/plain",
        file_path=path,
        extracted_text=text,
    )

    assert result.document_type == "lab_report"
    assert result.raw_json["patient"]["name"] == "Test Person"
    assert any(item["name"].lower().startswith("hemoglobin") for item in result.raw_json["lab_values"])
    assert "Laboratory values" in result.index_text
    assert result.extraction_source == "local_fallback"


def test_ingest_text_stores_document_understanding_and_indexes_structured_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = SQLiteStore(settings.sqlite_path)
    embeddings = DeterministicEmbeddings(settings)
    ingestion = DocumentIngestionService(
        settings.upload_dir,
        store,
        embeddings,
        VectorStore(settings, store),
        settings.max_upload_bytes,
        settings.allowed_upload_types,
    )

    document = ingestion.ingest_text(
        text="Patient: Test Person\nHemoglobin 11.2 g/dL\nMedication: Metformin 500 mg twice daily",
        filename="labs.txt",
        workspace_id="default",
    )

    assert document.document_type == "lab_report"
    assert document.understanding_confidence is not None
    stored = store.get_document_understanding(document.document_id)
    assert stored is not None
    assert stored["document_type"] == "lab_report"
    chunks = store.get_document_chunks(document.document_id)
    assert chunks
    assert chunks[0].metadata["source"] == "document_understanding"
    assert chunks[0].metadata["chunk_title"] == "Document Summary"
    assert any(chunk.metadata["chunk_type"] == "laboratory_results" for chunk in chunks)


def test_ocr_fallback_outputs_unified_schema_and_semantic_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "report-photo.png"
    path.write_bytes(b"stubbed image")
    text = (
        "City Hospital\n"
        "Patient: Test Person Age: 42 Sex: Male\n"
        "Report Date: 12/02/2025\n"
        "Hemoglobin 11.2 g/dL Low\n"
        "WBC 8.1 x10^9/L\n"
        "Dr. Ada Smith\n"
        "Impression: mild anemia noted"
    )

    result = DocumentUnderstandingAgent(settings).understand(
        document_id="doc-ocr",
        workspace_id="default",
        filename="report-photo.png.ocr.json",
        content_type="application/x-ocr-json",
        file_path=path,
        extracted_text=text,
        source_image_id="image-1",
        ocr_confidence=0.91,
    )

    payload = result.raw_json
    assert result.extraction_source == "ocr_postprocess"
    assert payload["ingestion_method"] == "ocr"
    assert payload["patient"]["name"] == "Test Person"
    assert payload["hospital"]["name"] == "City Hospital"
    assert payload["provider"]["physician_name"].startswith("Ada Smith")
    assert payload["tests"] == payload["lab_values"]
    assert payload["dates"]["report_date"] == "12/02/2025"
    assert any(item["name"].lower().startswith("hemoglobin") for item in payload["tests"])
    assert any(chunk["metadata"]["chunk_type"] == "laboratory_results" for chunk in result.semantic_chunks)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_VECTOR_BACKEND="sqlite",
        CAREMIND_EMBEDDING_DIM=4,
        CAREMIND_DOCUMENT_UNDERSTANDING_ENABLED=True,
        supabase_url=None,
        supabase_secret_key=None,
        PINECONE_API_KEY=None,
    )
