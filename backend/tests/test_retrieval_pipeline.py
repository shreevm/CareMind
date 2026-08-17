import asyncio
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import Mock

from starlette.datastructures import Headers, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agents.document_rag import DocumentRAGAgent
from backend.caremind.config import Settings
from backend.caremind.ingestion import DocumentIngestionService
from backend.caremind.store import SQLiteStore
from backend.caremind.tools import DocumentTools
from backend.caremind.vectorstore import VectorStore


VENKAT_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length 94 >> stream
BT /F1 12 Tf 72 720 Td (Mr. Venkat Ramanujam Sankar Ram. Tell me about the patient Venkat Ramanujam.) Tj ET
endstream endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000241 00000 n 
0000000311 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
455
%%EOF
"""


class DeterministicVenkatEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimension = settings.embedding_dimension
        self.last_provider = ""
        self.last_model = ""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_provider = "test"
        self.last_model = "deterministic-venkat"
        vectors = []
        for text in texts:
            if "venkat" in text.lower() and "ramanujam" in text.lower():
                vectors.append([1.0, 0.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0, 0.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def configured_model_label(self) -> str:
        return "deterministic-venkat"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_VECTOR_BACKEND="sqlite",
        CAREMIND_EMBEDDING_DIM=4,
        CAREMIND_MIN_RETRIEVAL_SIMILARITY=0.5,
        supabase_url=None,
        supabase_secret_key=None,
        PINECONE_API_KEY=None,
    )


def test_uploaded_pdf_patient_name_retrieves_above_threshold_and_generates_answer(tmp_path: Path) -> None:
    asyncio.run(_uploaded_pdf_patient_name_retrieves_above_threshold_and_generates_answer(tmp_path))


async def _uploaded_pdf_patient_name_retrieves_above_threshold_and_generates_answer(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = SQLiteStore(settings.sqlite_path)
    embeddings = DeterministicVenkatEmbeddings(settings)
    vectorstore = VectorStore(settings, store)
    ingestion = DocumentIngestionService(
        settings.upload_dir,
        store,
        embeddings,
        vectorstore,
        settings.max_upload_bytes,
        settings.allowed_upload_types,
    )
    upload = UploadFile(
        filename="venkat.pdf",
        file=BytesIO(VENKAT_PDF),
        headers=Headers({"content-type": "application/pdf"}),
    )

    await ingestion.ingest_upload(upload, workspace_id="default")

    llm = Mock()
    llm.answer.return_value = "The uploaded report identifies Mr. Venkat Ramanujam Sankar Ram. [1]"
    memory = Mock()
    memory.load.return_value = []
    tools = DocumentTools(embeddings, vectorstore, store)
    agent = DocumentRAGAgent(tools, llm, memory)
    diagnostic = tools.embedding_similarity_diagnostic(
        query="Tell me about the patient Venkat Ramanujam",
        workspace_id="default",
        document_name_contains="venkat",
    )

    result = agent.run(
        request=Mock(
            message="Tell me about the patient Venkat Ramanujam",
            workspace_id="default",
            top_k=5,
            session_id="session",
        ),
        tool_calls=[],
        trace_steps=[],
    )

    assert "document_search" in result["tool_calls"]
    assert result["citations"]
    assert result["retrieved_chunks"]
    assert result["retrieved_chunks"][0]["score"] >= settings.min_retrieval_similarity
    assert result["trace_steps"][-1]["below_threshold_count"] == 0
    assert diagnostic["query_embedding_model"] == "deterministic-venkat"
    assert diagnostic["stored_embedding_model"] == "deterministic-venkat"
    assert diagnostic["dimension_matches"] is True
    assert diagnostic["direct_local_cosine_similarity"] == 1.0
    llm.answer.assert_called_once()
