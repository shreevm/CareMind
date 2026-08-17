from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from backend.caremind.agent import CareMindAgent
from backend.caremind.agents.comparison import ReportComparisonAgent
from backend.caremind.agents.conversation_resolver import ConversationResolverAgent
from backend.caremind.agents.document_rag import DocumentRAGAgent
from backend.caremind.agents.document_understanding import DocumentUnderstandingAgent
from backend.caremind.agents.imaging import ImagingAgent, NO_IMAGE_EVIDENCE_ANSWER
from backend.caremind.config import Settings
from backend.caremind.memory import ConversationMemory
from backend.caremind.schemas import ChatRequest, CompareResponse, RetrievedChunk
from backend.caremind.safety import SafetyLayer
from backend.caremind.store import SQLiteStore


def test_current_message_attachment_resolves_this_to_image_a() -> None:
    attachment = SimpleNamespace(
        id="attachment_A",
        attachment_type="image",
        filename="image_A.jpg",
        document_id="doc_A",
        image_asset_id="image_A",
    )

    result = ConversationResolverAgent().resolve(
        message="What is this?",
        context={},
        history=[],
        documents=[],
        images=[],
        current_attachments=[attachment],
    )

    assert result["selected_attachment"] == "attachment_A"
    assert result["active_document_ids"] == ["doc_A"]
    assert result["reason"] == "current_message_attachments"
    assert result["intent_rewrite"]["route_hint"] == "retrieve"


def test_current_message_attachment_wins_over_old_active_document() -> None:
    old_doc = SimpleNamespace(document_id="document_B", filename="old-report.pdf")
    attachment = SimpleNamespace(
        id="attachment_A",
        attachment_type="image",
        filename="image_A.jpg",
        document_id="doc_A",
        image_asset_id="image_A",
    )

    result = ConversationResolverAgent().resolve(
        message="Explain this.",
        context={"active_document_id": "document_B", "active_document_name": "old-report.pdf"},
        history=[],
        documents=[old_doc],
        images=[],
        current_attachments=[attachment],
    )

    assert result["active_document_ids"] == ["doc_A"]
    assert result["selected_attachment"] == "attachment_A"
    assert result["reason"] == "current_message_attachments"


def test_followup_reuses_attachment_structured_evidence_without_search(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = SQLiteStore(settings.sqlite_path)
    memory = ConversationMemory(settings, store)
    memory.context_set(
        "s1",
        "w1",
        {
            "active_attachment_id": "attachment_A",
            "last_attachment_ids": ["attachment_A"],
            "active_document_id": "doc_A",
            "active_attachment_name": "image_A.jpg",
            "last_retrieved_chunks": [
                {
                    "chunk_id": "doc_A:0",
                    "document_id": "doc_A",
                    "document_name": "image_A.jpg.ocr.json",
                    "preview": "Glucose 95 mg/dL",
                    "score": 1.0,
                }
            ],
        },
    )
    tools = Mock()
    tools.document_search.side_effect = AssertionError("semantic search should not run")
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    agent = DocumentRAGAgent(tools, Mock(), memory)

    updates = agent.prepare(ChatRequest(message="Explain it simply.", session_id="s1", workspace_id="w1"), [], [])

    assert updates["tool_calls"] == ["retrieval_reuse"]
    assert updates["trace_steps"][-1]["search_bypassed"] is True
    assert updates["citations"][0].document_id == "doc_A"


def test_different_question_switches_to_medical_knowledge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = ConversationMemory(settings, SQLiteStore(settings.sqlite_path))
    memory.context_set("s1", "w1", {"active_attachment_id": "attachment_A", "active_document_id": "doc_A"})
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = memory
    agent.safety = SafetyLayer()
    agent.conversation_resolver_agent = ConversationResolverAgent()
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.store.workspace_revision.return_value = "1"
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = Mock()

    state = agent._observe_node({"request": ChatRequest(message="Different question: what causes anemia?", session_id="s1", workspace_id="w1")})
    route = agent._planned_route(state["request"].message, state["observations"])

    assert route == "medical_education"
    assert state["context_resolution"]["reason"] in {"explicit_general_topic", "standalone_general_education_query"}


def test_compare_these_uses_exact_two_current_attachments() -> None:
    tools = Mock()
    tools.attachment_document_ids.return_value = ["report_A", "report_B"]
    tools.compare_reports.return_value = CompareResponse(summary="Compared exactly A and B.", citations=[])
    agent = ReportComparisonAgent(tools)

    result = agent.run(
        ChatRequest(message="Compare these.", session_id="s1", workspace_id="w1", attachment_ids=["attachment_A", "attachment_B"]),
        [],
        [],
    )

    tools.compare_reports.assert_called_once_with(["report_A", "report_B"], workspace_id="w1")
    assert result["trace_steps"][-1]["document_ids"] == ["report_A", "report_B"]


def test_unauthorized_attachment_id_is_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(_settings(tmp_path).sqlite_path)
    store.create_message_attachment(
        attachment_id="attachment_other",
        conversation_id="other-session",
        workspace_id="w1",
        attachment_type="document",
        filename="other.pdf",
        mime_type="application/pdf",
        file_size=10,
        storage_path="private/path/other.pdf",
        document_id="doc-other",
    )

    try:
        store.require_message_attachments(
            attachment_ids=["attachment_other"],
            workspace_id="w1",
            conversation_id="s1",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized attachment should fail closed")


def test_low_confidence_groq_vision_falls_back_to_ocr_for_document_photo(tmp_path: Path) -> None:
    class LowConfidenceVision:
        def structure_document(self, **_):
            return {"confidence": 0.2, "summary": "uncertain"}

    settings = _settings(tmp_path)
    settings.document_vlm_base_url = "https://api.groq.com/openai/v1"
    settings.document_vlm_min_confidence = 0.55
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"image")

    result = DocumentUnderstandingAgent(settings, llm=LowConfidenceVision()).understand(
        document_id="doc_A",
        workspace_id="w1",
        filename="photo.jpg.ocr.json",
        content_type="application/x-ocr-json",
        file_path=path,
        extracted_text="Patient: Test Person\nGlucose 95 mg/dL",
        source_image_id="image_A",
        ocr_confidence=0.9,
    )

    assert result.extraction_source == "ocr_postprocess"
    assert any("low_confidence" in warning for warning in result.warnings)


def test_medgemma_unavailable_does_not_fake_radiology_interpretation() -> None:
    tools = Mock()
    tools.attachment_evidence_chunks.return_value = [
        RetrievedChunk(
            chunk_id="attachment:attachment_A",
            document_id="image_A",
            document_name="xray.jpg",
            text="Clinical image attachment: xray.jpg. No extracted document evidence or paired report is available.",
            score=0.0,
        )
    ]

    result = ImagingAgent(tools).run(
        ChatRequest(message="Explain this X-ray.", session_id="s1", workspace_id="w1", attachment_ids=["attachment_A"]),
        [],
        [],
    )

    assert result["answer"] == NO_IMAGE_EVIDENCE_ANSWER
    assert result["trace_steps"][-1]["fallback_reason"] == "medgemma_not_configured"


def test_expired_temporary_attachment_deletes_metadata_and_binary(tmp_path: Path) -> None:
    store = SQLiteStore(_settings(tmp_path).sqlite_path)
    binary = tmp_path / "temporary-upload.bin"
    binary.write_bytes(b"temporary")
    store.create_message_attachment(
        attachment_id="attachment_temp",
        conversation_id="s1",
        workspace_id="w1",
        attachment_type="document",
        filename="temp.txt",
        mime_type="text/plain",
        file_size=9,
        storage_path=str(binary),
        expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        persistence_mode="temporary",
    )

    deleted = store.delete_expired_temporary_attachments()

    assert [item.id for item in deleted] == ["attachment_temp"]
    assert store.get_message_attachment("attachment_temp") is None
    assert not binary.exists()


def test_saved_document_remains_when_chat_is_deleted(tmp_path: Path) -> None:
    store = SQLiteStore(_settings(tmp_path).sqlite_path)
    store.save_document(
        document_id="doc_saved",
        workspace_id="w1",
        filename="saved.pdf",
        content_type="application/pdf",
        file_path=tmp_path / "saved.pdf",
        summary="saved",
    )
    message_id = store.append_message("s1", "w1", "user", "Explain this.")
    store.create_message_attachment(
        attachment_id="attachment_saved",
        conversation_id="s1",
        workspace_id="w1",
        attachment_type="document",
        filename="saved.pdf",
        mime_type="application/pdf",
        file_size=10,
        storage_path=str(tmp_path / "saved.pdf"),
        document_id="doc_saved",
        persistence_mode="saved",
    )
    store.bind_attachments_to_message(
        attachment_ids=["attachment_saved"],
        message_id=message_id,
        workspace_id="w1",
        conversation_id="s1",
    )

    store.delete_session("s1", "w1")

    assert store.get_message_attachment("attachment_saved") is None
    assert store.get_document("doc_saved") is not None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        CAREMIND_SQLITE_PATH=tmp_path / "caremind.db",
        CAREMIND_DATA_DIR=tmp_path,
        CAREMIND_UPLOAD_DIR=tmp_path / "uploads",
        CAREMIND_VECTOR_BACKEND="sqlite",
        CAREMIND_RESPONSE_CACHE_ENABLED=False,
        PINECONE_API_KEY=None,
        supabase_url=None,
        supabase_secret_key=None,
    )
