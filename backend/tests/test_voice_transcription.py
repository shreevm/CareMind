from pathlib import Path
from unittest.mock import Mock

from backend.caremind.agent import CareMindAgent
from backend.caremind.config import Settings
from backend.caremind.schemas import ChatRequest, TranscriptMetadata
from backend.caremind.store import SQLiteStore


def test_transcript_metadata_preserves_detected_language_fields() -> None:
    transcript = TranscriptMetadata(
        text="¿Qué muestran mis resultados?",
        confidence=0.91,
        detected_language="es",
        language_confidence=0.98,
        timestamps=[{"text": "Qué", "start": 0.1, "end": 0.3, "type": "word"}],
        modality="voice",
        input_modality="voice",
    )

    payload = transcript.model_dump(mode="json")

    assert payload["text"] == "¿Qué muestran mis resultados?"
    assert payload["detected_language"] == "es"
    assert payload["language_confidence"] == 0.98
    assert payload["timestamps"][0]["text"] == "Qué"


def test_voice_transcript_persistence_stores_scribe_language_metadata(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "caremind.db")
    message_id = store.append_message("s1", "w1", "user", "¿Qué muestran mis resultados?")
    transcript = TranscriptMetadata(
        text="¿Qué muestran mis resultados?",
        confidence=0.87,
        language="es",
        detected_language="es",
        language_confidence=0.99,
        timestamps=[{"text": "resultados", "start": 0.4, "end": 0.9, "type": "word"}],
        modality="voice",
        input_modality="voice",
    )

    transcript_id = store.save_voice_transcript(
        message_id=message_id,
        session_id="s1",
        workspace_id="w1",
        transcript=transcript,
    )

    with store.connect() as conn:
        row = conn.execute("select * from voice_transcripts where id = ?", (transcript_id,)).fetchone()

    assert row["message_id"] == message_id
    assert row["transcript"] == "¿Qué muestran mis resultados?"
    assert row["detected_language"] == "es"
    assert row["language_confidence"] == 0.99
    assert "resultados" in row["timestamps_json"]


def test_response_language_prefers_session_override_over_detected_language() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    request = ChatRequest(
        message="¿Qué muestran mis resultados?",
        modality="voice",
        transcript=TranscriptMetadata(text="¿Qué muestran mis resultados?", detected_language="es"),
    )

    assert agent._response_language(request, {"response_language_override": "fr"}) == "fr"
    assert agent._response_language(request, {}) == "es"


def test_observe_node_passes_transcript_metadata_unchanged_to_trace() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    transcript = TranscriptMetadata(
        text="bonjour docteur",
        confidence=0.93,
        detected_language="fr",
        language_confidence=0.97,
        modality="voice",
    )

    state = agent._observe_node(
        {
            "request": ChatRequest(
                message="bonjour docteur",
                modality="voice",
                transcript=transcript,
                session_id="s1",
                workspace_id="w1",
            ),
            "started_at": 0.0,
        }
    )

    assert state["request"].message == "bonjour docteur"
    assert state["request"].response_language == "fr"
    assert state["trace_steps"][0]["transcript"]["text"] == "bonjour docteur"
    assert state["trace_steps"][0]["detected_language"] == "fr"
