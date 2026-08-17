from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agents.medical_education import MedicalEducationAgent
from backend.caremind.config import Settings
from backend.caremind.schemas import ChatRequest, RetrievedChunk
from backend.caremind.tools import DocumentTools


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "CAREMIND_SQLITE_PATH": tmp_path / "caremind.db",
        "CAREMIND_DATA_DIR": tmp_path,
        "CAREMIND_UPLOAD_DIR": tmp_path / "uploads",
        "supabase_url": None,
        "supabase_secret_key": None,
        "PINECONE_API_KEY": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_medical_education_uses_medlineplus_and_model_context_when_evidence_is_thin(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_MIN_CHARS=500)
    tools = Mock()
    tools.embeddings.settings = settings
    tools.medlineplus_available.return_value = True
    tools.medical_education_search.return_value = [
        RetrievedChunk(
            chunk_id="education:pneumonia",
            document_id="education:pneumonia",
            document_name="General Education - Pneumonia",
            text="Pneumonia is an infection or inflammation of lung tissue.",
        )
    ]
    tools.medlineplus_health_topic_search.return_value = []
    tools.external_literature_search.return_value = []
    memory = Mock()
    memory.load.return_value = []
    llm = Mock()

    updates = MedicalEducationAgent(tools, llm, memory).prepare(
        ChatRequest(message="Explain pneumonia"),
        [],
        [],
    )

    tools.medlineplus_health_topic_search.assert_called_once()
    assert updates["medical_model_context_used"] is True
    assert "medlineplus_health_topic_search" in updates["tool_calls"]


def test_medical_education_passes_model_context_flag_to_llm(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_MIN_CHARS=500)
    tools = Mock()
    tools.embeddings.settings = settings
    tools.medlineplus_available.return_value = False
    tools.medical_education_search.return_value = []
    tools.external_literature_search.return_value = []
    memory = Mock()
    memory.load.return_value = []
    llm = Mock()
    llm.answer_medical.return_value = "Educational answer."

    response = MedicalEducationAgent(tools, llm, memory).run(
        ChatRequest(message="Explain pneumonia"),
        [],
        [],
    )

    assert response["answer"] == "Educational answer."
    llm.answer_medical.assert_called_once()
    assert llm.answer_medical.call_args.kwargs["allow_model_context"] is True


def test_medical_education_search_filters_low_score_unrelated_chunks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, CAREMIND_MIN_RETRIEVAL_SIMILARITY=0.5)
    tools = DocumentTools.__new__(DocumentTools)
    tools.embeddings = Mock()
    tools.embeddings.settings = settings
    tools.education = Mock()
    tools.education.search.return_value = [
        RetrievedChunk(
            chunk_id="education:pneumonia",
            document_id="general-medical-education",
            document_name="General Education - Pneumonia",
            text="Pneumonia is an infection of the lung tissue.",
            score=0.63,
        ),
        RetrievedChunk(
            chunk_id="education:tuberculosis",
            document_id="general-medical-education",
            document_name="General Education - Tuberculosis",
            text="Tuberculosis can cause chronic cough and night sweats.",
            score=0.43,
        ),
    ]

    results = tools.medical_education_search("Explain pneumonia", top_k=3)

    assert [chunk.chunk_id for chunk in results] == ["education:pneumonia"]
    assert results[0].metadata["source"] == "medical_education"
