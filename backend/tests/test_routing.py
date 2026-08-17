from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agent import CareMindAgent
from backend.caremind.agents.supervisor import SupervisorAgent
from backend.caremind.education import MedicalEducationStore
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest


def test_supervisor_routes_iron_deficiency_food_question_to_education() -> None:
    message = "Tell about what food intake a person shpuld taken when he/she lacks iron difiecny?"

    assert SupervisorAgent().route(message) == "medical_education"


def test_general_typo_heavy_iron_food_question_is_not_patient_document_question() -> None:
    message = "Tell about what food intake a person shpuld taken when he/she lacks iron difiecny?"
    agent = CareMindAgent.__new__(CareMindAgent)

    assert agent._asks_general_not_patient(message)


def test_report_scoped_nutrition_question_still_uses_documents() -> None:
    message = "Based on this report, what food intake is relevant for this patient's iron deficiency?"

    assert SupervisorAgent().route(message) == "retrieve"


def test_supervisor_labeled_routing_set() -> None:
    cases = [
        ("What food should someone with iron deficiency eat?", "medical_education"),
        ("What foods are recommended based on the iron levels in this report?", "retrieve"),
        ("What are the key findings in my uploaded document?", "retrieve"),
        ("Explain what sinus rhythm means", "medical_education"),
        ("What does the sinus rhythm reading in my results indicate?", "retrieve"),
        ("What's the difference between ARIMA and VAR for forecasting?", "clarify"),
        ("Compare these two lab reports and tell me what changed", "compare"),
        ("What's a normal resting heart rate range for adults?", "medical_education"),
        ("I have chest pain radiating to my arm, what should I do?", "emergency_redirect"),
        ("What's the standard nursing protocol for monitoring post-op patients?", "medical_education"),
        ("How much ibuprofen should I take for a headache?", "emergency_redirect"),
        ("What does this X-ray show?", "retrieve"),
        ("how does one get testicular torsion", "emergency_redirect"),
    ]

    supervisor = SupervisorAgent()

    for query, route in cases:
        assert supervisor.route(query) == route


def test_tuberculosis_symptom_question_retrieves_tuberculosis_education() -> None:
    class Embeddings:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "tuberculosis" in text.lower() else [0.0, 1.0] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0] if "tuberculosis" in text.lower() else [0.0, 1.0]

    chunks = MedicalEducationStore(Embeddings()).search("symptoms of having tuberculosis", top_k=1)

    assert chunks[0].chunk_id == "education:tuberculosis"
    assert "night sweats" in chunks[0].text


def test_documents_existing_alone_do_not_force_general_question_to_retrieve() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="how does one get pneumonia"), "started_at": 0.0}
    )

    assert state["observations"]["document_count"] == 1
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_active_context_does_not_capture_standalone_general_education_question() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "active_patient": "Mr. Example Patient",
        "active_document_name": "uploaded-report.pdf",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="Explain pneumonia symptoms in simple language."), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is False
    assert state["context_resolution"]["reason"] == "standalone_general_education_query"
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_active_context_does_not_capture_general_topic_followup_wording() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "active_patient": "Mr. Example Patient",
        "active_document_name": "uploaded-report.pdf",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="What about pneumonia?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is False
    assert state["context_resolution"]["reason"] == "standalone_general_education_query"
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_document_scoped_medical_topic_still_uses_uploaded_evidence() -> None:
    assert SupervisorAgent().route("Does my report mention pneumonia?") == "retrieve"
    assert SupervisorAgent().route("What does the uploaded document say about cholesterol?") == "retrieve"


def test_common_general_medical_questions_route_to_education_without_disease_rules() -> None:
    supervisor = SupervisorAgent()

    assert supervisor.route("What foods help lower cholesterol?") == "medical_education"
    assert supervisor.route("What does a CBC test check for?") == "medical_education"
    assert supervisor.route("Tell me about blood pressure in general") == "medical_education"


def test_no_context_typo_heavy_low_iron_symptoms_routes_to_education() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="tell me about sympotms of hvaing less iron content?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is False
    assert state["context_resolution"]["reason"] == "standalone_general_education_query"
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_no_context_bare_iron_content_routes_to_education() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="tell me about iron content?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["reason"] == "standalone_general_education_query"
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_report_scoped_low_iron_question_still_uses_documents() -> None:
    assert SupervisorAgent().route("Does my report show less iron content?") == "retrieve"
    assert SupervisorAgent().route("What tablets are mentioned in my report?") == "retrieve"


def test_general_education_followup_wins_over_stale_document_context() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "conversation_focus": "general_medical_education",
        "current_topic": "low iron content in the body",
        "active_patient": "Mr. Example Patient",
        "active_document_id": "doc-1",
        "active_document_name": "uploaded-report.pdf",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(document_id="doc-1", filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="so wats the cure for that?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is True
    assert state["context_resolution"]["reason"] == "general_education_contextual_followup"
    assert "low iron" in state["request"].message.lower()
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_general_education_otc_followup_reuses_previous_topic() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "conversation_focus": "general_medical_education",
        "current_topic": "low iron content in the body",
        "active_patient": "Mr. Example Patient",
        "active_document_id": "doc-1",
        "active_document_name": "uploaded-report.pdf",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(document_id="doc-1", filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="any over the counter tablets please?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is True
    assert state["context_resolution"]["reason"] == "general_education_contextual_followup"
    assert "low iron" in state["request"].message.lower()
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_typo_heavy_otc_supplement_question_routes_to_education() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {
            "request": ChatRequest(message="any over the counter tablets please for deficieny in rion?"),
            "started_at": 0.0,
        }
    )

    assert state["context_resolution"]["reason"] == "standalone_general_education_query"
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_general_medical_followup_uses_previous_education_topic() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "conversation_focus": "general_medical_education",
        "current_topic": "What is pneumonia?",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="What causes it?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is True
    assert state["context_resolution"]["reason"] == "general_education_contextual_followup"
    assert "pneumonia" in state["request"].message.lower()
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_new_general_medical_topic_does_not_reuse_old_education_topic() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {
        "conversation_focus": "general_medical_education",
        "current_topic": "What is diabetes?",
    }
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = []
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="What causes pneumonia?"), "started_at": 0.0}
    )

    assert state["context_resolution"]["used_context"] is False
    assert "diabetes" not in state["request"].message.lower()
    assert agent._planned_route(state["request"].message, state["observations"]) == "medical_education"


def test_planner_routes_xray_question_to_imaging_agent() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.load.return_value = []
    agent.memory.context_get.return_value = {}
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [Mock(filename="uploaded-report.pdf")]
    agent.tools.store.list_images.return_value = []
    agent.tools.external_literature_available.return_value = False
    agent.supervisor_agent = SupervisorAgent()
    agent.safety = SafetyLayer()

    state = agent._observe_node(
        {"request": ChatRequest(message="What does this X-ray show?"), "started_at": 0.0}
    )

    assert agent._planned_route(state["request"].message, state["observations"]) == "imaging"


def test_emergency_redirect_preflight_bypasses_generation_and_retrieval() -> None:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.safety = SafetyLayer()
    agent.memory = Mock()
    agent.memory.context_get.return_value = {}

    response = agent._preflight_response(
        ChatRequest(message="I have chest pain radiating to my arm, what should I do?")
    )

    assert response is not None
    assert response.route == "emergency_redirect"
    assert response.tool_calls == []
    assert response.trace["generation_bypassed"] is True
