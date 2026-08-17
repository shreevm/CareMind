from pathlib import Path
from types import SimpleNamespace
import sys
import time
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agent import CareMindAgent
from backend.caremind.agents.supervisor import SupervisorAgent
from backend.caremind.config import Settings
from backend.caremind.memory import ConversationMemory
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest, Citation
from backend.caremind.store import SQLiteStore


def make_agent(tmp_path: Path) -> CareMindAgent:
    settings = Settings(
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
    memory = ConversationMemory(settings, SQLiteStore(settings.sqlite_path))
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = memory
    agent.safety = SafetyLayer()
    agent.supervisor_agent = SupervisorAgent()
    agent.tools = Mock()
    agent.tools.store.list_documents.return_value = [
        SimpleNamespace(filename="Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf")
    ]
    agent.tools.store.list_images.return_value = []
    agent.tools.store.workspace_revision.return_value = "1"
    agent.tools.external_literature_available.return_value = False
    return agent


def test_patient_name_question_routes_to_documents_when_documents_exist(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    request = ChatRequest(message="Tell me about Mr. Venkat Ramanujam.", session_id="s1", workspace_id="w1")

    state = agent._observe_node({"request": request, "started_at": time.perf_counter()})
    route = agent._planned_route(state["request"].message, state["observations"])

    assert route == "retrieve"
    assert state["context_resolution"]["used_context"] is False


def test_followup_questions_are_rewritten_and_routed_to_active_patient_report(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.memory.context_set(
        "s1",
        "w1",
        {
            "active_patient": "Mr. Venkat Ramanujam Sankar Ram",
            "active_document_name": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
            "active_report": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
        },
    )
    messages = [
        "What causes fatigue to him?",
        "Why was the test performed?",
        "Did he have chest pain?",
        "What happened next?",
        "I am asking about the patient earlier.",
    ]

    for message in messages:
        request = ChatRequest(message=message, session_id="s1", workspace_id="w1")
        state = agent._observe_node({"request": request, "started_at": time.perf_counter()})
        route = agent._planned_route(state["request"].message, state["observations"])

        assert route == "retrieve"
        assert state["context_resolution"]["used_context"] is True
        assert "Mr. Venkat Ramanujam Sankar Ram" in state["request"].message


def test_contextual_compare_followup_routes_to_comparison(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.memory.context_set(
        "s1",
        "w1",
        {
            "active_patient": "Mr. Venkat Ramanujam Sankar Ram",
            "active_document_name": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
            "active_document_id": "doc-current",
            "active_report": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
        },
    )

    state = agent._observe_node(
        {"request": ChatRequest(message="Compare it with the previous report.", session_id="s1", workspace_id="w1"), "started_at": time.perf_counter()}
    )

    assert state["context_resolution"]["used_context"] is True
    assert state["context_resolution"]["active_document_ids"] == ["doc-current"]
    assert agent._planned_route(state["request"].message, state["observations"]) == "compare"


def test_recap_style_intent_rewrite_routes_vague_comparison_followup(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.memory.context_set(
        "s1",
        "w1",
        {
            "active_patient": "Mr. Venkat Ramanujam Sankar Ram",
            "active_document_name": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
            "active_document_id": "doc-current",
            "active_report": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
        },
    )

    state = agent._observe_node(
        {
            "request": ChatRequest(
                message="Is this worse than last time? focus on kidney markers",
                session_id="s1",
                workspace_id="w1",
            ),
            "started_at": time.perf_counter(),
        }
    )
    route = agent._planned_route(state["request"].message, state["observations"])
    plan = agent._build_plan(state["request"].message, route, state["observations"])
    intent_rewrite = state["context_resolution"]["intent_rewrite"]

    assert route == "compare"
    assert intent_rewrite["route_hint"] == "compare"
    assert intent_rewrite["confidence"] >= 0.8
    assert "kidney markers" in intent_rewrite["constraints"]
    assert "Compare the current clinical report" in plan["current_intent"]
    assert plan["route_source"] == "intent_rewrite"


def test_contextual_chest_pain_question_does_not_trigger_emergency_redirect(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.memory.context_set(
        "s1",
        "w1",
        {
            "active_patient": "Mr. Venkat Ramanujam Sankar Ram",
            "active_document_name": "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
        },
    )

    response = agent._preflight_response(ChatRequest(message="Why chest pain?", session_id="s1", workspace_id="w1"))

    assert response is None


def test_active_patient_context_is_updated_from_retrieval_citations(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    citation = Citation(
        document_id="doc-1",
        document_name="Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf",
        chunk_id="doc-1:0",
        page=1,
        score=0.8,
        quote="Report for Mr. Venkat Ramanujam Sankar Ram. Reason For Test Unexplained Fatigue.",
    )
    state = {
        "request": ChatRequest(message="Tell me about Mr. Venkat Ramanujam.", session_id="s1", workspace_id="w1"),
        "route": "retrieve",
        "citations": [citation],
        "retrieved_chunks": [{"chunk_id": "doc-1:0", "preview": citation.quote}],
    }

    context = agent._updated_conversation_context(state, "Grounded answer [1]")

    assert context["active_patient"] == "Mr. Venkat Ramanujam Sankar Ram"
    assert context["active_document_id"] == "doc-1"
    assert context["active_document_name"] == "Report_-_Mr._Venkat_Ramanujam_Sankar_Ram_553217_.pdf"
