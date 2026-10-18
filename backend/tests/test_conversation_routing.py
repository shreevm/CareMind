"""Deterministic control-flow regressions; these do not measure live model accuracy."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.caremind.agent import CareMindAgent
from backend.caremind.config import Settings
from backend.caremind.llm import LLMClient
from backend.caremind.memory import ConversationMemory
from backend.caremind.routing import decide
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest, RetrievedChunk
from backend.caremind.store import SQLiteStore


def decision(route, query, confidence=0.95):
    return dict(route=route, scope={"retrieve": "documents", "medical_education": "general",
                "clarify": "uncertain"}[route], resolved_query=query, topic="iron deficiency",
                confidence=confidence, clarification="Which topic or report do you mean?" if route == "clarify" else "")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(ConversationMemory, "_connect_redis", lambda self: None)
    settings = Settings(_env_file=None, CAREMIND_SQLITE_PATH=tmp_path / "test.db",
                        CAREMIND_RESPONSE_CACHE_ENABLED=False, NVIDIA_API_KEY=None,
                        CAREMIND_VECTOR_BACKEND="sqlite", CAREMIND_EMBEDDING_PROVIDER="local")
    store = SQLiteStore(settings.sqlite_path)
    store.save_document(document_id="synthetic-report", workspace_id="w", filename="synthetic.txt",
                        content_type="text/plain", file_path=tmp_path / "synthetic.txt", summary=None)
    store.create_message_attachment(attachment_id="a", conversation_id="s", workspace_id="w",
                                    attachment_type="document", filename="synthetic.txt", mime_type="text/plain",
                                    file_size=40, storage_path="synthetic.txt", document_id="synthetic-report")
    memory = ConversationMemory(settings, store)
    tools = Mock()
    tools.store = store
    tools.embeddings.settings = settings
    tools.embeddings.degraded = False
    tools.embeddings.degraded_reason = ""
    tools.embeddings.last_provider = "local"
    tools.embeddings.dimension = 2
    tools.embeddings.embed_query.return_value = [1.0, 0.0]
    tools.external_literature_available.return_value = False
    tools.medlineplus_available.return_value = False
    report = RetrievedChunk(chunk_id="r:0", document_id="synthetic-report", document_name="synthetic.txt",
                            text="Synthetic patient Rowan has iron deficiency.", score=1.0)
    education = RetrievedChunk(chunk_id="e:0", document_id="education", document_name="Education",
                               text="Iron deficiency means depleted iron stores.", score=1.0)
    tools.attachment_evidence_chunks.return_value = [report]
    tools.medical_education_search.return_value = [education]
    tools.document_search.side_effect = AssertionError("Unexpected workspace document search")
    llm = LLMClient(settings)
    llm.route_conversation = Mock()
    llm.answer = Mock(return_value="Synthetic patient Rowan has iron deficiency [1].")
    llm.answer_medical = Mock(return_value="Iron deficiency means depleted iron stores [1].")
    agent = CareMindAgent(tools, llm, memory, SafetyLayer())
    return agent, tools, llm, memory


def request(message, attachments=None):
    return ChatRequest(message=message, session_id="s", workspace_id="w", attachment_ids=attachments or [], debug=True)


def test_report_general_correction_topic_followup_and_return(rig):
    agent, tools, llm, memory = rig
    turns = [
        ("Explain my uploaded report", "retrieve", "Explain my uploaded report", ["a"]),
        ("What causes iron deficiency in general?", "medical_education", "What causes iron deficiency in general?", ["a"]),
        ("I'm not asking about this patient.", "medical_education", "What causes iron deficiency in general?", []),
        ("How is that deficiency prevented in general?", "medical_education", "How is iron deficiency prevented in general?", []),
        ("Back to the report: what does it say?", "retrieve", "What does the previously uploaded attachment synthetic.txt say?", []),
        ("What about the other one?", "clarify", "What about the other one?", []),
    ]
    for message, route, resolved, attachments in turns:
        tools.reset_mock()
        llm.answer.reset_mock()
        llm.answer_medical.reset_mock()
        llm.route_conversation.return_value = decision(route, resolved)
        response = agent.answer(request(message, attachments))
        assert response.trace["internal_route"] == route
        payload = llm.route_conversation.call_args.args[0]
        assert payload["latest_message"] == message
        assert payload["available_documents"][0]["id"] == "synthetic-report"
        assert bool(payload["current_message_attachments"]) == (message == turns[0][0])
        if route == "medical_education":
            tools.medical_education_search.assert_called_once()
            tools.attachment_evidence_chunks.assert_not_called()
            tools.document_search.assert_not_called()
            llm.answer.assert_not_called()
            args = llm.answer_medical.call_args.kwargs
            assert args["question"] == resolved
            assert args["history"] == []
            assert all(chunk.document_id == "education" for chunk in args["chunks"])
            assert memory.context_get("s", "w")["active_document_id"] == "synthetic-report"
            resolution = agent._resolve_contextual_query(message, memory.context_get("s", "w"), [])
            assert resolution["active_document_ids"] == []
        elif route == "retrieve":
            tools.attachment_evidence_chunks.assert_called_once()
            tools.medical_education_search.assert_not_called()
            assert llm.answer.call_args.kwargs["chunks"][0].document_id == "synthetic-report"
        else:
            tools.attachment_evidence_chunks.assert_not_called()
            tools.medical_education_search.assert_not_called()
    assert "iron deficiency" in payload["context"]["current_topic"]


@pytest.mark.parametrize("streaming", [False, True])
def test_scope_failure_repairs_actual_agent_before_publishing(rig, monkeypatch, streaming):
    agent, tools, llm, memory = rig
    llm.route_conversation.return_value = decision("medical_education", "What causes iron deficiency in general?")
    # Inject a dispatch regression to exercise validation rather than just testing the router.
    monkeypatch.setattr(agent, "_planned_route", lambda *args: "retrieve")
    rejected = "Rejected patient draft with an invalid citation [99]."
    def wrong_document_run(*args):
        from backend.caremind.tools import citations_from_chunks
        chunks = tools.attachment_evidence_chunks(["a"], workspace_id="w", conversation_id="s")
        return dict(answer=rejected, citations=citations_from_chunks(chunks), evidence_chunks=chunks,
                    retrieved_chunks=[dict(document_id="synthetic-report")], tool_calls=["message_attachment_lookup"])
    monkeypatch.setattr(agent.document_rag_agent, "run", wrong_document_run)
    req = request("What causes iron deficiency in general?")
    if streaming:
        events = list(agent.stream_events(req))
        response = next(item["data"] for item in events if item["event"] == "final")
        assert rejected not in str(events)
        trace = response["trace"]
    else:
        response = agent.answer(req)
        trace = response.trace
        assert rejected not in response.answer
    repairs = [step for step in trace["steps"] if step["node"] == "repair"]
    assert len(repairs) == 1
    assert repairs[0]["route"] == "medical_education"
    assert trace["internal_route"] == "medical_education"
    tools.attachment_evidence_chunks.assert_called_once()
    tools.medical_education_search.assert_called_once()
    assert llm.answer_medical.call_args.kwargs["history"] == []
    assert rejected not in str(memory.load("s", "w"))


def test_failed_repair_is_bounded_and_draft_is_discarded(rig):
    agent, tools, llm, memory = rig
    llm.route_conversation.return_value = decision("medical_education", "Explain iron deficiency")
    llm.answer_medical.return_value = "Let's draft an answer: secret internal text"
    events = list(agent.stream_events(request("Explain iron deficiency")))
    response = events[-1]["data"]
    assert response["trace"]["internal_route"] == "clarify"
    assert len([s for s in response["trace"]["steps"] if s["node"] == "repair"]) == 1
    assert "secret internal text" not in str(events)
    assert "secret internal text" not in str(memory.load("s", "w"))


@pytest.mark.parametrize("output", [
    {"route": "unknown"}, decision("retrieve", "Explain report", confidence=0.4),
    {**decision("retrieve", "Explain report"), "scope": "general"},
])
def test_invalid_or_uncertain_decision_clarifies(output):
    llm = Mock()
    llm.route_conversation.return_value = output
    assert decide(llm, {"latest_message": "Explain it"}).route == "clarify"


def test_unavailable_model_and_generation_fallback_do_not_dump_evidence(rig):
    agent, tools, llm, _ = rig
    llm.route_conversation.side_effect = RuntimeError("offline")
    assert agent.answer(request("Explain it")).trace["internal_route"] == "clarify"
    answer = llm._answer_locally(question="Explain", chunks=tools.attachment_evidence_chunks.return_value)
    assert "Rowan" not in answer
    assert "try again" in answer


def test_raw_evidence_answer_is_repaired_to_concise_grounded_summary(rig):
    agent, tools, llm, memory = rig
    chunks = [
        RetrievedChunk(
            chunk_id="synthetic-report:0",
            document_id="synthetic-report",
            document_name="synthetic.txt",
            text=(
                "Patient name: Rowan Example\n"
                "Finding: Hemoglobin 10 g/dL, below the stated reference range.\n"
                "Measurement: ferritin: ??.\n"
                "Finding: Hemoglobin 10 g/dL, below the stated reference range."
            ),
            score=1.0,
        )
    ]
    tools.attachment_evidence_chunks.return_value = chunks
    llm.route_conversation.return_value = decision("retrieve", "Summarize the uploaded patient report")
    llm.answer.return_value = (
        '{"patient":{"name":"Rowan Example"},'
        '"measurements":[{"ferritin":"??"}],'
        '"findings":["Hemoglobin 10 g/dL"]}'
    )

    response = agent.answer(request("Summarize the uploaded patient report", ["a"]))

    assert response.trace["internal_route"] == "retrieve"
    assert "{" not in response.answer
    assert response.answer.count("Hemoglobin 10 g/dL") == 1
    assert "unclear value" in response.answer
    assert "verify it in the source report" in response.answer.lower()
    assert "ferritin" in response.answer
    assert len([step for step in response.trace["steps"] if step["node"] == "repair"]) == 1
    assert any(step.get("answer_source") == "safe_summary_repair" for step in response.trace["steps"])
    assert "measurements" not in memory.load("s", "w")[-1].content
