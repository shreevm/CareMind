from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agents.document_rag import (
    NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER,
    NO_REPORT_INSTRUCTIONS_ANSWER,
    DocumentRAGAgent,
)
from backend.caremind.schemas import ChatRequest, RetrievedChunk


def test_document_rag_filters_low_similarity_chunks_and_skips_generation() -> None:
    low_score_chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="cardiology-report.pdf",
            text="Sinus rhythm Avg HR.",
            score=0.07,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_name="cardiology-report.pdf",
            text="Supraventricular tachycardia Avg HR.",
            score=0.09,
        ),
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = low_score_chunks
    llm = Mock()
    memory = Mock()
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(
        ChatRequest(message="What food should someone with iron deficiency eat?"),
        tool_calls=[],
        trace_steps=[],
    )

    assert result["answer"] == NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER
    assert result["citations"] == []
    assert result["retrieved_chunks"] == []
    assert result["no_relevant_document_evidence"] is True
    llm.answer.assert_not_called()
    trace_step = result["trace_steps"][-1]
    assert trace_step["retrieved_count"] == 2
    assert trace_step["kept_count"] == 0
    assert trace_step["filtered_count"] == 2
    assert trace_step["below_threshold_count"] == 2
    assert trace_step["min_similarity"] == 0.5
    assert trace_step["counter_invariant_ok"] is True


def test_document_rag_generates_only_from_chunks_above_similarity_threshold() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="relevant.txt",
            text="Hemoglobin is low and iron studies were ordered.",
            score=0.82,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            document_name="noise.txt",
            text="Unrelated rhythm table.",
            score=0.12,
        ),
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = chunks
    llm = Mock()
    llm.answer.return_value = "Grounded answer [1]"
    memory = Mock()
    memory.load.return_value = []
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="What does the report say about iron?"), [], [])

    llm.answer.assert_called_once()
    assert llm.answer.call_args.kwargs["chunks"] == [chunks[0]]
    assert len(result["citations"]) == 1
    trace_step = result["trace_steps"][-1]
    assert trace_step["retrieved_count"] == 2
    assert trace_step["kept_count"] == 1
    assert trace_step["filtered_count"] == 1
    assert trace_step["below_threshold_count"] == 1


def test_document_rag_filters_broad_summary_chunks_when_similarity_is_low() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="report.txt",
            text="Patient reports fatigue. Hemoglobin is 11.2 g/dL. Iron studies were ordered.",
            score=0.08,
        )
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = chunks
    llm = Mock()
    memory = Mock()
    memory.load.return_value = []
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="What are the key findings?"), [], [])

    assert result["answer"] == NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER
    llm.answer.assert_not_called()
    trace_step = result["trace_steps"][-1]
    assert trace_step["below_threshold_count"] == 1
    assert trace_step["filtered_count"] == 1


def test_document_rag_does_not_override_threshold_for_findings_of_report() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="report.txt",
            text="Report for Mr. Venkat. Avg HR 80 bpm. Sinus rhythm noted. No symptoms logged.",
            score=0.3,
        )
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = chunks
    llm = Mock()
    memory = Mock()
    memory.load.return_value = []
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="What are u findings of the report?"), [], [])

    assert result["answer"] == NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER
    llm.answer.assert_not_called()
    trace_step = result["trace_steps"][-1]
    assert trace_step["kept_count"] == 0
    assert trace_step["filtered_count"] == 1


def test_document_rag_reuses_previous_active_chunks_for_followup() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    llm = Mock()
    llm.answer.return_value = "Fatigue answer [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {
        "active_patient": "Mr. Venkat Ramanujam",
        "active_document_id": "doc-1",
        "active_document_name": "venkat-report.pdf",
        "last_retrieved_chunks": [
            {
                "chunk_id": "doc-1:0",
                "document_id": "doc-1",
                "document_name": "venkat-report.pdf",
                "preview": "Report for Mr. Venkat Ramanujam. Reason for test: unexplained fatigue.",
                "score": 0.88,
            }
        ],
        "last_evidence_need": "specific_report_fact",
    }
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(
        ChatRequest(
            message="According to the uploaded report for Mr. Venkat Ramanujam, what explains the unexplained fatigue?"
        ),
        [],
        [],
    )

    tools.document_search.assert_not_called()
    llm.answer.assert_called_once()
    assert result["retrieval_reused"] is True
    assert result["tool_calls"] == ["retrieval_reuse"]
    assert result["citations"][0].document_id == "doc-1"
    assert result["trace_steps"][-1]["search_bypassed"] is True


def test_document_rag_does_not_reuse_when_evidence_need_changes() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = [
        RetrievedChunk(
            chunk_id="doc-1:plan",
            document_id="doc-1",
            document_name="venkat-report.pdf",
            text="Plan: follow up with the treating clinician.",
            score=0.8,
        )
    ]
    llm = Mock()
    llm.answer.return_value = "Plan answer [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {
        "active_document_id": "doc-1",
        "last_evidence_need": "key_findings",
        "last_retrieved_chunks": [
            {
                "chunk_id": "doc-1:findings",
                "document_id": "doc-1",
                "document_name": "venkat-report.pdf",
                "preview": "Heart rate average 80 bpm. Sinus rhythm noted.",
                "score": 0.9,
            }
        ],
    }
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="What does the report say about fatigue?"), [], [])

    tools.document_search.assert_called_once()
    assert result["tool_calls"] == ["document_search"]


def test_document_rag_reuses_previous_chunks_for_vague_contextual_followup() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    llm = Mock()
    llm.answer.return_value = "Explanation [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {
        "active_document_id": "doc-1",
        "active_document_name": "followup-report.txt",
        "last_evidence_need": "key_findings",
        "last_retrieved_chunks": [
            {
                "chunk_id": "doc-1:0",
                "document_id": "doc-1",
                "document_name": "followup-report.txt",
                "preview": "Fatigue has improved. Hemoglobin is 12.6 g/dL.",
                "score": 0.9,
            }
        ],
    }
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="According to the previously discussed uploaded report, what does that mean?"), [], [])

    tools.document_search.assert_not_called()
    assert result["tool_calls"] == ["retrieval_reuse"]
    assert result["citations"][0].chunk_id == "doc-1:0"


def test_document_rag_scans_active_document_for_report_mention_question() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="doc-1:0",
            document_id="doc-1",
            document_name="followup-report.txt",
            text="No pneumonia diagnosis is documented.",
            score=None,
        )
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.store.get_document_chunks.return_value = chunks
    llm = Mock()
    llm.answer.return_value = "The report says no pneumonia diagnosis is documented. [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {"active_document_id": "doc-1"}
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="Does my report mention pneumonia?"), [], [])

    tools.document_search.assert_not_called()
    tools.store.get_document_chunks.assert_called_once_with("doc-1")
    assert result["tool_calls"] == ["active_document_lookup"]
    assert result["citations"][0].chunk_id == "doc-1:0"


def test_report_guidance_changed_evidence_need_bypasses_reuse_and_searches_active_document() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.store.get_document_chunks.return_value = [
        RetrievedChunk(
            chunk_id="doc-1:findings",
            document_id="doc-1",
            document_name="venkat-report.pdf",
            text="Heart rate average 80 bpm. Sinus rhythm noted.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="doc-1:plan",
            document_id="doc-1",
            document_name="venkat-report.pdf",
            text="Conclusion: review the cardiac monitoring report with the treating clinician.",
            score=0.9,
        ),
    ]
    tools.medical_education_search.return_value = []
    llm = Mock()
    llm.answer.return_value = "Report instructions answer [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {
        "active_document_id": "doc-1",
        "last_retrieved_chunks": [
            {
                "chunk_id": "doc-1:findings",
                "document_id": "doc-1",
                "document_name": "venkat-report.pdf",
                "preview": "Heart rate average 80 bpm. Sinus rhythm noted.",
                "score": 0.9,
            }
        ],
    }
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="From the report, what should the patient do?"), [], [])

    tools.document_search.assert_not_called()
    tools.store.get_document_chunks.assert_called_once_with("doc-1")
    llm.answer.assert_called_once()
    assert result["tool_calls"] == ["document_instruction_search"]
    assert result["citations"][0].chunk_id == "doc-1:plan"
    assert result["trace_steps"][-1]["report_guidance_workflow"] is True
    assert result["trace_steps"][-1]["report_instruction_count"] == 1


def test_report_guidance_uses_general_education_fallback_when_no_report_instructions() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.store.get_document_chunks.return_value = [
        RetrievedChunk(
            chunk_id="doc-1:findings",
            document_id="doc-1",
            document_name="venkat-report.pdf",
            text="Heart rate average 80 bpm. Sinus rhythm noted.",
            score=0.9,
        )
    ]
    education_chunk = RetrievedChunk(
        chunk_id="education:medical-report-follow-up",
        document_id="general-medical-education",
        document_name="General Education - Medical Report Follow-Up",
        text="Review the report with the clinician who ordered the test.",
        score=0.88,
    )
    tools.medical_education_search.return_value = [education_chunk]
    llm = Mock()
    llm.answer.return_value = "No instructions were found. General next steps [1]"
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {"active_document_id": "doc-1"}
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="From the report, what must the patient do?"), [], [])

    llm.answer.assert_called_once()
    assert result["tool_calls"] == ["document_instruction_search", "medical_education_search"]
    assert result["citations"][0].document_id == "general-medical-education"
    assert result["trace_steps"][-1]["education_fallback_used"] is True


def test_report_guidance_does_not_fabricate_general_steps_without_education() -> None:
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.store.get_document_chunks.return_value = [
        RetrievedChunk(
            chunk_id="doc-1:findings",
            document_id="doc-1",
            document_name="venkat-report.pdf",
            text="Heart rate average 80 bpm. Sinus rhythm noted.",
            score=0.9,
        )
    ]
    tools.medical_education_search.return_value = []
    llm = Mock()
    memory = Mock()
    memory.load.return_value = []
    memory.context_get.return_value = {"active_document_id": "doc-1"}
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="From the report, what must the patient do?"), [], [])

    assert result["answer"] == NO_REPORT_INSTRUCTIONS_ANSWER
    assert result["citations"] == []
    llm.answer.assert_not_called()


def test_followup_report_label_is_not_report_guidance_query() -> None:
    tools = Mock()
    llm = Mock()
    memory = Mock()
    agent = DocumentRAGAgent(tools, llm, memory)

    assert agent._is_report_guidance_query("Summarize the synthetic follow-up report.") is False
    assert agent._is_report_guidance_query("What should the patient do next from the report?") is True


def test_document_rag_deduplicates_identical_chunks_before_generation() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="report.txt",
            text="Hemoglobin is low and iron studies were ordered.",
            page=1,
            score=0.82,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_name="report.txt",
            text="Hemoglobin is low and iron studies were ordered.",
            page=1,
            score=0.81,
        ),
    ]
    tools = Mock()
    tools.embeddings.settings.min_retrieval_similarity = 0.5
    tools.document_search.return_value = chunks
    llm = Mock()
    llm.answer.return_value = "Grounded answer [1]"
    memory = Mock()
    memory.load.return_value = []
    agent = DocumentRAGAgent(tools, llm, memory)

    result = agent.run(ChatRequest(message="What does the report say about iron?"), [], [])

    assert len(llm.answer.call_args.kwargs["chunks"]) == 1
    assert len(result["citations"]) == 1
    trace_step = result["trace_steps"][-1]
    assert trace_step["filter_reasons"]["duplicate"] == 1
    assert trace_step["kept_count"] == 1
