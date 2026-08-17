from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.agent import CareMindAgent
from backend.caremind.schemas import ChatRequest


def make_agent(response_cache_enabled: bool = True) -> CareMindAgent:
    agent = CareMindAgent.__new__(CareMindAgent)
    agent.memory = Mock()
    agent.memory.settings = SimpleNamespace(response_cache_enabled=response_cache_enabled)
    agent.memory.cache_get.return_value = None
    agent.memory.semantic_cache_get.return_value = (None, 0.0)
    agent.tools = Mock()
    agent.tools.store.workspace_revision.return_value = "rev-1"
    agent.tools.embeddings.embed_query.return_value = [1.0, 0.0, 0.0]
    agent.tools.embeddings.dimension = 384
    agent.tools.embeddings.settings = SimpleNamespace(
        embedding_provider="auto",
        ollama_embedding_model=None,
        nvidia_api_key=None,
        nvidia_embedding_model="nvolveqa_40k",
        vector_backend="supabase",
        min_retrieval_similarity=0.5,
    )
    agent.llm = Mock()
    agent.llm.settings = SimpleNamespace(
        nvidia_api_key=None,
        nvidia_chat_model="nvidia/nemotron-3-ultra-550b-a55b",
    )
    return agent


def test_debug_request_bypasses_exact_and_semantic_cache() -> None:
    agent = make_agent()
    state = {
        "request": ChatRequest(message="Tell me about the patient Venkat Ramanujam", debug=True),
        "route": "retrieve",
        "cache_key": "exact-key",
        "cache_namespace": "namespace",
        "started_at": 0.0,
        "trace_steps": [],
    }

    result = agent._check_cache_node(state)

    agent.memory.cache_get.assert_not_called()
    agent.memory.semantic_cache_get.assert_not_called()
    assert result["query_embedding"] == [1.0, 0.0, 0.0]
    assert result["trace_steps"][-1]["cache_bypassed"] is True
    assert result["trace_steps"][-1]["debug"] is True
    assert result["trace_steps"][-1]["cache_debug"] == {}


def test_bypass_cache_request_bypasses_exact_and_semantic_cache() -> None:
    agent = make_agent()
    state = {
        "request": ChatRequest(message="Tell me about the patient Venkat Ramanujam", bypass_cache=True),
        "route": "retrieve",
        "cache_key": "exact-key",
        "cache_namespace": "namespace",
        "started_at": 0.0,
        "trace_steps": [],
    }

    result = agent._check_cache_node(state)

    agent.memory.cache_get.assert_not_called()
    agent.memory.semantic_cache_get.assert_not_called()
    assert result["trace_steps"][-1]["cache_bypassed"] is True
    assert result["trace_steps"][-1]["bypass_cache"] is True


def test_disabled_response_cache_bypasses_cache_even_without_request_flag() -> None:
    agent = make_agent(response_cache_enabled=False)
    state = {
        "request": ChatRequest(message="Tell me about the patient Venkat Ramanujam"),
        "route": "retrieve",
        "cache_key": "exact-key",
        "cache_namespace": "namespace",
        "started_at": 0.0,
        "trace_steps": [],
    }

    result = agent._check_cache_node(state)

    agent.memory.cache_get.assert_not_called()
    agent.memory.semantic_cache_get.assert_not_called()
    assert result["trace_steps"][-1]["cache_bypassed"] is True
    assert result["trace_steps"][-1]["response_cache_enabled"] is False


def test_normal_request_can_use_exact_cache() -> None:
    agent = make_agent()
    agent.memory.cache_get.return_value = {
        "session_id": "default",
        "route": "retrieve",
        "answer": "cached answer",
        "citations": [],
        "safety_notes": [],
        "tool_calls": ["document_search"],
        "trace": {
            "route": "retrieve",
            "cache_hit": False,
            "steps": [{"node": "document_rag_agent"}],
            "cache_metadata": {
                "cached_at": "2026-07-14T20:00:00+00:00",
                "query": "Tell me about the patient Venkat Ramanujam",
            },
        },
    }
    state = {
        "request": ChatRequest(message="Tell me about the patient Venkat Ramanujam"),
        "route": "retrieve",
        "cache_key": "exact-key",
        "cache_namespace": "namespace",
        "started_at": 0.0,
        "trace_steps": [],
    }

    result = agent._check_cache_node(state)

    agent.memory.cache_get.assert_called_once_with(
        "default",
        "exact-key",
        current_query="Tell me about the patient Venkat Ramanujam",
    )
    assert result["cache_hit"] is True
    assert result["response"].answer == "cached answer"
    assert result["response"].trace["cache_hit"] is True
    assert result["response"].trace["retrieval_bypassed"] is True
    assert result["response"].trace["generation_bypassed"] is True
    assert result["response"].trace["cached_response_trace"] == {"redacted": True}
    assert result["response"].trace["steps"][-1]["node"] == "return_cached_response"
    assert result["response"].trace["steps"][-1]["tool_calls_from_cached_response"] == ["document_search"]


def test_cache_key_changes_when_retrieval_or_model_settings_change() -> None:
    agent = make_agent()
    request = ChatRequest(message="Tell me about the patient Venkat Ramanujam", top_k=5)

    original_key = agent._cache_key(request, "retrieve")

    agent.tools.embeddings.settings.min_retrieval_similarity = 0.7
    assert agent._cache_key(request, "retrieve") != original_key

    agent.tools.embeddings.settings.min_retrieval_similarity = 0.5
    agent.tools.embeddings.settings.vector_backend = "sqlite"
    assert agent._cache_key(request, "retrieve") != original_key

    agent.tools.embeddings.settings.vector_backend = "supabase"
    agent.llm.settings.nvidia_api_key = "configured"
    assert agent._cache_key(request, "retrieve") != original_key


def test_cache_key_uses_resolved_entity_and_active_document_context() -> None:
    agent = make_agent()
    request = ChatRequest(message="Why does Mr. Venkat have fatigue?")
    base_context = {
        "active_patient": "Mr. Venkat Ramanujam",
        "active_document_name": "venkat-report.pdf",
        "active_document_ids": ["doc-1"],
        "conversation_entities": {
            "patient": [{"name": "Mr. Venkat Ramanujam", "id": "doc-1"}],
            "report": [{"name": "venkat-report.pdf", "id": "doc-1"}],
        },
    }

    original_key = agent._cache_key(request, "retrieve", base_context)

    changed_context = {
        **base_context,
        "active_patient": "Ms. Example Patient",
        "active_document_ids": ["doc-2"],
    }

    assert agent._cache_key(request, "retrieve", changed_context) != original_key


def test_cache_debug_summary_without_redis_reports_key_patterns() -> None:
    from backend.caremind.memory import ConversationMemory

    memory = ConversationMemory.__new__(ConversationMemory)
    memory.settings = SimpleNamespace(
        response_cache_enabled=True,
        semantic_cache_enabled=True,
        cache_ttl_seconds=60,
    )
    memory.redis_error = "ConnectionError"
    memory._redis = None

    summary = memory.cache_debug_summary("default", "session-1")

    assert summary["redis_enabled"] is False
    assert summary["session_redis_key"] == "caremind:default:session:session-1"
    assert summary["exact_cache_pattern"] == "caremind:default:cache:*"
    assert summary["semantic_cache_pattern"] == "caremind:default:semcache:*"
    assert summary["counts"] == {
        "session": 0,
        "exact_response_cache": 0,
        "semantic_response_cache": 0,
    }
