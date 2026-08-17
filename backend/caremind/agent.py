import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterator, TypedDict

from langgraph.graph import END, StateGraph

from .agents import (
    ClarificationAgent,
    ConversationResolverAgent,
    DirectResponseAgent,
    DocumentRAGAgent,
    ImagingAgent,
    MedicalEducationAgent,
    ReportComparisonAgent,
    Route,
    SupervisorAgent,
)
from .llm import LLMClient
from .memory import ConversationMemory
from .metrics import metrics
from .observability import chat_request_inputs, chat_response_outputs, trace_block
from .intent import asks_general_medical_not_document, explicitly_general_topic
from .safety import SafetyLayer
from .schemas import ChatRequest, ChatResponse, Citation
from .tools import DocumentTools


class AgentState(TypedDict, total=False):
    request: ChatRequest
    original_message: str
    context_resolution: dict[str, Any]
    session_state: dict[str, Any]
    conversation_entities: dict[str, Any]
    started_at: float
    observations: dict[str, Any]
    plan: dict[str, Any]
    validation: dict[str, Any]
    repair_count: int
    route: Route
    cache_key: str
    cache_namespace: str
    query_embedding: list[float]
    cache_hit: bool
    answer: str
    citations: list[Citation]
    safety_notes: list[str]
    tool_calls: list[str]
    require_citations: bool
    retrieved_chunks: list[dict]
    no_relevant_document_evidence: bool
    trace_steps: list[dict]
    response: ChatResponse


class CareMindAgent:
    """
    CareMind multi-agent system for medical document Q&A.

    Architecture:
        - SupervisorAgent: Routes queries to specialized agents
        - DocumentRAGAgent: Answers questions about uploaded documents
        - MedicalEducationAgent: Provides medical education
        - ImagingAgent: Handles imaging-related queries
        - ReportComparisonAgent: Compares multiple documents
        - ClarificationAgent: Handles ambiguous queries
        - DirectResponseAgent: Handles direct information requests

    Workflow:
        1. Preflight checks (emergency, injection, confidence)
        2. Route via SupervisorAgent
        3. Generate response with specialized agent
        4. Safety finalization
        5. Cache storage
        6. Return to user

    Key Features:
        - Multi-turn conversation memory
        - Response caching with semantic similarity
        - Safety guardrails (emergency detection, prompt injection)
        - Citation tracking from documents
        - Comprehensive logging and tracing
    """

    def __init__(
        self,
        tools: DocumentTools,
        llm: LLMClient,
        memory: ConversationMemory,
        safety: SafetyLayer,
    ):
        self.tools = tools
        self.llm = llm
        self.memory = memory
        self.safety = safety
        self.conversation_resolver_agent = ConversationResolverAgent()
        self.supervisor_agent = SupervisorAgent()
        self.direct_agent = DirectResponseAgent()
        self.clarification_agent = ClarificationAgent()
        self.document_rag_agent = DocumentRAGAgent(tools, llm, memory)
        self.imaging_agent = ImagingAgent(tools)
        self.medical_education_agent = MedicalEducationAgent(tools, llm, memory)
        self.report_comparison_agent = ReportComparisonAgent(tools)
        self.graph = self._build_graph()

    def answer(self, request: ChatRequest) -> ChatResponse:
        """
        Generate answer to user's medical query.

        Workflow:
            1. Validate input (preflight checks)
            2. Check for emergencies (911, critical symptoms)
            3. Detect prompt injection attempts
            4. Check transcript confidence if voice input
            5. Route to appropriate agent via SupervisorAgent
            6. Generate response
            7. Finalize with safety checks
            8. Cache response for future use

        Args:
            request: ChatRequest with message, session, workspace IDs

        Returns:
            ChatResponse with answer, route, citations, trace

        Raises:
            Exception: Any uncaught errors during processing
        """
        with trace_block(
            self.llm.settings,
            "CareMindAgent.answer",
            "chain",
            chat_request_inputs(self.llm.settings, request),
        ) as run:
            preflight = self._preflight_response(request)
            if preflight is not None:
                run.end(chat_response_outputs(self.llm.settings, preflight))
                return preflight
            state = self.graph.invoke(
                {"request": request, "started_at": time.perf_counter()},
                config=self._run_config(request),
            )
            response = state["response"]
            run.end(chat_response_outputs(self.llm.settings, response))
            return response

    def stream_events(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        started_at = time.perf_counter()
        yield {"event": "start", "data": {"session_id": request.session_id, "workspace_id": request.workspace_id}}
        preflight = self._preflight_response(request)
        if preflight is not None:
            yield from self._stream_completed_response(preflight, started_at)
            return

        state: AgentState = {"request": request, "started_at": started_at}
        state = self._observe_node(state)
        state = self._conversation_resolver_node(state)
        state = self._plan_node(state)
        yield {
            "event": "route",
            "data": {
                "route": self._public_route(state["route"], state["request"]),
                "internal_route": state["route"],
            },
        }
        state = self._check_cache_node(state)
        if state.get("cache_hit"):
            yield from self._stream_completed_response(state["response"], started_at)
            return

        route = state["route"]
        if route == "retrieve":
            updates = self.document_rag_agent.prepare(
                state["request"],
                state.get("tool_calls", []),
                state.get("trace_steps", []),
            )
            state = {**state, **updates}
            if "answer" not in state:
                yield from self._stream_llm_answer(state, medical=False, started_at=started_at)
        elif route == "medical_education":
            updates = self.medical_education_agent.prepare(
                state["request"],
                state.get("tool_calls", []),
                state.get("trace_steps", []),
            )
            state = {**state, **updates}
            yield from self._stream_llm_answer(state, medical=True, started_at=started_at)
        else:
            node = {
                "direct": self._direct_node,
                "clarify": self._clarify_node,
                "emergency_redirect": self._emergency_redirect_node,
                "compare": self._compare_node,
                "imaging": self._imaging_node,
            }.get(route)
            if node is None:
                state = {**state, "answer": self.clarification_agent.answer(), "route": "clarify"}
            else:
                state = node(state)

        if "response" not in state:
            state = self._validate_node(state)
            state = self._finalize_node(state)
        response = self._with_streaming_trace(state["response"], started_at, state.get("time_to_first_token_ms"))
        streamed_answer = state.get("streamed_answer", "")
        if not streamed_answer:
            yield from self._stream_completed_response(response, started_at)
            return
        if response.answer.startswith(streamed_answer):
            suffix = response.answer[len(streamed_answer) :]
            if suffix:
                yield {"event": "delta", "data": {"text": suffix}}
        elif response.answer != streamed_answer:
            yield {"event": "replace", "data": {"answer": response.answer}}
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _stream_llm_answer(self, state: AgentState, *, medical: bool, started_at: float) -> Iterator[dict[str, Any]]:
        chunks = state.pop("chunks", [])
        history = state.pop("history", [])
        pieces: list[str] = []
        first_token_ms: float | None = None
        yield {"event": "status", "data": {"stage": "generating"}}
        for delta in self.llm.answer_stream(
            question=state["request"].message,
            chunks=chunks,
            history=history,
            medical=medical,
            response_language=state["request"].response_language,
            allow_medical_model_context=bool(state.get("medical_model_context_used")),
        ):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started_at) * 1000, 2)
                yield {"event": "metric", "data": {"time_to_first_token_ms": first_token_ms}}
            pieces.append(delta)
            yield {"event": "delta", "data": {"text": delta}}
        state["answer"] = "".join(pieces)
        state["streamed_answer"] = state["answer"]
        state["time_to_first_token_ms"] = first_token_ms

    def _stream_completed_response(self, response: ChatResponse, started_at: float) -> Iterator[dict[str, Any]]:
        response = self._with_streaming_trace(response, started_at, 0.0)
        for index in range(0, len(response.answer), 64):
            yield {"event": "delta", "data": {"text": response.answer[index : index + 64]}}
        yield {"event": "final", "data": response.model_dump(mode="json")}

    def _with_streaming_trace(
        self,
        response: ChatResponse,
        started_at: float,
        time_to_first_token_ms: float | None,
    ) -> ChatResponse:
        trace = dict(response.trace or {})
        trace["streaming"] = True
        trace["time_to_first_token_ms"] = time_to_first_token_ms
        trace["stream_total_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return response.model_copy(update={"trace": trace})

    def _preflight_response(self, request: ChatRequest) -> ChatResponse | None:
        """
        Pre-flight safety checks before main agent processing.

        Checks (in order):
            1. Emergency keywords (911, heart attack, etc.)
                â†’ Route to emergency_redirect
            2. Prompt injection patterns (ignore rules, system prompt, etc.)
                â†’ Route to prompt_injection_blocked
            3. Transcript confidence (if voice input)
                â†’ Route to clarify if too low

        If any check fails, returns early with safety response.
        If all checks pass, returns None to proceed to full agent.

        Args:
            request: ChatRequest with message and context

        Returns:
            ChatResponse with early response, or None to continue
        """
        route = ""
        answer = ""
        notes: list[str] = []
        guardrail = ""
        conversation_context = self.memory.context_get(request.session_id, request.workspace_id)
        if self._should_route_emergency(request.message, conversation_context):
            route = "emergency_redirect"
            guardrail = "emergency_redirect"
            answer = self._emergency_redirect_answer()
        elif self.safety.detects_prompt_injection(request.message):
            route = "prompt_injection_blocked"
            guardrail = "prompt_injection"
            answer = (
                "I can't follow instructions that try to override CareMind's safety or evidence rules. "
                "Ask your medical-document question directly, and I will answer from the available evidence with citations."
            )
            notes.append("Prompt-injection language was blocked before retrieval or generation.")
        elif (
            request.transcript is not None
            and request.transcript.confidence is not None
            and request.transcript.confidence < self.llm.settings.transcript_min_confidence
        ):
            route = "clarify"
            guardrail = "low_confidence_transcript"
            answer = (
                "I may have misheard that voice input. Please confirm or rephrase the question before I use it "
                "for document retrieval."
            )
            notes.append(
                f"Transcript confidence {request.transcript.confidence:.2f} is below "
                f"{self.llm.settings.transcript_min_confidence:.2f}."
            )
        if not route:
            return None

        answer, notes = self.safety.finalize(answer, False, notes, require_citations=False)
        user_message_id = self.memory.append(request.session_id, request.workspace_id, "user", request.message)
        if request.transcript is not None:
            self.tools.store.save_voice_transcript(
                message_id=user_message_id,
                session_id=request.session_id,
                workspace_id=request.workspace_id,
                transcript=request.transcript,
            )
        self.memory.append(request.session_id, request.workspace_id, "assistant", answer)
        metrics.record_guardrail(guardrail, "blocked")
        metrics.record_agent(route, [], 0, cache_hit=False)
        return ChatResponse(
            session_id=request.session_id,
            route=route,
            answer=answer,
            citations=[],
            safety_notes=notes,
            tool_calls=[],
            trace={
                "session_id": request.session_id,
                "workspace_id": request.workspace_id,
                "route": route,
                "cache_hit": False,
                "guardrail": guardrail,
                "generation_bypassed": True,
                "total_latency_ms": 0,
                "steps": [
                    {
                        "node": "input_guardrail",
                        "route": route,
                        "action": "blocked",
                        "generation_bypassed": True,
                    }
                ],
            },
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("observe", self._observe_node)
        graph.add_node("conversation_resolver", self._conversation_resolver_node)
        graph.add_node("build_plan", self._plan_node)
        graph.add_node("check_cache", self._check_cache_node)
        graph.add_node("cached_response", self._cached_response_node)
        graph.add_node("direct", self._direct_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("emergency_redirect", self._emergency_redirect_node)
        graph.add_node("compare", self._compare_node)
        graph.add_node("imaging", self._imaging_node)
        graph.add_node("medical_education", self._medical_education_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("repair", self._repair_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("observe")
        graph.add_edge("observe", "conversation_resolver")
        graph.add_edge("conversation_resolver", "build_plan")
        graph.add_edge("build_plan", "check_cache")
        graph.add_conditional_edges(
            "check_cache",
            self._next_after_cache,
            {
                "cached_response": "cached_response",
                "direct": "direct",
                "clarify": "clarify",
                "emergency_redirect": "emergency_redirect",
                "compare": "compare",
                "imaging": "imaging",
                "medical_education": "medical_education",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("cached_response", END)
        for node in ["direct", "clarify", "emergency_redirect", "compare", "imaging", "medical_education", "retrieve"]:
            graph.add_edge(node, "validate")
        graph.add_conditional_edges(
            "validate",
            self._next_after_validation,
            {
                "repair": "repair",
                "finalize": "finalize",
            },
        )
        graph.add_edge("repair", "check_cache")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _observe_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        original_message = request.message
        started = time.perf_counter()
        history = self.memory.load(request.session_id, request.workspace_id, limit=8)
        conversation_context = self.memory.context_get(request.session_id, request.workspace_id)
        response_language = self._response_language(request, conversation_context)
        documents = self.tools.store.list_documents(request.workspace_id)
        list_images = getattr(self.tools.store, "list_images", None)
        images = list_images(request.workspace_id) if callable(list_images) else []
        current_attachments = self._current_message_attachments(request)
        context_resolution = self._resolve_contextual_query(
            original_message,
            conversation_context,
            history,
            documents=documents,
            images=images,
            current_attachments=current_attachments,
        )
        effective_message = context_resolution["rewritten_query"]
        effective_request = request.model_copy(update={"message": effective_message, "response_language": response_language})
        transcript_metadata = request.transcript.model_dump(mode="json") if request.transcript is not None else {}
        observations = {
            "workspace_id": request.workspace_id,
            "session_id": request.session_id,
            "modality": request.modality,
            "message": effective_message,
            "original_message": original_message,
            "context_resolution": context_resolution,
            "intent_rewrite": context_resolution.get("intent_rewrite", {}),
            "conversation_context": conversation_context,
            "transcript": transcript_metadata,
            "response_language": response_language,
            "history_count": len(history),
            "document_count": len(documents),
            "document_names": [document.filename for document in documents[:8]],
            "image_count": len(images),
            "image_names": [image.filename for image in images[:8]],
            "attachment_count": len(current_attachments),
            "attachment_ids": [attachment.id for attachment in current_attachments],
            "attachment_types": [attachment.attachment_type for attachment in current_attachments],
            "current_message_attachments": [
                self._attachment_trace(attachment) for attachment in current_attachments
            ],
            "available_tools": [
                "document_search",
                "medical_education_search",
                "medlineplus_health_topic_search",
                "compare_reports",
                "pubmed_literature_search",
            ],
            "external_search_available": self.tools.external_literature_available(),
            "medlineplus_available": self.tools.medlineplus_available(),
        }
        return {
            **state,
            "request": effective_request,
            "original_message": original_message,
            "context_resolution": context_resolution,
            "session_state": context_resolution.get("session_state", {}),
            "conversation_entities": context_resolution.get("conversation_entities", {}),
            "observations": observations,
            "repair_count": state.get("repair_count", 0),
            "trace_steps": [
                {
                    "node": "observe",
                    "agent": "ConversationResolverAgent",
                    "document_count": observations["document_count"],
                    "image_count": observations["image_count"],
                    "attachment_count": observations["attachment_count"],
                    "attachment_ids": observations["attachment_ids"],
                    "attachment_types": observations["attachment_types"],
                    "current_message_attachment": bool(observations["attachment_count"]),
                    "history_count": observations["history_count"],
                    "available_tools": observations["available_tools"],
                    "original_message": original_message,
                    "rewritten_query": effective_message,
                    "transcript": transcript_metadata,
                    "detected_language": transcript_metadata.get("detected_language") or transcript_metadata.get("language"),
                    "language_confidence": transcript_metadata.get("language_confidence"),
                    "response_language": response_language,
                    "resolved_query": context_resolution.get("resolved_query", effective_message),
                    "current_intent": context_resolution.get("intent_rewrite", {}).get("current_intent", ""),
                    "route_hint": context_resolution.get("intent_rewrite", {}).get("route_hint", ""),
                    "intent_confidence": context_resolution.get("intent_rewrite", {}).get("confidence", 0.0),
                    "is_followup": context_resolution.get("is_followup", False),
                    "resolution_reason": context_resolution.get("reason"),
                    "active_document_ids": context_resolution.get("active_document_ids", []),
                    "active_attachment_ids": context_resolution.get("active_attachment_ids", []),
                    "selected_attachment": context_resolution.get("selected_attachment"),
                    "selection_reason": context_resolution.get("reason"),
                    "context_used": context_resolution["used_context"],
                    "active_patient": context_resolution.get("active_patient"),
                    "active_document_name": context_resolution.get("active_document_name"),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            ],
        }

    def _conversation_resolver_node(self, state: AgentState) -> AgentState:
        if state.get("context_resolution"):
            return {
                **state,
                "trace_steps": [
                    *state.get("trace_steps", []),
                    {
                        "node": "conversation_resolver",
                        "agent": "ConversationResolverAgent",
                        "resolved_query": state["request"].message,
                        "current_intent": state["context_resolution"].get("intent_rewrite", {}).get("current_intent", ""),
                        "route_hint": state["context_resolution"].get("intent_rewrite", {}).get("route_hint", ""),
                        "intent_confidence": state["context_resolution"].get("intent_rewrite", {}).get("confidence", 0.0),
                        "used_context": state["context_resolution"].get("used_context", False),
                        "reason": state["context_resolution"].get("reason"),
                        "entity_counts": {
                            key: len(value) if isinstance(value, list) else 0
                            for key, value in state.get("conversation_entities", {}).items()
                        },
                        "active_document_ids": state["context_resolution"].get("active_document_ids", []),
                        "active_attachment_ids": state["context_resolution"].get("active_attachment_ids", []),
                        "selected_attachment": state["context_resolution"].get("selected_attachment"),
                        "current_message_attachment": state["context_resolution"].get("current_message_attachment", False),
                        "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                    },
                ],
            }
        return self._observe_node(state)

    def _plan_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        route = self._planned_route(request.message, state.get("observations", {}))
        plan = self._build_plan(request.message, route, state.get("observations", {}))
        return {
            **state,
            "plan": plan,
            "route": route,
            "cache_key": self._cache_key(request, route, state.get("context_resolution", {})),
            "cache_namespace": self._cache_namespace(request, route, state.get("context_resolution", {})),
            "cache_hit": False,
            "safety_notes": self.safety.inspect_user_message(request.message),
            "tool_calls": [],
            "citations": [],
            "require_citations": route not in {"direct", "clarify", "emergency_redirect"},
            "retrieved_chunks": [],
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "plan",
                    "agent": "SupervisorAgent",
                    "route": route,
                    "intent": plan["intent"],
                    "current_intent": plan.get("current_intent", ""),
                    "route_source": plan.get("route_source", "supervisor"),
                    "route_confidence": plan.get("route_confidence", 0.0),
                    "goal": plan["goal"],
                    "steps": plan["steps"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                    "notes": "Supervisor routed from the resolved query and active conversation state.",
                }
            ],
        }

    def _cache_debug_info(self, request: ChatRequest, cache_key: str, cache_namespace: str) -> dict[str, Any]:
        cache_debug_info = getattr(self.memory, "cache_debug_info", None)
        if not callable(cache_debug_info):
            return {}
        try:
            payload = cache_debug_info(request.workspace_id, cache_key, cache_namespace)
        except Exception as exc:
            return {"redis_enabled": False, "redis_error": exc.__class__.__name__}
        return payload if isinstance(payload, dict) else {}

    def _check_cache_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        bypass_cache = self._bypass_cache(request)
        cached = (
            None
            if bypass_cache
            else self.memory.cache_get(request.workspace_id, state["cache_key"], current_query=request.message)
        )
        cache_step = {
            "node": "check_cache",
            "cache_key": self._redacted_token(state["cache_key"]),
            "cache_namespace": self._redacted_token(state["cache_namespace"]),
            "cache_debug": self._sanitize_public_trace(self._cache_debug_info(request, state["cache_key"], state["cache_namespace"])),
            "cache_hit": False,
            "semantic_cache_hit": False,
            "cache_bypassed": bypass_cache,
            "debug": request.debug,
            "bypass_cache": request.bypass_cache,
            "response_cache_enabled": self.memory.settings.response_cache_enabled,
            "current_query": "[redacted]",
            "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
        }
        cache_source = "exact"
        if cached is None and state["route"] in {"retrieve", "medical_education", "compare", "imaging"}:
            query_embedding = self.tools.embeddings.embed_query(request.message)
            if bypass_cache:
                semantic_cached = None
                similarity = 0.0
            else:
                semantic_cached, similarity = self.memory.semantic_cache_get(
                    request.workspace_id,
                    state["cache_namespace"],
                    query_embedding,
                    current_query=request.message,
                )
            if semantic_cached is not None:
                cached = semantic_cached
                cache_source = "semantic"
                cache_step["cache_hit"] = True
                cache_step["semantic_cache_hit"] = True
                cache_step["semantic_similarity"] = round(similarity, 4)
            else:
                cache_step["semantic_similarity"] = round(similarity, 4)
                state = {**state, "query_embedding": query_embedding}
        if cached is None:
            metrics.record_cache_lookup("exact", "miss", "bypassed" if bypass_cache else "not_found")
            return {
                **state,
                "trace_steps": [
                    *state.get("trace_steps", []),
                    cache_step,
                ],
            }
        response = self._with_public_route(ChatResponse(**cached), request)
        metrics.record_cache_lookup(cache_source, "hit", "threshold_met" if cache_source == "semantic" else "exact_key")
        metrics.record_agent(response.route, response.tool_calls, 0, cache_hit=True)
        cached_trace = dict(response.trace or {})
        cached_metadata = dict(cached_trace.get("cache_metadata", {}))
        total_latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        cache_step = {
            **cache_step,
            "cache_hit": True,
            "cache_source": cache_source,
            "cached_at": cached_metadata.get("cached_at"),
            "elapsed_ms": total_latency_ms,
        }
        return_step = {
            "node": "return_cached_response",
            "cache_source": cache_source,
            "cache_key": self._redacted_token(state["cache_key"]),
            "cached_at": cached_metadata.get("cached_at"),
            "current_query": "[redacted]",
            "cached_route": response.route,
            "tool_calls_from_cached_response": response.tool_calls,
            "retrieval_bypassed": True,
            "generation_bypassed": True,
            "elapsed_ms": total_latency_ms,
        }
        merged_trace = {
            **self._sanitize_public_trace(cached_trace),
            "cache_hit": True,
            "cache_source": cache_source,
            "cache_key": self._redacted_token(state["cache_key"]),
            "cache_namespace": self._redacted_token(state["cache_namespace"]),
            "cache_debug": self._sanitize_public_trace(self._cache_debug_info(request, state["cache_key"], state["cache_namespace"])),
            "cache_bypassed": False,
            "retrieval_bypassed": True,
            "generation_bypassed": True,
            "cached_response_trace": {"redacted": True},
            "steps": [
                *self._sanitize_public_trace(state.get("trace_steps", [])),
                cache_step,
                return_step,
            ],
            "total_latency_ms": total_latency_ms,
        }
        response = response.model_copy(
            update={
                "trace": merged_trace
            }
        )
        return {**state, "response": response, "cache_hit": True}

    def _cached_response_node(self, state: AgentState) -> AgentState:
        return state

    def _direct_node(self, state: AgentState) -> AgentState:
        return {
            **state,
            "answer": self.direct_agent.answer(state["request"].message),
        }

    def _clarify_node(self, state: AgentState) -> AgentState:
        return {
            **state,
            "answer": self.clarification_agent.answer(),
        }

    def _emergency_redirect_node(self, state: AgentState) -> AgentState:
        notes = list(state.get("safety_notes", []))
        notes.append("Emergency or medication-dosing pattern was routed before retrieval or generation.")
        return {
            **state,
            "answer": self._emergency_redirect_answer(),
            "safety_notes": notes,
            "tool_calls": [],
            "citations": [],
            "retrieved_chunks": [],
            "require_citations": False,
        }

    def _compare_node(self, state: AgentState) -> AgentState:
        updates = self.report_comparison_agent.run(
            state["request"],
            state.get("tool_calls", []),
            state.get("trace_steps", []),
        )
        return {**state, **updates}

    def _imaging_node(self, state: AgentState) -> AgentState:
        updates = self.imaging_agent.run(
            state["request"],
            state.get("tool_calls", []),
            state.get("trace_steps", []),
        )
        return {**state, **updates}

    def _medical_education_node(self, state: AgentState) -> AgentState:
        updates = self.medical_education_agent.run(
            state["request"],
            state.get("tool_calls", []),
            state.get("trace_steps", []),
        )
        return {**state, **updates}

    def _retrieve_node(self, state: AgentState) -> AgentState:
        updates = self.document_rag_agent.run(
            state["request"],
            state.get("tool_calls", []),
            state.get("trace_steps", []),
        )
        return {**state, **updates}

    def _validate_node(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        request = state["request"]
        route = state["route"]
        plan = state.get("plan", {})
        issues = []
        repair_route = ""
        no_relevant_document_evidence = bool(state.get("no_relevant_document_evidence"))

        if not state.get("answer"):
            issues.append("missing_answer")
            repair_route = "clarify"
        if state.get("require_citations", True) and not state.get("citations") and not no_relevant_document_evidence:
            issues.append("missing_citations")
            repair_route = repair_route or "clarify"
        if self._asks_general_not_patient(request.message) and route == "retrieve":
            issues.append("general_question_routed_to_patient_documents")
            repair_route = "medical_education"
        if route == "retrieve" and no_relevant_document_evidence:
            issues.append("no_relevant_document_evidence")
        elif route == "retrieve" and not state.get("retrieved_chunks"):
            issues.append("no_retrieved_document_chunks")
            repair_route = "clarify"
        if plan.get("needs_external_evidence") and not plan.get("external_search_available"):
            issues.append("external_evidence_not_configured")
        if self._has_synthetic_result_mix(state.get("retrieved_chunks", [])):
            issues.append("possible_demo_document_mix")

        validation = {
            "passed": not any(issue in issues for issue in [
                "missing_answer",
                "missing_citations",
                "general_question_routed_to_patient_documents",
                "no_retrieved_document_chunks",
            ]),
            "issues": issues,
            "repair_route": repair_route,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        notes = list(state.get("safety_notes", []))
        if "external_evidence_not_configured" in issues:
            notes.append(
                "External evidence search is not configured; this answer uses only uploaded documents or the internal education corpus."
            )
        if "possible_demo_document_mix" in issues:
            notes.append(
                "Retrieved evidence may include demo or synthetic documents from the same workspace."
            )
        return {
            **state,
            "validation": validation,
            "safety_notes": notes,
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "validate",
                    **validation,
                },
            ],
        }

    def _repair_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        validation = state.get("validation", {})
        repair_route = validation.get("repair_route") or "clarify"
        route = repair_route if repair_route in {"direct", "retrieve", "compare", "imaging", "medical_education", "clarify", "emergency_redirect"} else "clarify"
        plan = {
            **state.get("plan", {}),
            "route": route,
            "repair_reason": validation.get("issues", []),
            "steps": [
                *state.get("plan", {}).get("steps", []),
                f"Repair by re-routing to {route}.",
            ],
        }
        return {
            **state,
            "plan": plan,
            "route": route,
            "cache_key": self._cache_key(request, route, state.get("context_resolution", {})),
            "cache_namespace": self._cache_namespace(request, route, state.get("context_resolution", {})),
            "cache_hit": False,
            "answer": "",
            "citations": [],
            "tool_calls": [],
            "retrieved_chunks": [],
            "require_citations": route not in {"direct", "clarify", "emergency_redirect"},
            "repair_count": state.get("repair_count", 0) + 1,
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "repair",
                    "route": route,
                    "issues": validation.get("issues", []),
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                },
            ],
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        answer = state["answer"]
        if "external_evidence_not_configured" in state.get("validation", {}).get("issues", []):
            answer = (
                f"{answer.strip()}\n\n"
                "Note: I cannot search external real-world cases yet in this environment. "
                "A trusted literature-search tool such as PubMed/NCBI should be enabled for that workflow, "
                "and patient identifiers should not be sent to external services."
            )
        embedding_degraded = bool(getattr(self.tools.embeddings, "degraded", False))
        embedding_degraded_reason = getattr(self.tools.embeddings, "degraded_reason", "")
        safety_notes_before_finalize = list(state.get("safety_notes", []))
        if embedding_degraded:
            degraded_note = (
                "Embedding retrieval is degraded because local hashing embeddings are active; "
                "similarity scores are not reliable until NVIDIA embeddings are configured and documents are re-indexed."
            )
            if degraded_note not in safety_notes_before_finalize:
                safety_notes_before_finalize.append(degraded_note)
        answer, safety_notes = self.safety.finalize(
            answer,
            bool(state.get("citations")),
            safety_notes_before_finalize,
            require_citations=state.get("require_citations", True),
        )
        original_message = state.get("original_message", request.message)
        user_message_id = self.memory.append(request.session_id, request.workspace_id, "user", original_message)
        if request.attachment_ids:
            self.tools.store.bind_attachments_to_message(
                attachment_ids=request.attachment_ids,
                message_id=user_message_id,
                workspace_id=request.workspace_id,
                conversation_id=request.session_id,
            )
        if request.transcript is not None:
            self.tools.store.save_voice_transcript(
                message_id=user_message_id,
                session_id=request.session_id,
                workspace_id=request.workspace_id,
                transcript=request.transcript,
            )
        self.memory.append(request.session_id, request.workspace_id, "assistant", answer)

        total_latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        updated_context = self._updated_conversation_context(state, answer)
        if updated_context:
            self.memory.context_set(request.session_id, request.workspace_id, updated_context)
        public_route = self._public_route(state["route"], request)
        raw_trace = {
            "session_id": request.session_id,
            "workspace_id": request.workspace_id,
            "route": public_route,
            "internal_route": state["route"],
            "plan": state.get("plan", {}),
            "validation": state.get("validation", {}),
            "cache_hit": False,
            "cache_key": state.get("cache_key", ""),
            "cache_namespace": state.get("cache_namespace", ""),
            "cache_debug": self._cache_debug_info(
                request,
                state.get("cache_key", ""),
                state.get("cache_namespace", ""),
            ),
            "cache_bypassed": self._bypass_cache(request),
            "vector_backend": self.tools.embeddings.settings.vector_backend,
            "embedding_model": self._embedding_model_label(),
            "embedding_provider": getattr(self.tools.embeddings, "last_provider", ""),
            "embedding_degraded": embedding_degraded,
            "embedding_degraded_reason": embedding_degraded_reason,
            "generation_model": self._generation_model_label(),
            "langsmith_enabled": self.llm.settings.langsmith_enabled,
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "original_message": original_message,
            "rewritten_query": request.message,
            "resolved_query": request.message,
            "context_resolution": state.get("context_resolution", {}),
            "session_state": state.get("session_state", {}),
            "transcript": request.transcript.model_dump(mode="json") if request.transcript is not None else {},
            "response_language": request.response_language or "",
            "conversation_entities": state.get("conversation_entities", {}),
            "conversation_context": updated_context,
            "attachment_count": len(request.attachment_ids),
            "attachment_ids": request.attachment_ids,
            "attachment_types": state.get("context_resolution", {}).get("current_attachment_types", []),
            "selected_attachment": state.get("context_resolution", {}).get("selected_attachment"),
            "selection_reason": state.get("context_resolution", {}).get("reason"),
            "current_message_attachment": bool(request.attachment_ids),
            "reused_attachment": state.get("context_resolution", {}).get("reason") == "contextual_attachment_followup",
            "processing_route": state.get("plan", {}).get("intent", ""),
            "vision_attempted": any(step.get("vision_attempted") for step in state.get("trace_steps", [])),
            "vision_provider": self._vision_provider_label(state),
            "vision_model": self._vision_model_label(state),
            "ocr_fallback": any(step.get("ocr_fallback") for step in state.get("trace_steps", [])),
            "fallback_reason": self._fallback_reason(state),
            "structured_evidence_reused": any(step.get("structured_evidence_reused") for step in state.get("trace_steps", [])),
            "steps": [
                *state.get("trace_steps", []),
                {
                    "node": "finalize",
                    "citation_count": len(state.get("citations", [])),
                    "safety_note_count": len(safety_notes),
                    "elapsed_ms": total_latency_ms,
                },
            ],
            "total_latency_ms": total_latency_ms,
        }
        response = ChatResponse(
            session_id=request.session_id,
            route=public_route,
            answer=answer,
            citations=state.get("citations", []),
            safety_notes=safety_notes,
            tool_calls=state.get("tool_calls", []),
            trace=self._sanitize_public_trace(raw_trace),
        )
        if (
            state["route"] in {"retrieve", "medical_education", "compare", "imaging"}
            and state.get("validation", {}).get("passed", True)
            and not self._bypass_cache(request)
        ):
            response = response.model_copy(
                update={
                    "trace": {
                        **response.trace,
                        "cache_metadata": self._cache_metadata(
                            request,
                            public_route,
                            state["cache_key"],
                            state["cache_namespace"],
                        ),
                    }
                }
            )
            payload = response.model_dump(mode="json")
            self.memory.cache_set(request.workspace_id, state["cache_key"], payload)
            query_embedding = state.get("query_embedding") or self.tools.embeddings.embed_query(request.message)
            self.memory.semantic_cache_set(
                request.workspace_id,
                state["cache_namespace"],
                state["cache_key"],
                query_embedding,
                payload,
            )
        metrics.record_agent(
            response.route,
            response.tool_calls,
            (time.perf_counter() - state["started_at"]) * 1000,
            cache_hit=False,
        )
        return {**state, "answer": answer, "safety_notes": safety_notes, "response": response}

    def _with_public_route(self, response: ChatResponse, request: ChatRequest) -> ChatResponse:
        public_route = self._public_route(response.route, request)
        if public_route == response.route and response.trace.get("internal_route"):
            return response
        trace = dict(response.trace or {})
        trace.setdefault("internal_route", response.route)
        trace["route"] = public_route
        return response.model_copy(update={"route": public_route, "trace": trace})

    def _public_route(self, route: str, request: ChatRequest) -> str:
        if route == "imaging":
            return "imaging_qa"
        if route == "retrieve":
            if request.modality == "image" or self._looks_like_image_question(request.message.lower()):
                return "imaging_qa"
            return "clinical_document_qa"
        if route == "compare":
            return "report_comparison"
        if route == "medical_education":
            if self._looks_like_nursing_question(request.message.lower()):
                return "nursing_care_qa"
            return "medical_knowledge_qa"
        return route

    def _response_language(self, request: ChatRequest, conversation_context: dict[str, Any]) -> str:
        override = (
            conversation_context.get("response_language_override")
            or conversation_context.get("selected_response_language")
            or conversation_context.get("response_language")
        )
        if override:
            return str(override)
        if request.transcript is None:
            return request.response_language or ""
        return (
            request.response_language
            or request.transcript.detected_language
            or request.transcript.language
            or ""
        )

    def _next_after_cache(self, state: AgentState) -> str:
        if state.get("cache_hit"):
            return "cached_response"
        return state["route"]

    def _next_after_validation(self, state: AgentState) -> str:
        validation = state.get("validation", {})
        if not validation.get("passed", True) and validation.get("repair_route") and state.get("repair_count", 0) < 1:
            return "repair"
        return "finalize"

    def _bypass_cache(self, request: ChatRequest) -> bool:
        return request.debug or request.bypass_cache or not self.memory.settings.response_cache_enabled

    def _cache_key(
        self,
        request: ChatRequest,
        route: str,
        context_resolution: dict[str, Any] | None = None,
    ) -> str:
        raw = "|".join(
            str(part) for part in self._cache_key_components(request, route, include_query=True, context_resolution=context_resolution)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_namespace(
        self,
        request: ChatRequest,
        route: str,
        context_resolution: dict[str, Any] | None = None,
    ) -> str:
        raw = "|".join(
            str(part) for part in self._cache_key_components(request, route, include_query=False, context_resolution=context_resolution)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_key_components(
        self,
        request: ChatRequest,
        route: str,
        include_query: bool,
        context_resolution: dict[str, Any] | None = None,
    ) -> list[Any]:
        workspace_revision = self.tools.store.workspace_revision(request.workspace_id)
        context_resolution = context_resolution or {}
        entity_fingerprint = self._entity_fingerprint(context_resolution.get("conversation_entities", {}))
        settings = self.tools.embeddings.settings
        components: list[Any] = [
            "v7",
            route,
            request.workspace_id,
            workspace_revision,
            request.modality,
            request.response_language or "",
            request.top_k,
            ",".join(request.attachment_ids),
            ",".join(context_resolution.get("active_document_ids", [])),
            ",".join(context_resolution.get("active_attachment_ids", [])),
            context_resolution.get("active_patient", ""),
            context_resolution.get("active_document_name", ""),
            entity_fingerprint,
            settings.vector_backend,
            settings.embedding_provider,
            self._embedding_model_label(),
            getattr(settings, "embedding_dimension", ""),
            getattr(settings, "embedding_index_version", ""),
            self._generation_model_label(),
            settings.min_retrieval_similarity,
            getattr(settings, "medlineplus_health_topics_enabled", ""),
            getattr(settings, "medlineplus_max_results", ""),
            getattr(settings, "medical_education_model_context_enabled", ""),
            getattr(settings, "medical_education_model_context_min_chars", ""),
        ]
        if include_query:
            components.append(self._rewrite_query(request.message).lower())
        return components

    def _entity_fingerprint(self, entities: dict[str, Any]) -> str:
        if not isinstance(entities, dict):
            return ""
        parts: list[str] = []
        for kind in sorted(entities):
            values = entities.get(kind)
            if not isinstance(values, list):
                continue
            names = sorted(self._entity_name(item) for item in values[:8] if item)
            if names:
                parts.append(f"{kind}:{','.join(names)}")
        return ";".join(parts)

    def _entity_name(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name", "")).lower()
        return str(item).lower()

    def _cache_metadata(
        self,
        request: ChatRequest,
        route: str,
        cache_key: str,
        cache_namespace: str,
    ) -> dict[str, Any]:
        return {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": request.workspace_id,
            "session_id": request.session_id,
            "route": route,
            "response_language": request.response_language or "",
            "cache_key": self._redacted_token(cache_key),
            "cache_namespace": self._redacted_token(cache_namespace),
            "workspace_revision": self.tools.store.workspace_revision(request.workspace_id),
            "cache_reason": "resolved_query_workspace_revision_entity_context",
            "cache_debug": self._sanitize_public_trace(self._cache_debug_info(request, cache_key, cache_namespace)),
            "retrieval_settings": {
                "top_k": request.top_k,
                "min_retrieval_similarity": self.tools.embeddings.settings.min_retrieval_similarity,
                "vector_backend": self.tools.embeddings.settings.vector_backend,
                "embedding_model": self._embedding_model_label(),
                "embedding_provider": self.tools.embeddings.settings.embedding_provider,
                "embedding_dimension": self.tools.embeddings.settings.embedding_dimension,
                "embedding_index_version": self.tools.embeddings.settings.embedding_index_version,
                "generation_model": self._generation_model_label(),
            },
        }

    def _sanitize_public_trace(self, value: Any) -> Any:
        sensitive_keys = {
            "active_patient",
            "active_document_name",
            "active_document_id",
            "active_document_ids",
            "active_attachment_ids",
            "active_report",
            "attachment_ids",
            "available_report",
            "cache_key",
            "cache_namespace",
            "chunk_id",
            "conversation_context",
            "conversation_entities",
            "current_query",
            "document_name",
            "document_names",
            "document_id",
            "exact_redis_key",
            "exact_cache_pattern",
            "message",
            "name",
            "normalized_query",
            "original_message",
            "preview",
            "query",
            "resolved_query",
            "rewritten_query",
            "selected_attachment",
            "semantic_redis_key",
            "semantic_redis_prefix",
            "session_state",
            "standalone_query",
            "text",
            "retrieval_query",
        }
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if key in sensitive_keys:
                    if key in {"cache_key", "cache_namespace", "exact_redis_key", "semantic_redis_key", "semantic_redis_prefix", "exact_cache_pattern"}:
                        result[key] = self._redacted_token(str(item))
                    elif key in {"document_name", "document_names"}:
                        result[key] = self._document_label(item)
                    elif key in {"active_document_id", "active_document_ids", "active_attachment_ids", "attachment_ids", "chunk_id", "document_id", "selected_attachment"}:
                        result[key] = "[redacted]"
                    elif key in {"conversation_context", "conversation_entities", "session_state"}:
                        result[key] = {"redacted": True}
                    else:
                        result[key] = "[redacted]"
                else:
                    result[key] = self._sanitize_public_trace(item)
            return result
        if isinstance(value, list):
            return [self._sanitize_public_trace(item) for item in value]
        return value

    def _redacted_token(self, value: str) -> str:
        return f"{value[:8]}..." if value else ""

    def _document_label(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._document_label(item) for item in value]
        return "Uploaded document" if value else value

    def _embedding_model_label(self) -> str:
        provider = self.tools.embeddings.settings.embedding_provider.lower()
        if provider in {"qwen", "sentence-transformers", "sentence_transformers"}:
            return self.tools.embeddings.settings.embedding_model
        if provider == "ollama" and self.tools.embeddings.settings.ollama_embedding_model:
            return self.tools.embeddings.settings.ollama_embedding_model
        if self.tools.embeddings.settings.nvidia_api_key:
            return self.tools.embeddings.settings.nvidia_embedding_model
        return f"local-hashing-{self.tools.embeddings.dimension}d"

    def _generation_model_label(self) -> str:
        if self.llm.settings.nvidia_api_key:
            return self.llm.settings.nvidia_chat_model
        return "local-grounded-fallback"

    def _run_config(self, request: ChatRequest) -> dict:
        return {
            "run_name": "CareMindAgent",
            "tags": [
                "caremind",
                f"env:{self.llm.settings.environment}",
                f"workspace:{request.workspace_id}",
            ],
            "metadata": {
                "app": self.llm.settings.app_name,
                "app_version": self.llm.settings.app_version,
                "environment": self.llm.settings.environment,
                "workspace_id": request.workspace_id,
                "session_id": request.session_id,
                "top_k": request.top_k,
                "langsmith_enabled": self.llm.settings.langsmith_enabled,
            },
            "configurable": {
                "thread_id": f"{request.workspace_id}:{request.session_id}",
            },
        }

    def _planned_route(self, message: str, observations: dict[str, Any] | None = None) -> Route:
        lowered = message.lower()
        observations = observations or {}
        context_resolution = observations.get("context_resolution", {})
        intent_rewrite = context_resolution.get("intent_rewrite", {})
        route_hint = intent_rewrite.get("route_hint", "")
        route_confidence = float(intent_rewrite.get("confidence") or 0.0)
        conversation_context = observations.get("conversation_context", {})
        if self._should_route_emergency(message, conversation_context):
            return "emergency_redirect"
        if context_resolution.get("current_message_attachment") and route_hint in {"retrieve", "compare", "imaging"} and route_confidence >= 0.7:
            return route_hint
        if self._asks_general_not_patient(lowered):
            return "medical_education"
        if route_hint in {"direct", "retrieve", "compare", "medical_education", "imaging"} and route_confidence >= 0.7:
            return route_hint
        if observations.get("modality") == "image" or self._looks_like_image_question(lowered):
            return "imaging"
        if "compare" in lowered:
            return "compare"
        if context_resolution.get("used_context") and not self._explicitly_changes_to_general_topic(lowered):
            if not self._asks_for_external_cases(lowered) and not self._looks_direct_product_question(lowered):
                return "retrieve"
        if self._looks_like_patient_reference(lowered) and observations.get("document_count", 0) > 0:
            return "retrieve"
        if self._asks_for_external_cases(lowered):
            return "medical_education"
        return self.supervisor_agent.route(message)

    def _build_plan(self, message: str, route: Route, observations: dict[str, Any]) -> dict[str, Any]:
        lowered = message.lower()
        intent_rewrite = observations.get("intent_rewrite", {})
        needs_external = self._asks_for_external_cases(lowered)
        intent = {
            "direct": "product_or_help",
            "retrieve": "patient_document_question",
            "compare": "compare_documents",
            "imaging": "image_grounded_question",
            "medical_education": "general_medical_education",
            "clarify": "needs_clarification",
            "emergency_redirect": "emergency_redirect",
        }[route]
        if needs_external:
            intent = "external_case_or_literature_request"
        steps_by_route = {
            "direct": ["Answer from CareMind capability instructions."],
            "retrieve": ["Search uploaded workspace documents.", "Generate a cited answer from retrieved passages."],
            "compare": ["Load selected reports.", "Compare indexed passages and cite differences."],
            "imaging": ["Search uploaded image metadata and paired reports.", "Answer only from image-linked grounding evidence."],
            "medical_education": ["Search internal medical education corpus.", "Generate a general educational answer."],
            "clarify": ["Ask a focused clarification before using tools."],
            "emergency_redirect": ["Bypass retrieval and generation.", "Return urgent-care or clinician/pharmacist redirect guidance."],
        }
        if needs_external:
            steps = [
                "Extract the clinical pattern without patient identifiers.",
                "Search trusted external literature through PubMed/NCBI.",
                "Generate a cited educational answer without sending patient identifiers.",
            ]
        else:
            steps = steps_by_route[route]
        return {
            "intent": intent,
            "current_intent": intent_rewrite.get("current_intent") or self._rewrite_query(message),
            "intent_rewrite": intent_rewrite,
            "route_source": "intent_rewrite" if intent_rewrite.get("route_hint") == route else "supervisor",
            "route_confidence": intent_rewrite.get("confidence", 0.0),
            "goal": self._goal_for_route(route, needs_external),
            "route": route,
            "needs_patient_docs": route in {"retrieve", "compare"},
            "needs_image_evidence": route == "imaging",
            "needs_general_education": route == "medical_education",
            "needs_external_evidence": needs_external,
            "external_search_available": bool(observations.get("external_search_available")),
            "rewritten_query": self._rewrite_query(message),
            "steps": steps,
        }

    def _goal_for_route(self, route: Route, needs_external: bool) -> str:
        if needs_external:
            return "Find comparable clinical context while avoiding patient identifiers."
        return {
            "direct": "Explain CareMind or help the user operate it.",
            "retrieve": "Answer using uploaded patient/document evidence.",
            "compare": "Compare uploaded reports using indexed evidence.",
            "imaging": "Answer image questions using uploaded image metadata and paired report evidence.",
            "medical_education": "Explain the medical topic generally.",
            "clarify": "Ask for the missing evidence source or intent.",
            "emergency_redirect": "Bypass generation and redirect urgent or dosing questions to appropriate care.",
        }[route]

    def _emergency_redirect_answer(self) -> str:
        return (
            "I can't help with emergency symptoms, crisis situations, or medication dosing decisions. "
            "If this may be urgent, contact local emergency services now. For medication or dosing questions, "
            "contact a licensed clinician or pharmacist."
        )

    def _should_route_emergency(self, message: str, conversation_context: dict | None = None) -> bool:
        if not self.safety.should_emergency_redirect(message):
            return False
        return not self._should_treat_as_document_followup_for_safety(message, conversation_context or {})

    def _rewrite_query(self, message: str) -> str:
        return " ".join(message.strip().split())

    def _resolve_contextual_query(
        self,
        message: str,
        context: dict,
        history: list[Any],
        documents: list[Any] | None = None,
        images: list[Any] | None = None,
        current_attachments: list[Any] | None = None,
    ) -> dict[str, Any]:
        resolver = getattr(self, "conversation_resolver_agent", None) or ConversationResolverAgent()
        return resolver.resolve(
            message=message,
            context=context,
            history=history,
            documents=documents or [],
            images=images or [],
            current_attachments=current_attachments or [],
        )

    def _current_message_attachments(self, request: ChatRequest) -> list[Any]:
        if not request.attachment_ids:
            return []
        require = getattr(self.tools.store, "require_message_attachments", None)
        if not callable(require):
            return []
        return require(
            attachment_ids=request.attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
        )

    def _attachment_trace(self, attachment: Any) -> dict[str, Any]:
        return {
            "id": getattr(attachment, "id", ""),
            "attachment_type": getattr(attachment, "attachment_type", ""),
            "filename": getattr(attachment, "filename", ""),
            "mime_type": getattr(attachment, "mime_type", ""),
            "image_asset_id": getattr(attachment, "image_asset_id", None),
            "document_id": getattr(attachment, "document_id", None),
            "processing_status": getattr(attachment, "processing_status", ""),
            "persistence_mode": getattr(attachment, "persistence_mode", ""),
        }

    def _vision_provider_label(self, state: AgentState) -> str:
        for step in state.get("trace_steps", []):
            if step.get("vision_provider"):
                return str(step["vision_provider"])
        if state.get("route") == "imaging":
            return "medgemma" if self.llm.settings.radiology_vision_base_url else ""
        if any(step.get("tool") == "message_attachment_lookup" for step in state.get("trace_steps", [])):
            return "groq-or-configured-openai-compatible-vlm" if self.llm.settings.document_vlm_base_url else ""
        return ""

    def _vision_model_label(self, state: AgentState) -> str:
        for step in state.get("trace_steps", []):
            if step.get("vision_model"):
                return str(step["vision_model"])
        if state.get("route") == "imaging":
            return self.llm.settings.radiology_vision_model if self.llm.settings.radiology_vision_base_url else ""
        if any(step.get("tool") == "message_attachment_lookup" for step in state.get("trace_steps", [])):
            return self.llm.settings.document_vlm_model if self.llm.settings.document_vlm_base_url else ""
        return ""

    def _fallback_reason(self, state: AgentState) -> str:
        for step in state.get("trace_steps", []):
            if step.get("fallback_reason"):
                return str(step["fallback_reason"])
        return ""

    def _updated_conversation_context(self, state: AgentState, answer: str) -> dict:
        request = state["request"]
        previous = self.memory.context_get(request.session_id, request.workspace_id)
        citations = state.get("citations", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        now = datetime.now(timezone.utc).isoformat()
        current_attachments = self._current_message_attachments(request)
        voice_context = self._voice_context_update(request)
        if state.get("route") == "medical_education" and not current_attachments:
            return {
                **previous,
                **voice_context,
                "conversation_focus": "general_medical_education",
                "current_topic": request.message[:160],
                "active_attachment_id": "",
                "active_attachment_type": "",
                "updated_at": now,
            }
        if current_attachments:
            first_attachment = current_attachments[0]
            document_ids = [
                getattr(item, "document_id", None)
                for item in current_attachments
                if getattr(item, "document_id", None)
            ]
            image_ids = [
                getattr(item, "image_asset_id", None)
                for item in current_attachments
                if getattr(item, "image_asset_id", None)
            ]
            updated = {
                **previous,
                **voice_context,
                "active_attachment_id": first_attachment.id,
                "active_attachment_type": first_attachment.attachment_type,
                "active_attachment_name": first_attachment.filename,
                "last_attachment_ids": [item.id for item in current_attachments],
                "conversation_focus": "attachment",
                "current_topic": first_attachment.filename,
                "last_structured_evidence_id": first_attachment.document_id or "",
                "last_evidence_need": state.get("evidence_need", ""),
                "last_retrieval_result": retrieved_chunks[:5],
                "last_retrieved_chunks": retrieved_chunks[:5],
                "last_answer_preview": answer[:500],
                "updated_at": now,
            }
            if document_ids:
                updated["active_document_id"] = document_ids[0]
                updated["active_document_ids"] = document_ids
                updated["active_document_name"] = first_attachment.filename
                updated["active_report"] = first_attachment.filename
            if image_ids:
                updated["active_image_id"] = image_ids[0]
                updated["active_image_ids"] = image_ids
                updated["active_image_name"] = first_attachment.filename
            return updated
        if state.get("route") != "retrieve" or not citations:
            return {**previous, **voice_context} if voice_context else previous

        first_citation = citations[0]
        evidence_texts = [citation.quote for citation in citations]
        evidence_texts.extend(str(chunk.get("preview", "")) for chunk in retrieved_chunks[:3])
        context_resolution = state.get("context_resolution", {})
        active_patient = (
            self._extract_patient_name(" ".join([request.message, *evidence_texts]))
            or previous.get("active_patient")
            or ""
        )
        last_citations = [citation.model_dump(mode="json") for citation in citations[:5]]
        document_memory = dict(previous.get("document_memory", {}))
        document_memory[first_citation.document_id] = {
            "document_id": first_citation.document_id,
            "document_name": first_citation.document_name,
            "summary": previous.get("document_memory", {})
            .get(first_citation.document_id, {})
            .get("summary", ""),
            "last_chunks": retrieved_chunks[:5],
            "last_citations": last_citations,
            "last_answer_preview": answer[:500],
            "updated_at": now,
        }
        conversation_entities = self._merge_entity_context(
            previous.get("conversation_entities", {}),
            state.get("conversation_entities", {}),
            active_patient,
            first_citation,
        )
        updated = {
            **previous,
            **voice_context,
            "active_patient": active_patient,
            "active_document_id": first_citation.document_id,
            "active_document_name": first_citation.document_name,
            "active_report": first_citation.document_name,
            "active_document_ids": context_resolution.get("active_document_ids") or [first_citation.document_id],
            "current_patient": active_patient,
            "current_document": first_citation.document_name,
            "current_report": first_citation.document_name,
            "last_route": state.get("route", ""),
            "last_tool": "document_search" if "document_search" in state.get("tool_calls", []) else "",
            "last_evidence_need": state.get("evidence_need", ""),
            "last_cited_document": {
                "document_id": first_citation.document_id,
                "document_name": first_citation.document_name,
                "chunk_id": first_citation.chunk_id,
                "page": first_citation.page,
            },
            "last_retrieval_result": retrieved_chunks[:5],
            "last_retrieved_chunks": retrieved_chunks[:5],
            "last_citations": last_citations,
            "last_answer_preview": answer[:500],
            "last_summary": answer[:500],
            "conversation_entities": conversation_entities,
            "document_memory": document_memory,
            "updated_at": now,
        }
        return updated

    def _voice_context_update(self, request: ChatRequest) -> dict[str, Any]:
        if request.transcript is None:
            return {}
        return {
            "last_voice_transcript_language": request.transcript.detected_language or request.transcript.language or "",
            "last_voice_language_confidence": request.transcript.language_confidence,
            "last_voice_transcript_confidence": request.transcript.confidence,
            "last_input_modality": request.transcript.modality or request.transcript.input_modality or request.modality,
            "last_voice_transcript_at": datetime.now(timezone.utc).isoformat(),
        }

    def _merge_entity_context(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        active_patient: str,
        citation: Citation,
    ) -> dict[str, Any]:
        merged: dict[str, list[dict[str, str]]] = {}
        for source in [previous, current]:
            if not isinstance(source, dict):
                continue
            for kind, values in source.items():
                if not isinstance(values, list):
                    continue
                bucket = merged.setdefault(kind, [])
                for value in values:
                    if not isinstance(value, dict):
                        value = {"name": str(value), "id": ""}
                    name = str(value.get("name", ""))
                    if name and not any(item.get("name", "").lower() == name.lower() for item in bucket):
                        bucket.append({"name": name, "id": str(value.get("id", ""))})
        if active_patient:
            patient_bucket = merged.setdefault("patient", [])
            if not any(item.get("name", "").lower() == active_patient.lower() for item in patient_bucket):
                patient_bucket.append({"name": active_patient, "id": citation.document_id})
        report_bucket = merged.setdefault("report", [])
        if not any(item.get("id") == citation.document_id for item in report_bucket):
            report_bucket.append({"name": citation.document_name, "id": citation.document_id})
        return merged

    def _extract_patient_name(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text)
        patterns = [
            r"\b(Mr|Mrs|Ms|Miss)\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5})\b",
            r"\bpatient\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b",
        ]
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                if match.lastindex and match.lastindex >= 2:
                    prefix = match.group(1)
                    name = match.group(2)
                    cleaned = f"{prefix}. {name}".strip()
                else:
                    cleaned = match.group(1).strip()
                cleaned = re.sub(r"\b(ID|Age|Gender|MRD|Reason|Doctor|Centre|Study|Duration)\b.*$", "", cleaned).strip()
                if len(cleaned.split()) >= 2:
                    candidates.append(cleaned)
        if not candidates:
            return ""
        return max(candidates, key=lambda candidate: (len(candidate.split()), len(candidate)))

    def _is_contextual_followup(self, message: str) -> bool:
        lowered = message.lower().strip()
        contextual_terms = [
            " he ",
            " him",
            " his ",
            " she ",
            " her ",
            " the patient",
            "this report",
            "that report",
            "earlier",
            "same patient",
            "those findings",
            "these findings",
            " it ",
        ]
        padded = f" {lowered} "
        if any(term in padded for term in contextual_terms):
            return True
        if lowered in {"why chest pain?", "why chest pain", "chest pain?", "chest pain", "what happened next?", "what happened next"}:
            return True
        if lowered.startswith(("why ", "did he ", "did she ", "was he ", "was she ", "what happened next")):
            return True
        clinical_followup_terms = ["fatigue", "test performed", "reason for test", "chest pain", "symptom", "finding"]
        return len(lowered.split()) <= 6 and any(term in lowered for term in clinical_followup_terms)

    def _explicitly_changes_to_general_topic(self, message: str) -> bool:
        return explicitly_general_topic(message)

    def _looks_direct_product_question(self, message: str) -> bool:
        lowered = message.lower().strip(" ?!.")
        return lowered in {"help", "hello", "hi", "hey"} or any(
            pattern in lowered
            for pattern in [
                "is this caremind",
                "what is caremind",
                "who are you",
                "what can you do",
                "what can u do",
                "how do you work",
            ]
        )

    def _looks_like_patient_reference(self, message: str) -> bool:
        lowered = message.lower()
        if any(term in lowered for term in ["the patient", "this patient", "patient "]):
            return True
        return bool(re.search(r"\b(mr|mrs|ms|miss)\.?\s+[a-z][a-z]+", lowered))

    def _should_treat_as_document_followup_for_safety(self, message: str, context: dict) -> bool:
        if not (context.get("active_patient") or context.get("active_document_name") or context.get("active_report")):
            return False
        lowered = message.lower()
        explicit_document_followup = any(
            term in lowered
            for term in [
                "uploaded report",
                "this report",
                "that report",
                "previously discussed uploaded report",
                "according to the uploaded report",
            ]
        )
        if not explicit_document_followup and not self._is_contextual_followup(message):
            return False
        acute_personal = bool(re.search(r"\b(i|i'm|im|me|my)\b", lowered)) and any(
            term in lowered
            for term in ["now", "severe", "having", "feel", "emergency", "urgent", "today"]
        )
        return not acute_personal

    def _asks_general_not_patient(self, message: str) -> bool:
        return asks_general_medical_not_document(message)

    def _asks_for_external_cases(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            phrase in lowered
            for phrase in [
                "real world case",
                "real-world case",
                "similar case",
                "case report",
                "published case",
                "literature",
                "pubmed",
                "external",
                "like another case",
                "not the same patient",
            ]
        )

    def _looks_like_nursing_question(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            phrase in lowered
            for phrase in [
                "nursing",
                "nurse",
                "care plan",
                "care protocol",
                "patient education",
                "post-op",
                "postoperative",
                "monitoring protocol",
            ]
        )

    def _looks_like_image_question(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            phrase in lowered
            for phrase in [
                "x-ray",
                "xray",
                "chest x",
                "scan",
                "image",
                "radiograph",
                "cxr",
            ]
        )

    def _has_synthetic_result_mix(self, retrieved_chunks: list[dict]) -> bool:
        return any(
            "synthetic" in str(chunk.get("document_name", "")).lower()
            or "baseline lab report" in str(chunk.get("document_name", "")).lower()
            for chunk in retrieved_chunks
        )
