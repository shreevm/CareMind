import hashlib
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agents import (
    ClarificationAgent,
    DirectResponseAgent,
    DocumentRAGAgent,
    MedicalEducationAgent,
    ReportComparisonAgent,
    Route,
    SupervisorAgent,
)
from .llm import LLMClient
from .memory import ConversationMemory
from .metrics import metrics
from .safety import SafetyLayer
from .schemas import ChatRequest, ChatResponse, Citation
from .tools import DocumentTools


class AgentState(TypedDict, total=False):
    request: ChatRequest
    started_at: float
    route: Route
    cache_key: str
    cache_hit: bool
    answer: str
    citations: list[Citation]
    safety_notes: list[str]
    tool_calls: list[str]
    require_citations: bool
    retrieved_chunks: list[dict]
    trace_steps: list[dict]
    response: ChatResponse


class CareMindAgent:
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
        self.supervisor_agent = SupervisorAgent()
        self.direct_agent = DirectResponseAgent()
        self.clarification_agent = ClarificationAgent()
        self.document_rag_agent = DocumentRAGAgent(tools, llm, memory)
        self.medical_education_agent = MedicalEducationAgent(tools, llm, memory)
        self.report_comparison_agent = ReportComparisonAgent(tools)
        self.graph = self._build_graph()

    def answer(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke(
            {"request": request, "started_at": time.perf_counter()},
            config=self._run_config(request),
        )
        return state["response"]

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_query", self._route_node)
        graph.add_node("check_cache", self._check_cache_node)
        graph.add_node("cached_response", self._cached_response_node)
        graph.add_node("direct", self._direct_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("compare", self._compare_node)
        graph.add_node("medical_education", self._medical_education_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("route_query")
        graph.add_edge("route_query", "check_cache")
        graph.add_conditional_edges(
            "check_cache",
            self._next_after_cache,
            {
                "cached_response": "cached_response",
                "direct": "direct",
                "clarify": "clarify",
                "compare": "compare",
                "medical_education": "medical_education",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("cached_response", END)
        for node in ["direct", "clarify", "compare", "medical_education", "retrieve"]:
            graph.add_edge(node, "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _route_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        route = self.supervisor_agent.route(request.message)
        return {
            **state,
            "route": route,
            "cache_key": self._cache_key(request, route),
            "cache_hit": False,
            "safety_notes": self.safety.inspect_user_message(request.message),
            "tool_calls": [],
            "citations": [],
            "require_citations": route not in {"direct", "clarify"},
            "retrieved_chunks": [],
            "trace_steps": [
                {
                    "node": "route_query",
                    "agent": "SupervisorAgent",
                    "route": route,
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                    "notes": "Supervisor selected the specialist agent branch.",
                }
            ],
        }

    def _check_cache_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        cached = self.memory.cache_get(request.workspace_id, state["cache_key"])
        if cached is None:
            return {
                **state,
                "trace_steps": [
                    *state.get("trace_steps", []),
                    {
                        "node": "check_cache",
                        "cache_hit": False,
                        "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                    },
                ],
            }
        response = ChatResponse(**cached)
        metrics.record_agent(response.route, response.tool_calls, 0, cache_hit=True)
        response = response.model_copy(
            update={
                "trace": {
                    **response.trace,
                    "cache_hit": True,
                    "steps": [
                        *state.get("trace_steps", []),
                        {
                            "node": "check_cache",
                            "cache_hit": True,
                            "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                        },
                    ],
                }
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

    def _compare_node(self, state: AgentState) -> AgentState:
        updates = self.report_comparison_agent.run(
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

    def _finalize_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        answer, safety_notes = self.safety.finalize(
            state["answer"],
            bool(state.get("citations")),
            state.get("safety_notes", []),
            require_citations=state.get("require_citations", True),
        )
        self.memory.append(request.session_id, request.workspace_id, "user", request.message)
        self.memory.append(request.session_id, request.workspace_id, "assistant", answer)

        total_latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        response = ChatResponse(
            session_id=request.session_id,
            route=state["route"],
            answer=answer,
            citations=state.get("citations", []),
            safety_notes=safety_notes,
            tool_calls=state.get("tool_calls", []),
            trace={
                "session_id": request.session_id,
                "workspace_id": request.workspace_id,
                "route": state["route"],
                "cache_hit": False,
                "vector_backend": (
                    "pinecone"
                    if self.tools.vectorstore.settings.should_use_pinecone
                    else "local-sqlite"
                ),
                "embedding_model": (
                    self.tools.embeddings.settings.nvidia_embedding_model
                    if self.tools.embeddings.settings.nvidia_api_key
                    else f"local-hashing-{self.tools.embeddings.dimension}d"
                ),
                "generation_model": (
                    self.llm.settings.nvidia_chat_model
                    if self.llm.settings.nvidia_api_key
                    else "local-grounded-fallback"
                ),
                "langsmith_enabled": self.llm.settings.langsmith_enabled,
                "retrieved_chunks": state.get("retrieved_chunks", []),
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
            },
        )
        if state["route"] in {"retrieve", "medical_education", "compare"}:
            self.memory.cache_set(request.workspace_id, state["cache_key"], response.model_dump(mode="json"))
        metrics.record_agent(
            response.route,
            response.tool_calls,
            (time.perf_counter() - state["started_at"]) * 1000,
            cache_hit=False,
        )
        return {**state, "answer": answer, "safety_notes": safety_notes, "response": response}

    def _next_after_cache(self, state: AgentState) -> str:
        if state.get("cache_hit"):
            return "cached_response"
        return state["route"]

    def _cache_key(self, request: ChatRequest, route: str) -> str:
        workspace_revision = self.tools.store.workspace_revision(request.workspace_id)
        raw = (
            f"v4|{route}|{request.workspace_id}|{workspace_revision}|"
            f"{request.message.strip().lower()}|{request.top_k}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
