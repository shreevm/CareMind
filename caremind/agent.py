import hashlib
import time
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .llm import LLMClient
from .memory import ConversationMemory
from .metrics import metrics
from .safety import SafetyLayer
from .schemas import ChatRequest, ChatResponse, Citation
from .tools import DocumentTools, citations_from_chunks


Route = Literal["direct", "retrieve", "compare", "medical_education", "clarify"]


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
        self.graph = self._build_graph()

    def answer(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke({"request": request, "started_at": time.perf_counter()})
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
        route = self._route(request.message)
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
                    "route": route,
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
                    "notes": "Rule-based router selected the workflow branch.",
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
            "answer": (
                "Yes, this is CareMind. I can upload and search medical PDFs or text files, "
                "answer with citations from those documents, compare two reports, and answer general "
                "medical education questions with cited educational context."
            ),
        }

    def _clarify_node(self, state: AgentState) -> AgentState:
        return {
            **state,
            "answer": "Which uploaded report or medical topic should I use as the evidence source?",
        }

    def _compare_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        started = time.perf_counter()
        documents = self.tools.store.list_documents(request.workspace_id)
        if len(documents) < 2:
            return {**state, "answer": "Please upload at least two reports before asking me to compare them."}
        comparison = self.tools.compare_reports(
            [doc.document_id for doc in documents[:2]],
            workspace_id=request.workspace_id,
        )
        return {
            **state,
            "answer": comparison.summary,
            "citations": comparison.citations,
            "retrieved_chunks": [citation.model_dump(mode="json") for citation in comparison.citations],
            "tool_calls": [*state.get("tool_calls", []), "compare_reports"],
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "compare",
                    "tool": "compare_reports",
                    "document_count": min(len(documents), 2),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

    def _medical_education_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        started = time.perf_counter()
        chunks = self.tools.medical_education_search(request.message, top_k=min(request.top_k, 3))
        history = self.memory.load(request.session_id, request.workspace_id)
        return {
            **state,
            "answer": self.llm.answer(question=request.message, chunks=chunks, history=history),
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "tool_calls": [*state.get("tool_calls", []), "medical_education_search"],
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "medical_education",
                    "tool": "medical_education_search",
                    "retrieved_count": len(chunks),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        request = state["request"]
        started = time.perf_counter()
        chunks = self.tools.document_search(
            request.message,
            workspace_id=request.workspace_id,
            top_k=request.top_k,
        )
        history = self.memory.load(request.session_id, request.workspace_id)
        return {
            **state,
            "answer": self.llm.answer(question=request.message, chunks=chunks, history=history),
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "tool_calls": [*state.get("tool_calls", []), "document_search"],
            "trace_steps": [
                *state.get("trace_steps", []),
                {
                    "node": "retrieve",
                    "tool": "document_search",
                    "top_k": request.top_k,
                    "retrieved_count": len(chunks),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

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

    def _route(self, message: str) -> Route:
        lowered = message.lower()
        direct_patterns = [
            "is this caremind",
            "what is caremind",
            "who are you",
            "what can you do",
            "help",
            "hello",
            "hi",
        ]
        if any(pattern in lowered.strip(" ?!.") for pattern in direct_patterns):
            return "direct"
        if len(lowered.split()) < 3 and "?" not in lowered:
            return "clarify"
        if any(term in lowered for term in ["compare", "changed", "difference", "trend", "versus", "vs "]):
            return "compare"
        education_terms = [
            "what is",
            "explain",
            "symptoms",
            "causes",
            "treatment",
            "guideline",
            "general",
            "education",
            "pneumonia",
            "diabetes",
            "hypertension",
            "anemia",
            "chest pain",
        ]
        if any(term in lowered for term in education_terms) and not any(
            term in lowered for term in ["report", "document", "uploaded", "evidence supports", "key findings"]
        ):
            return "medical_education"
        return "retrieve"

    def _cache_key(self, request: ChatRequest, route: str) -> str:
        raw = f"{route}|{request.workspace_id}|{request.message.strip().lower()}|{request.top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _chunk_trace(self, chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "page": chunk.page,
            "score": chunk.score,
            "preview": chunk.text[:500],
        }
