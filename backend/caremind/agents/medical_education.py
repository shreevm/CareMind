import time
from typing import Any

from ..llm import LLMClient
from ..memory import ConversationMemory
from ..schemas import ChatRequest, RetrievedChunk
from ..tools import DocumentTools, citations_from_chunks


class MedicalEducationAgent:
    """Answers general medical education questions from the trusted education corpus."""

    def __init__(self, tools: DocumentTools, llm: LLMClient, memory: ConversationMemory):
        self.tools = tools
        self.llm = llm
        self.memory = memory

    def run(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        updates = self.prepare(request, tool_calls, trace_steps)
        chunks = updates.pop("chunks")
        history = updates.pop("history")
        return {
            **updates,
            "answer": self.llm.answer_medical(
                question=request.message,
                chunks=chunks,
                history=history,
                response_language=request.response_language,
                allow_model_context=updates.get("medical_model_context_used", False),
            ),
        }

    def prepare(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        started = time.perf_counter()
        internal_chunks = self.tools.medical_education_search(request.message, top_k=min(request.top_k, 3))
        medlineplus_chunks: list[RetrievedChunk] = []
        external_chunks: list[RetrievedChunk] = []
        used_tools = [*tool_calls, "medical_education_search"]
        if self.tools.medlineplus_available():
            medlineplus_chunks = self.tools.medlineplus_health_topic_search(
                request.message,
                top_k=min(request.top_k, self.tools.embeddings.settings.medlineplus_max_results),
            )
            used_tools.append("medlineplus_health_topic_search")
        if self._asks_for_external_literature(request.message):
            external_chunks = self.tools.external_literature_search(
                request.message,
                top_k=min(request.top_k, 4),
            )
            used_tools.append("pubmed_literature_search")
        chunks = [*external_chunks, *medlineplus_chunks, *internal_chunks]
        history = self.memory.load(request.session_id, request.workspace_id)
        evidence_chars = sum(len(chunk.text or "") for chunk in chunks)
        model_context_used = (
            self.tools.embeddings.settings.medical_education_model_context_enabled
            and not external_chunks
            and evidence_chars < self.tools.embeddings.settings.medical_education_model_context_min_chars
        )
        return {
            "chunks": chunks,
            "history": history,
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "tool_calls": used_tools,
            "medical_model_context_used": model_context_used,
            "require_citations": bool(chunks) or not model_context_used,
            "trace_steps": [
                *trace_steps,
                {
                    "node": "medical_education_agent",
                    "tool": "+".join(used_tools),
                    "medlineplus_enabled": self.tools.medlineplus_available(),
                    "medlineplus_retrieved_count": len(medlineplus_chunks),
                    "external_literature": bool(external_chunks),
                    "internal_retrieved_count": len(internal_chunks),
                    "external_retrieved_count": len(external_chunks),
                    "retrieved_count": len(chunks),
                    "evidence_chars": evidence_chars,
                    "model_context_used": model_context_used,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

    def _asks_for_external_literature(self, message: str) -> bool:
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
            ]
        )

    def _chunk_trace(self, chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "page": chunk.page,
            "score": chunk.score,
            "preview": chunk.text[:500],
        }
