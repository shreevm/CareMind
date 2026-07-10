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
        started = time.perf_counter()
        chunks = self.tools.medical_education_search(request.message, top_k=min(request.top_k, 3))
        history = self.memory.load(request.session_id, request.workspace_id)
        return {
            "answer": self.llm.answer_medical(question=request.message, chunks=chunks, history=history),
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "tool_calls": [*tool_calls, "medical_education_search"],
            "trace_steps": [
                *trace_steps,
                {
                    "node": "medical_education_agent",
                    "tool": "medical_education_search",
                    "retrieved_count": len(chunks),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

    def _chunk_trace(self, chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "page": chunk.page,
            "score": chunk.score,
            "preview": chunk.text[:500],
        }
