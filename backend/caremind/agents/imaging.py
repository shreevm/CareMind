import time
from typing import Any

from ..schemas import ChatRequest, RetrievedChunk
from ..tools import DocumentTools, citations_from_chunks


NO_IMAGE_EVIDENCE_ANSWER = (
    "I do not have an uploaded image with a paired report or caption to ground this answer. "
    "Upload a clinical image and, when possible, include the associated report text so CareMind can cite what it uses."
)


class ImagingAgent:
    """Handles image-scoped questions using uploaded image metadata and paired report text."""

    def __init__(self, tools: DocumentTools):
        self.tools = tools

    def run(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        started = time.perf_counter()
        attachment_chunks = self._attachment_chunks(request)
        used_attachment_lookup = bool(attachment_chunks)
        chunks = attachment_chunks or self.tools.imaging_search(
            request.message,
            workspace_id=request.workspace_id,
            top_k=min(request.top_k, 3),
        )
        grounded_chunks = [chunk for chunk in chunks if (chunk.score or 0.0) > 0 and self._has_paired_report(chunk)]
        trace_step = {
            "node": "imaging_agent",
            "tool": "message_attachment_lookup" if used_attachment_lookup else "imaging_search",
            "attachment_ids": request.attachment_ids,
            "retrieved_count": len(chunks),
            "grounded_count": len(grounded_chunks),
            "search_bypassed": used_attachment_lookup,
            "vision_attempted": False,
            "vision_provider": "",
            "vision_model": "",
            "fallback_reason": "medgemma_not_configured" if not grounded_chunks else "",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if not grounded_chunks:
            return {
                "answer": NO_IMAGE_EVIDENCE_ANSWER,
                "citations": [],
                "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
                "no_relevant_document_evidence": True,
                "tool_calls": [*tool_calls, trace_step["tool"]],
                "trace_steps": [*trace_steps, trace_step],
                "require_citations": False,
            }

        answer = (
            "Based on the paired report or caption for the uploaded image, "
            f"{grounded_chunks[0].text} [1]\n\n"
            "I am not independently diagnosing the pixels; this answer is grounded in the provided image-linked text."
        )
        return {
            "answer": answer,
            "citations": citations_from_chunks(grounded_chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in grounded_chunks],
            "tool_calls": [*tool_calls, trace_step["tool"]],
            "trace_steps": [*trace_steps, trace_step],
        }

    def _attachment_chunks(self, request: ChatRequest) -> list[RetrievedChunk]:
        if not request.attachment_ids:
            return []
        lookup = getattr(self.tools, "attachment_evidence_chunks", None)
        if not callable(lookup):
            return []
        return lookup(
            request.attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
            top_k=max(3, len(request.attachment_ids)),
        )

    def _has_paired_report(self, chunk: RetrievedChunk) -> bool:
        return "Paired report or caption for grounding:" in chunk.text

    def _chunk_trace(self, chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "page": chunk.page,
            "score": chunk.score,
            "preview": chunk.text[:500],
        }
