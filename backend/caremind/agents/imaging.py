import time
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from ..schemas import ChatRequest, RetrievedChunk
from ..tools import DocumentTools, citations_from_chunks


NO_IMAGE_EVIDENCE_ANSWER = (
    "I could not inspect the uploaded image pixels, and there is no paired report or caption to use as evidence. "
    "Upload a PNG/JPEG/WebP image and enable a vision endpoint, or include a paired report."
)


class ImagingAgent:
    """Handles image-scoped questions using uploaded image metadata and paired report text."""

    def __init__(self, tools: DocumentTools, llm: LLMClient | None = None):
        self.tools = tools
        self.llm = llm

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
        image_assets = self._attachment_images(request)
        vision_provider = ""
        vision_model = ""
        vision_method = None
        if self.llm and getattr(self.llm.settings, "radiology_vision_base_url", None):
            vision_provider = "openai_compatible_medgemma"
            vision_model = self.llm.settings.radiology_vision_model
            vision_method = self.llm.analyze_radiology_image
        elif self.llm and getattr(self.llm.settings, "document_vlm_base_url", None):
            vision_provider = "openai_compatible_document_vlm"
            vision_model = self.llm.settings.document_vlm_model
            vision_method = self.llm.analyze_clinical_image
        vision_attempted = False
        vision_answer = ""
        fallback_reason = ""
        if vision_method and image_assets:
            vision_attempted = True
            image = image_assets[0]
            try:
                vision_answer = vision_method(
                    question=request.message,
                    file_path=Path(image.file_path),
                    content_type=image.content_type or "application/octet-stream",
                    paired_report="\n\n".join(chunk.text for chunk in grounded_chunks),
                ) or ""
            except Exception as exc:
                fallback_reason = f"image_vision_{exc.__class__.__name__}"
        elif not vision_method:
            fallback_reason = "image_vision_not_configured"
        elif not image_assets:
            fallback_reason = "no_current_image_attachment"
        trace_step = {
            "node": "imaging_agent",
            "tool": "message_attachment_lookup" if used_attachment_lookup else "imaging_search",
            "attachment_ids": request.attachment_ids,
            "retrieved_count": len(chunks),
            "grounded_count": len(grounded_chunks),
            "search_bypassed": used_attachment_lookup,
            "vision_attempted": vision_attempted,
            "vision_provider": vision_provider if vision_attempted else "",
            "vision_model": vision_model if vision_attempted else "",
            "fallback_reason": fallback_reason,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if vision_answer:
            paired_report_section = ""
            citations = []
            if grounded_chunks:
                paired_report_section = (
                    "\n\nPaired report/caption evidence:\n"
                    f"{grounded_chunks[0].text} [1]"
                )
                citations = citations_from_chunks(grounded_chunks)
            return {
                "answer": (
                    f"Experimental image analysis by {vision_model} (requires qualified human review):\n"
                    f"{vision_answer.strip()}"
                    f"{paired_report_section}"
                ),
                "citations": citations,
                "evidence_chunks": grounded_chunks,
                "retrieved_chunks": [self._chunk_trace(chunk) for chunk in grounded_chunks],
                "tool_calls": [*tool_calls, "radiology_vision" if vision_provider == "openai_compatible_medgemma" else "document_vision"],
                "trace_steps": [*trace_steps, trace_step],
                "require_citations": bool(citations),
                "safety_notes": [
                    "Pixel-level model output is experimental and requires qualified human review."
                ],
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

    def _attachment_images(self, request: ChatRequest) -> list[Any]:
        if not request.attachment_ids:
            return []
        lookup = getattr(self.tools, "attachment_image_assets", None)
        if not callable(lookup):
            return []
        return lookup(
            request.attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
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
