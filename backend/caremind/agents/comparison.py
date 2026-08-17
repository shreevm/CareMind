import time
from typing import Any

from ..schemas import ChatRequest
from ..tools import DocumentTools


class ReportComparisonAgent:
    """Compares two indexed reports and returns cited differences."""

    def __init__(self, tools: DocumentTools):
        self.tools = tools

    def run(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        started = time.perf_counter()
        selected_document_ids = self._attachment_document_ids(request)
        if selected_document_ids:
            if len(selected_document_ids) < 2:
                return {
                    "answer": "Please attach two indexed reports before asking me to compare them.",
                    "trace_steps": [
                        *trace_steps,
                        {
                            "node": "report_comparison_agent",
                            "tool": "message_attachment_lookup",
                            "attachment_ids": request.attachment_ids,
                            "document_count": len(selected_document_ids),
                            "selection_reason": "current_message_attachments",
                            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        },
                    ],
                }
            comparison = self.tools.compare_reports(
                selected_document_ids[:2],
                workspace_id=request.workspace_id,
            )
            return {
                "answer": comparison.summary,
                "citations": comparison.citations,
                "retrieved_chunks": [citation.model_dump(mode="json") for citation in comparison.citations],
                "tool_calls": [*tool_calls, "message_attachment_lookup", "compare_reports"],
                "trace_steps": [
                    *trace_steps,
                    {
                        "node": "report_comparison_agent",
                        "tool": "compare_reports",
                        "attachment_ids": request.attachment_ids,
                        "document_ids": selected_document_ids[:2],
                        "document_count": 2,
                        "search_bypassed": True,
                        "selection_reason": "current_message_attachments",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                ],
            }
        documents = self.tools.store.list_documents(request.workspace_id)
        if len(documents) < 2:
            return {
                "answer": "Please upload at least two reports before asking me to compare them.",
                "trace_steps": [
                    *trace_steps,
                    {
                        "node": "report_comparison_agent",
                        "tool": "compare_reports",
                        "document_count": len(documents),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                ],
            }
        comparison = self.tools.compare_reports(
            [doc.document_id for doc in documents[:2]],
            workspace_id=request.workspace_id,
        )
        return {
            "answer": comparison.summary,
            "citations": comparison.citations,
            "retrieved_chunks": [citation.model_dump(mode="json") for citation in comparison.citations],
            "tool_calls": [*tool_calls, "compare_reports"],
            "trace_steps": [
                *trace_steps,
                {
                    "node": "report_comparison_agent",
                    "tool": "compare_reports",
                    "document_count": min(len(documents), 2),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            ],
        }

    def _attachment_document_ids(self, request: ChatRequest) -> list[str]:
        if not request.attachment_ids:
            return []
        resolver = getattr(self.tools, "attachment_document_ids", None)
        if not callable(resolver):
            return []
        return resolver(
            request.attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
        )
