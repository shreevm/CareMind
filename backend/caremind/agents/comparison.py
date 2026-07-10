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
