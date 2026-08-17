from typing import Literal

from ..intent import asks_general_medical_not_document, has_document_scope
from ..safety import EMERGENCY_OR_DOSING


Route = Literal[
    "direct",
    "retrieve",
    "compare",
    "medical_education",
    "imaging",
    "clarify",
    "emergency_redirect",
    "prompt_injection_blocked",
]


class SupervisorAgent:
    """Routes a user question to the narrow specialist agent that should handle it."""

    def route(self, message: str) -> Route:
        lowered = message.lower()
        normalized = lowered.strip(" ?!.")
        document_reference = self._has_document_reference(lowered)
        if EMERGENCY_OR_DOSING.search(lowered) and not document_reference:
            return "emergency_redirect"
        direct_patterns = [
            "is this caremind",
            "what is caremind",
            "who are you",
            "tell me about yourself",
            "tell me about you",
            "about yourself",
            "introduce yourself",
            "what do you do",
            "what do u do",
            "what can you do",
            "what can u do",
            "how do you work",
            "how do u work",
            "how to use",
            "how can i use",
            "what are your capabilities",
        ]
        greetings = {"help", "hello", "hi", "hey"}
        if normalized in greetings or any(pattern in normalized for pattern in direct_patterns):
            return "direct"
        if self._is_report_comparison(lowered):
            return "compare"
        if document_reference:
            return "retrieve"

        if asks_general_medical_not_document(lowered):
            return "medical_education"
        if len(lowered.split()) < 3 and "?" not in lowered:
            return "clarify"
        return "clarify"

    def _is_report_comparison(self, lowered: str) -> bool:
        comparison_terms = ["compare", "changed", "difference", "trend", "versus", "vs "]
        document_terms = ["report", "reports", "document", "documents", "lab", "labs", "result", "results", "uploaded"]
        return any(term in lowered for term in comparison_terms) and any(term in lowered for term in document_terms)

    def _has_document_reference(self, lowered: str) -> bool:
        return has_document_scope(lowered) or "key findings" in lowered
