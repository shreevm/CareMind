from typing import Literal


Route = Literal["direct", "retrieve", "compare", "medical_education", "clarify"]


class SupervisorAgent:
    """Routes a user question to the narrow specialist agent that should handle it."""

    def route(self, message: str) -> Route:
        lowered = message.lower()
        normalized = lowered.strip(" ?!.")
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
        if any(term in lowered for term in ["compare", "changed", "difference", "trend", "versus", "vs "]):
            return "compare"

        education_terms = [
            "what is",
            "what are",
            "explain",
            "tell me about",
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
            "iron deficiency",
            "iron-deficiency",
            "ferritin",
            "hemoglobin",
            "b12 deficiency",
            "vitamin deficiency",
            "chest pain",
        ]
        document_terms = [
            "report",
            "document",
            "uploaded",
            "based on this",
            "based on the",
            "in this",
            "in the file",
            "evidence supports",
            "key findings",
            "patient",
        ]
        if any(term in lowered for term in education_terms) and not any(term in lowered for term in document_terms):
            return "medical_education"
        if len(lowered.split()) < 3 and "?" not in lowered:
            return "clarify"
        return "retrieve"
