class DirectResponseAgent:
    """Answers product/help questions without touching retrieved medical records."""

    def answer(self, message: str) -> str:
        lowered = message.lower()
        if any(term in lowered for term in ["what do", "what can", "tell me", "about yourself", "help", "use", "capab"]):
            return (
                "I am CareMind, a medical-document assistant. I can index uploaded PDFs or text reports, "
                "answer questions using those documents with citations, compare two reports, and answer general "
                "medical education questions. I should not diagnose, prescribe treatment, or replace a clinician."
            )
        return (
            "Yes, this is CareMind. I can upload and search medical PDFs or text files, "
            "answer with citations from those documents, compare two reports, and answer general "
            "medical education questions with cited educational context."
        )
