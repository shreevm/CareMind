class DirectResponseAgent:
    """Answers product/help questions without touching retrieved medical records."""

    def answer(self, message: str) -> str:
        lowered = message.lower().strip().rstrip("!?.")
        # Handle simple greetings
        if lowered in {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}:
            return "Hello! I am CareMind. Do you want help with anything?"

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

