import re

UNSUPPORTED_MEDICAL_ADVICE = re.compile(
    r"\b(diagnose me|should i take|dosage|dose|prescribe|stop taking|emergency|chest pain|suicidal)\b",
    re.IGNORECASE,
)

DISCLAIMER = (
    "CareMind is for medical education and document interpretation only. "
    "It is not a diagnosis or treatment plan."
)


class SafetyLayer:
    def inspect_user_message(self, message: str) -> list[str]:
        notes = []
        if UNSUPPORTED_MEDICAL_ADVICE.search(message):
            notes.append(
                "The question may be asking for personal medical advice. I will stay grounded in the uploaded documents."
            )
        return notes

    def finalize(
        self,
        answer: str,
        has_citations: bool,
        safety_notes: list[str],
        require_citations: bool = True,
    ) -> tuple[str, list[str]]:
        notes = list(safety_notes)
        if require_citations and not has_citations:
            notes.append("No supporting document passage was found for this answer.")
        if DISCLAIMER not in answer:
            answer = f"{answer.strip()}\n\n{DISCLAIMER}"
        return answer, notes
