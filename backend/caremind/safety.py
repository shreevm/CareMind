import re

UNSUPPORTED_MEDICAL_ADVICE = re.compile(
    r"\b(diagnose me|should i take|dosage|dose|prescribe|stop taking|emergency|chest pain|suicidal)\b",
    re.IGNORECASE,
)
EMERGENCY_OR_DOSING = re.compile(
    r"\b("
    r"chest pain|trouble breathing|difficulty breathing|shortness of breath now|can't breathe|"
    r"stroke|stroke signs|face drooping|weakness on one side|heart attack|"
    r"testicular torsion|torsion of the testicle|suicidal|suicide|overdose|"
    r"bleeding heavily|severe bleeding|uncontrolled bleeding|"
    r"911|call 911|emergency|call emergency|"
    r"double dose|dosage|dose|should i take|how much .* should i take|"
    r"stop taking|start taking|prescribe"
    r")\b",
    re.IGNORECASE,
)
PROMPT_INJECTION = re.compile(
    r"\b("
    r"ignore (?:all )?(?:previous|prior|above)? instructions|"
    r"ignore (?:your|my|the) (?:guidelines|rules|instructions)|"
    r"disregard (?:all )?(?:previous|prior|above)? instructions|"
    r"disregard (?:your|my|the) (?:guidelines|rules|instructions)|"
    r"override (?:your|my|the) (?:guidelines|rules|instructions)|"
    r"system prompt|developer message|reveal.*prompt|"
    r"jailbreak|DAN mode|bypass (?:safety|guardrails)|"
    r"do not cite|without citations|forget your instructions|"
    r"pretend (?:you|you're|you are) (?:a|different|another)"
    r")\b",
    re.IGNORECASE,
)
OUTPUT_LEAK = re.compile(
    r"("
    r"\b(?:system|developer|hidden)\s+(?:prompt|message|instructions)\s*(?:is|are|:)|"
    r"<\s*/?\s*system\s*>|"
    r"\bBEGIN[ _-]?SYSTEM\b|\bEND[ _-]?SYSTEM\b|"
    r"\b(?:NVIDIA_API_KEY|SUPABASE_SECRET_KEY|SUPABASE_SERVICE_ROLE_KEY|LANGCHAIN_API_KEY|"
    r"MEDICAL_LLM_API_KEY|CAREMIND_PASSWORD)\s*=\s*\S+|"
    r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}|"
    r"\b(?:sk|nvapi)-[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)
REASONING_LEAK_PREFIX = re.compile(
    r"^\s*(?:here(?:'s| is)\s+(?:a\s+)?(?:thinking|reasoning|analysis)\s+process\s*:|"
    r"(?:thinking|reasoning|analysis)\s+process\s*:|"
    r"let(?:'s| us)\s+(?:think|reason)\s+(?:through\s+)?(?:this|it)\s*:)\s*",
    re.IGNORECASE,
)
REASONING_STEP_MARKER = re.compile(
    r"^\s*\d+\.\s+(?:analy[sz]e|review|extract|identify|synthesi[sz]e|draft|outline|check)\b",
    re.IGNORECASE | re.MULTILINE,
)

DISCLAIMER = (
    "CareMind is for medical education and document interpretation only. "
    "It is not a diagnosis or treatment plan."
)
DISCLAIMER_GUARDRAIL_NOTE = "Output guardrail confirmed the medical education and document interpretation disclaimer."
OUTPUT_LEAK_GUARDRAIL_NOTE = "Output guardrail blocked possible system prompt or internal configuration leakage."
REASONING_LEAK_GUARDRAIL_NOTE = "Output guardrail removed hidden reasoning-style text from the final answer."
OUTPUT_LEAK_RESPONSE = (
    "I can't display internal instructions, hidden prompts, secrets, or private configuration. "
    "Ask your medical-document or education question directly, and I will answer from the available evidence."
)


class SafetyLayer:
    def should_emergency_redirect(self, message: str) -> bool:
        return bool(EMERGENCY_OR_DOSING.search(message))

    def detects_prompt_injection(self, message: str) -> bool:
        return bool(PROMPT_INJECTION.search(message))

    def detects_output_leak(self, answer: str) -> bool:
        return bool(OUTPUT_LEAK.search(answer))

    def detects_reasoning_leak(self, answer: str) -> bool:
        return bool(REASONING_LEAK_PREFIX.search(answer) or REASONING_STEP_MARKER.search(answer[:2000]))

    def strip_reasoning_leak(self, answer: str) -> str:
        if not self.detects_reasoning_leak(answer):
            return answer
        markers = [
            "Patient Details",
            "Report instructions",
            "General next steps",
            "Heart Rate",
            "Key Findings",
            "Overall/Clinical Context",
            "The uploaded report",
            "From the retrieved evidence",
            "No instructions were found",
        ]
        lower_answer = answer.lower()
        best_index: int | None = None
        for marker in markers:
            index = lower_answer.find(marker.lower())
            if index >= 0 and (best_index is None or index < best_index):
                best_index = index
        if best_index is None:
            return OUTPUT_LEAK_RESPONSE
        return answer[best_index:].strip()

    def inspect_user_message(self, message: str) -> list[str]:
        notes = []
        if UNSUPPORTED_MEDICAL_ADVICE.search(message):
            notes.append(
                "The question may be asking for personal medical advice. I will stay grounded in the uploaded documents."
            )
        if self.detects_prompt_injection(message):
            notes.append(
                "The request contains instruction-override language, so I will ignore that portion and keep the answer grounded in evidence."
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
        if self.detects_output_leak(answer):
            answer = OUTPUT_LEAK_RESPONSE
            require_citations = False
            if OUTPUT_LEAK_GUARDRAIL_NOTE not in notes:
                notes.append(OUTPUT_LEAK_GUARDRAIL_NOTE)
        elif self.detects_reasoning_leak(answer):
            answer = self.strip_reasoning_leak(answer)
            if REASONING_LEAK_GUARDRAIL_NOTE not in notes:
                notes.append(REASONING_LEAK_GUARDRAIL_NOTE)
        if require_citations and not has_citations:
            notes.append("No supporting document passage was found for this answer.")
        if DISCLAIMER not in answer:
            answer = f"{answer.strip()}\n\n{DISCLAIMER}"
        if DISCLAIMER_GUARDRAIL_NOTE not in notes:
            notes.append(DISCLAIMER_GUARDRAIL_NOTE)
        return answer, notes
