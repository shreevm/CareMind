import re

import httpx

from .config import Settings
from .schemas import ChatMessage, RetrievedChunk
from .tools import clean_evidence_text


SYSTEM_PROMPT = """You are CareMind, an agentic RAG assistant for medical and research documents.
Answer only from the provided evidence. Include concise clinical language, note uncertainty,
and never claim to diagnose or prescribe. Cite evidence using bracketed citation numbers."""

MEDICAL_SYSTEM_PROMPT = """You are CareMind's medical education specialist.
Use the provided education evidence first, answer in plain clinical language, and explain uncertainty.
Do not diagnose, prescribe, or replace a clinician. Cite evidence using bracketed citation numbers."""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage],
    ) -> str:
        if self.settings.nvidia_api_key:
            try:
                return self._answer_with_nvidia(question=question, chunks=chunks, history=history)
            except Exception:
                pass
        return self._answer_locally(question=question, chunks=chunks)

    def answer_medical(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage],
    ) -> str:
        if self.settings.medical_llm_base_url and self.settings.medical_llm_model:
            try:
                return self._answer_with_chat_endpoint(
                    question=question,
                    chunks=chunks,
                    history=history,
                    base_url=self.settings.medical_llm_base_url,
                    model=self.settings.medical_llm_model,
                    api_key=self.settings.medical_llm_api_key,
                    system_prompt=MEDICAL_SYSTEM_PROMPT,
                )
            except Exception:
                pass
        return self.answer(question=question, chunks=chunks, history=history)

    def _answer_with_nvidia(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage],
    ) -> str:
        return self._answer_with_chat_endpoint(
            question=question,
            chunks=chunks,
            history=history,
            base_url=self.settings.nvidia_base_url,
            model=self.settings.nvidia_chat_model,
            api_key=self.settings.nvidia_api_key,
            system_prompt=SYSTEM_PROMPT,
        )

    def _answer_with_chat_endpoint(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage],
        base_url: str,
        model: str,
        api_key: str | None,
        system_prompt: str,
    ) -> str:
        evidence = "\n\n".join(
            f"[{index}] {chunk.document_name}"
            f"{' page ' + str(chunk.page) if chunk.page else ''}: {chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )
        messages = [{"role": "system", "content": system_prompt}]
        for item in history[-6:]:
            if item.role in {"user", "assistant"}:
                messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Evidence:\n{evidence or 'No retrieved evidence.'}\n\n"
                    f"{self._response_instructions(question)}\n"
                    "Return a concise answer with citations such as [1]."
                ),
            }
        )
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 700,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

    def _answer_locally(self, *, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "I do not know based on the uploaded documents. Upload a relevant report or ask about a document "
                "that has already been indexed."
            )
        if self._asks_for_clinical_summary(question):
            return self._clinical_summary_answer(chunks)
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z]{4,}", question)}
        selected_sentences: list[str] = []
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", clean_evidence_text(chunk.text))
            for sentence in sentences:
                sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z]{4,}", sentence)}
                if query_terms & sentence_terms:
                    selected_sentences.append(sentence.strip())
                if len(selected_sentences) >= 4:
                    break
            if len(selected_sentences) >= 4:
                break
        if not selected_sentences:
            selected_sentences = [clean_evidence_text(chunks[0].text)[:450].strip()]
        summary = " ".join(selected_sentences)
        return f"Based on the retrieved document passages, {summary} [1]"

    def _asks_for_clinical_summary(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            phrase in lowered
            for phrase in [
                "key findings",
                "health condition",
                "condition of the patient",
                "patient condition",
                "tell me about the patient",
                "patient details",
                "patient profile",
                "clinical summary",
                "summarize",
            ]
        )

    def _response_instructions(self, question: str) -> str:
        if not self._asks_for_clinical_summary(question):
            return ""
        return (
            "For key-finding or patient-summary questions, structure the answer as Patient, Study, "
            "Heart Rate, Key Findings, Ectopics, and Overall/Clinical Context when the evidence supports it. "
            "Use short prose and bullet lists. Do not use markdown tables, HTML tables, or Parameter/Value tables. "
            "Do not use markdown formatting, bold text, or asterisks. Use plain headings like Patient Details:"
            "If this is a Holter/ECG report, distinguish the overall max/average/min heart rate from rhythm-table "
            "Avg HR entries. Do not list every rhythm-row average heart rate as a separate key finding. "
            "Highlight major positives and negatives such as SVT/VT/AF/AV block/pauses, symptom events, and ectopic burden."
        )

    def _clinical_summary_answer(self, chunks: list[RetrievedChunk]) -> str:
        evidence_texts = [clean_evidence_text(chunk.text) for chunk in chunks]
        combined = "\n".join(evidence_texts)
        patient_profile = self._extract_patient_profile(combined)
        findings = self._extract_holter_findings(combined) or self._extract_clinical_findings(evidence_texts)
        triggered = self._extract_unique_matches(
            combined,
            [
                r"Patient Triggered[^:\n]*(?::|\n)?\s*[^\n.]{0,160}",
                r"Manual[^:\n]*(?::|\n)?\s*[^\n.]{0,160}",
            ],
        )

        lines = [
            "From the uploaded report, this appears to be an ambulatory cardiac rhythm monitoring summary rather than a full diagnostic workup.",
        ]
        if patient_profile:
            lines.extend(
                [
                    "",
                    "Patient profile:",
                    f"- Name: {patient_profile.get('name', 'Not found')} [1]",
                    f"- ID: {patient_profile.get('id', 'Not found')} [1]",
                    f"- Age: {patient_profile.get('age', 'Not found')} [1]",
                    f"- Gender: {patient_profile.get('gender', 'Not found')} [1]",
                ]
            )
        lines.extend(["", "Key findings:"])
        if findings:
            for index, finding in enumerate(findings[:5], start=1):
                lines.append(f"- {finding} [{min(index, len(chunks))}]")
        else:
            lines.append("- I found rhythm-monitoring entries, but the extracted PDF text does not include a clear final impression section. [1]")

        if triggered:
            lines.append("")
            lines.append("Patient/manual event evidence:")
            for item in triggered[:3]:
                lines.append(f"- {item} [1]")

        lines.extend(
            [
                "",
                "Clinical interpretation:",
                "- The report evidence points to recorded sinus rhythm with episodes/entries involving sinus tachycardia and ectopic beats where listed. This should be interpreted by the ordering clinician or cardiologist in context of symptoms, medications, and the complete report.",
            ]
        )
        return "\n".join(lines)

    def _extract_holter_findings(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        lowered = compact.lower()
        if not any(term in lowered for term in ["holter", "ecg", "sinus", "svt", "ectopic", "heart rate", "hr"]):
            return []

        findings: list[str] = []

        max_hr = self._first_match(
            compact,
            [
                r"\b(?:max|maximum)(?:imum)?(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b",
            ],
        )
        avg_hr = self._first_match(
            compact,
            [
                r"\b(?:avg|average)(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b",
            ],
        )
        min_hr = self._first_match(
            compact,
            [
                r"\b(?:min|minimum)(?:\s+(?:heart\s+rate|hr))?\s*(?:[:=]|~|-)?\s*(\d+\s*bpm)\b",
            ],
        )
        heart_rate_parts = []
        if max_hr:
            heart_rate_parts.append(f"maximum {max_hr}")
        if avg_hr:
            heart_rate_parts.append(f"average {avg_hr}")
        if min_hr:
            heart_rate_parts.append(f"minimum {min_hr}")
        if heart_rate_parts:
            findings.append("Overall heart rate: " + ", ".join(heart_rate_parts) + ".")

        if re.search(r"\bsinus rhythm\b", compact, flags=re.IGNORECASE):
            findings.append("Baseline/recorded rhythm includes sinus rhythm.")

        svt_count = self._first_match(
            compact,
            [
                r"\b(\d+)\s+episodes?\s+of\s+SVT\b",
                r"\bSVT\s+episodes?\D{0,40}(\d+)\b",
            ],
        )
        if svt_count:
            svt_detail = self._first_match(
                compact,
                [
                    r"\bSVT\b.{0,80}?(\d+\s*bpm.{0,40}?\d+(?:\.\d+)?\s*(?:secs?|seconds?))",
                    r"(\d+\s*bpm.{0,40}?\d+(?:\.\d+)?\s*(?:secs?|seconds?)).{0,40}?\bSVT\b",
                ],
            )
            detail = f" ({svt_detail})" if svt_detail else ""
            findings.append(f"SVT noted: {svt_count} episode(s){detail}.")
        elif re.search(r"\bSVT\b|supraventricular tachycardia", compact, flags=re.IGNORECASE):
            findings.append("Supraventricular tachycardia/SVT is mentioned in the report.")

        if re.search(r"\bsinus tachycardia\b", compact, flags=re.IGNORECASE):
            tachy_hr = self._first_match(compact, [r"\bsinus tachycardia\b.{0,60}?(\d+\s*bpm)"])
            findings.append(f"Sinus tachycardia is noted{f' around {tachy_hr}' if tachy_hr else ''}.")

        ventricular, supraventricular = self._extract_ectopic_counts(compact)
        ectopic_parts = []
        if ventricular:
            ectopic_parts.append(f"ventricular ectopics {ventricular}")
        if supraventricular:
            ectopic_parts.append(f"supraventricular ectopics {supraventricular}")
        if ectopic_parts:
            findings.append("Ectopic burden/counts: " + "; ".join(ectopic_parts) + ".")
        elif re.search(r"\bectopic", compact, flags=re.IGNORECASE):
            findings.append("Ectopic beats are mentioned; use the source report for exact burden/counts.")

        negatives = []
        negative_patterns = [
            ("VT", r"\bno\s+(?:vt|ventricular tachycardia)\b"),
            ("AF", r"\bno\s+(?:af|atrial fibrillation)\b"),
            ("advanced AV block", r"\bno\s+(?:advanced\s+)?av blocks?\b"),
            ("pauses", r"\bno\s+pauses?\b"),
        ]
        for label, pattern in negative_patterns:
            if re.search(pattern, compact, flags=re.IGNORECASE):
                negatives.append(label)
        if negatives:
            findings.append("No " + ", ".join(negatives) + " reported.")

        if re.search(r"\bno symptoms?\b|\bno patient triggered\b|\bno manual\b", compact, flags=re.IGNORECASE):
            findings.append("No patient symptoms/manual events are documented in the retrieved evidence.")

        return self._dedupe_findings(findings)

    def _extract_ectopic_counts(self, text: str) -> tuple[str, str]:
        table_match = re.search(
            r"\bVentricular\s+Supraventricular\s+Total\s+([0-9,]+)\s+([0-9,]+)",
            text,
            flags=re.IGNORECASE,
        )
        if table_match:
            return table_match.group(1), table_match.group(2)
        ventricular = self._first_match(
            text,
            [
                r"\bventricular(?:\s+ectopic(?:s|\s*\(?ve\)?)?)?\s*[:=-]?\s*([0-9,]+)\s*(?:total|beats?)\b",
            ],
        )
        supraventricular = self._first_match(
            text,
            [
                r"\bsupraventricular(?:\s+ectopic(?:s|\s*\(?sve\)?)?)?\s*[:=-]?\s*([0-9,]+)\s*(?:total|beats?)\b",
            ],
        )
        return ventricular, supraventricular

    def _first_match(self, text: str, patterns: list[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,;")
        return ""

    def _dedupe_findings(self, findings: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for finding in findings:
            key = finding.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(finding)
        return deduped

    def _extract_patient_profile(self, text: str) -> dict[str, str]:
        compact = re.sub(r"\s+", " ", text).strip()
        profile_match = re.search(
            r"Report for\s+(.+?)\s+ID\s+([A-Za-z0-9-]+)\s+Age\s+(\d{1,3}\s*(?:yrs?|years?)?)\s+Gender\s+([A-Za-z-]+)",
            compact,
            flags=re.IGNORECASE,
        )
        if profile_match:
            return {
                "name": profile_match.group(1).strip(),
                "id": profile_match.group(2).strip(),
                "age": profile_match.group(3).strip(),
                "gender": profile_match.group(4).strip(),
            }

        def find(pattern: str) -> str:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            return match.group(1).strip() if match else ""

        profile = {
            "name": find(r"(?:patient name|name)\s*[:=-]\s*([A-Za-z ,.'-]+)"),
            "id": find(r"(?:patient id|mrn|medical record(?: number)?|id)\s*[:#=-]?\s*([A-Za-z0-9-]+)"),
            "age": find(r"age\s*[:=-]?\s*(\d{1,3}\s*(?:yrs?|years?)?)"),
            "gender": find(r"(?:gender|sex)\s*[:=-]?\s*(male|female|man|woman|nonbinary|non-binary|other)"),
        }
        return {key: value for key, value in profile.items() if value}

    def _extract_clinical_findings(self, evidence_texts: list[str]) -> list[str]:
        findings: list[str] = []
        for text in evidence_texts:
            for line in re.split(r"[\n\r]+", text):
                compact = re.sub(r"\s+", " ", line).strip(" -:")
                if len(compact) < 12:
                    continue
                lowered = compact.lower()
                if any(
                    term in lowered
                    for term in [
                        "sinus rhythm",
                        "sinus tachycardia",
                        "sinus bradycardia",
                        "ectopic",
                        "avg hr",
                        "patient triggered",
                        "manual",
                        "summary report",
                    ]
                ):
                    findings.append(compact[:220])
        deduped = []
        seen = set()
        for finding in findings:
            key = finding.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(finding)
        return deduped

    def _extract_unique_matches(self, text: str, patterns: list[str]) -> list[str]:
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        deduped = []
        seen = set()
        for match in matches:
            compact = re.sub(r"\s+", " ", match).strip(" -:")
            key = compact.lower()
            if compact and key not in seen:
                seen.add(key)
                deduped.append(compact)
        return deduped
