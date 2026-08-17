import re
from typing import Any

from ..intent import asks_general_medical_not_document, explicitly_general_topic


class ConversationResolverAgent:
    """Resolves follow-up turns into standalone, retrieval-ready queries."""
    
    def resolve(
        self,
        *,
        message: str,
        context: dict[str, Any],
        history: list[Any],
        documents: list[Any],
        images: list[Any],
        current_attachments: list[Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._rewrite_query(message)
        current_attachments = current_attachments or []
        session_state = self._session_state(context)
        active_patient = session_state.get("current_patient") or ""
        active_document_name = session_state.get("current_document") or session_state.get("current_report") or ""
        active_document_id = context.get("active_document_id") or ""
        active_image = session_state.get("current_image") or ""
        entities = self._conversation_entities(context, normalized, documents, images)
        attachment_document_ids = [
            str(getattr(attachment, "document_id", "") or "")
            for attachment in current_attachments
            if getattr(attachment, "document_id", None)
        ]
        attachment_image_ids = [
            str(getattr(attachment, "image_asset_id", "") or "")
            for attachment in current_attachments
            if getattr(attachment, "image_asset_id", None)
        ]
        active_document_ids = [active_document_id] if active_document_id else []
        if attachment_document_ids:
            active_document_ids = attachment_document_ids
        if not active_document_ids and active_document_name:
            active_document_ids = [
                getattr(document, "document_id", "")
                for document in documents
                if getattr(document, "filename", "") == active_document_name
            ]
            active_document_ids = [document_id for document_id in active_document_ids if document_id]

        base = {
            "agent": "ConversationResolverAgent",
            "original_query": normalized,
            "resolved_query": normalized,
            "rewritten_query": normalized,
            "used_context": False,
            "is_followup": False,
            "reason": "standalone_query",
            "confidence": 0.0,
            "conversation_entities": entities,
            "active_document_ids": active_document_ids,
            "active_patient": active_patient,
            "active_document_name": active_document_name,
            "active_image": active_image,
            "session_state": session_state,
            "history_count": len(history),
            "current_message_attachment": bool(current_attachments),
            "active_attachment_ids": [str(getattr(attachment, "id", "") or "") for attachment in current_attachments],
            "current_attachment_types": [str(getattr(attachment, "attachment_type", "") or "") for attachment in current_attachments],
            "active_image_ids": attachment_image_ids or ([context.get("active_image_id")] if context.get("active_image_id") else []),
            "selected_attachment": str(getattr(current_attachments[0], "id", "") or "") if current_attachments else "",
        }
        
        def with_intent(payload: dict[str, Any], *, route_hint: str = "", confidence: float | None = None) -> dict[str, Any]:
            intent_rewrite = self._rewrite_current_intent(
                original_query=normalized,
                resolved_query=payload["rewritten_query"],
                route_hint=route_hint,
                confidence=payload["confidence"] if confidence is None else confidence,
                used_context=payload["used_context"],
                session_state=session_state,
                documents=documents,
                images=images,
            )
            return {**payload, "intent_rewrite": intent_rewrite}

        lowered = normalized.lower()
        if current_attachments:
            target_names = ", ".join(str(getattr(item, "filename", "attachment")) for item in current_attachments)
            if len(current_attachments) >= 2 and self._looks_like_contextual_comparison(lowered):
                rewritten = f"Compare the attachments from the current message: {target_names}. {normalized}"
                return with_intent(
                    {
                        **base,
                        "resolved_query": rewritten,
                        "rewritten_query": rewritten,
                        "used_context": True,
                        "is_followup": False,
                        "reason": "current_message_attachments",
                        "confidence": 0.98,
                    },
                    route_hint="compare",
                    confidence=0.98,
                )
            first_type = str(getattr(current_attachments[0], "attachment_type", "attachment") or "attachment")
            route_hint = "imaging" if first_type == "image" and not attachment_document_ids else "retrieve"
            rewritten = f"Using the attachment(s) from the current message ({target_names}), {normalized}"
            return with_intent(
                {
                    **base,
                    "resolved_query": rewritten,
                    "rewritten_query": rewritten,
                    "used_context": True,
                    "is_followup": False,
                    "reason": "current_message_attachments",
                    "confidence": 0.98,
                },
                route_hint=route_hint,
                confidence=0.98,
            )

        if self._has_general_education_context(context) and self._is_general_education_followup(lowered):
            topic = str(context.get("current_topic") or session_state.get("current_topic") or "the previous medical topic")
            rewritten = f"In general medical education context of {topic}, {normalized}"
            return with_intent(
                {
                    **base,
                    "resolved_query": rewritten,
                    "rewritten_query": rewritten,
                    "used_context": True,
                    "is_followup": True,
                    "reason": "general_education_contextual_followup",
                    "confidence": 0.88,
                },
                route_hint="medical_education",
                confidence=0.88,
            )

        if self._explicitly_changes_to_general_topic(lowered):
            return with_intent({**base, "reason": "explicit_general_topic"}, route_hint="medical_education", confidence=0.9)
        if self._asks_general_not_patient(lowered):
            return with_intent(
                {**base, "reason": "standalone_general_education_query"},
                route_hint="medical_education",
                confidence=0.88,
            )
        if not active_patient and not active_document_name and not active_image and not context.get("active_attachment_id"):
            return with_intent({**base, "reason": "no_active_context"})
        if not self._is_contextual_followup(normalized):
            return with_intent(base)

        active_attachment_id = str(context.get("active_attachment_id") or "")
        last_attachment_ids = context.get("last_attachment_ids") if isinstance(context.get("last_attachment_ids"), list) else []
        if active_attachment_id and self._mentions_previous_attachment(lowered):
            attachment_name = context.get("active_attachment_name") or "the previously uploaded attachment"
            rewritten = f"According to the previously uploaded attachment {attachment_name}, {normalized}"
            return with_intent(
                {
                    **base,
                    "resolved_query": rewritten,
                    "rewritten_query": rewritten,
                    "used_context": True,
                    "is_followup": True,
                    "reason": "contextual_attachment_followup",
                    "confidence": 0.9,
                    "active_attachment_ids": last_attachment_ids or [active_attachment_id],
                    "selected_attachment": active_attachment_id,
                },
                route_hint="retrieve" if context.get("active_document_id") else "imaging",
                confidence=0.9,
            )

        subject = active_patient or "the previously discussed patient"
        report_phrase = f"the uploaded report for {subject}" if active_patient else "the previously discussed uploaded report"
        route_hint = "retrieve"
        if self._looks_like_contextual_comparison(lowered):
            rewritten = f"Compare {report_phrase} with the previous active report."
            route_hint = "compare"
        elif "chest pain" in lowered:
            rewritten = f"Does {report_phrase} mention chest pain?"
        elif "fatigue" in lowered and any(term in lowered for term in ["cause", "causes", "why", "him", "her", "patient"]):
            rewritten = f"According to {report_phrase}, what explains the unexplained fatigue?"
        elif "test performed" in lowered or ("why" in lowered and "test" in lowered):
            rewritten = f"According to {report_phrase}, why was the test performed?"
        elif "what happened next" in lowered or "next" == lowered.strip(" ?!."):
            rewritten = f"According to {report_phrase}, what happened next or what follow-up/events are documented?"
        elif "earlier" in lowered or "asking about the patient" in lowered or "same patient" in lowered:
            rewritten = f"Continue discussing {report_phrase}."
        else:
            rewritten = f"According to {report_phrase}, {normalized}"

        return with_intent(
            {
                **base,
                "resolved_query": rewritten,
                "rewritten_query": rewritten,
                "used_context": True,
                "is_followup": True,
                "reason": "contextual_followup",
                "confidence": 0.85,
            },
            route_hint=route_hint,
        )

    def _rewrite_current_intent(
        self,
        *,
        original_query: str,
        resolved_query: str,
        route_hint: str,
        confidence: float,
        used_context: bool,
        session_state: dict[str, Any],
        documents: list[Any],
        images: list[Any],
    ) -> dict[str, Any]:
        lowered = original_query.lower()
        constraints = self._extract_constraints(original_query)
        inferred_route = route_hint or self._infer_route_hint(lowered, used_context, session_state, documents, images)
        inferred_confidence = max(confidence, self._route_hint_confidence(inferred_route, lowered, used_context))
        context_target = self._context_target(session_state)
        if inferred_route == "compare":
            current_intent = "Compare the current clinical report with the previous or selected report."
        elif inferred_route == "imaging":
            current_intent = "Answer the image-grounded clinical question using uploaded image/report evidence."
        elif inferred_route == "retrieve":
            current_intent = f"Answer using uploaded patient or document evidence: {resolved_query}"
        elif inferred_route == "medical_education":
            current_intent = f"Answer as general medical education: {original_query}"
        elif inferred_route == "direct":
            current_intent = f"Answer a CareMind product/help question: {original_query}"
        else:
            current_intent = f"Clarify or route the latest user goal: {resolved_query}"
        if constraints:
            current_intent = f"{current_intent} Constraints: {', '.join(constraints)}."
        if context_target and used_context:
            current_intent = f"{current_intent} Context target: {context_target}."
        return {
            "style": "recap_deterministic",
            "original_query": original_query,
            "resolved_query": resolved_query,
            "current_intent": current_intent,
            "route_hint": inferred_route,
            "confidence": round(min(inferred_confidence, 0.99), 2),
            "constraints": constraints,
            "used_context": used_context,
            "context_target": context_target,
        }

    def _session_state(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "current_patient": context.get("active_patient") or "",
            "current_document": context.get("active_document_name") or "",
            "current_image": context.get("active_image_name") or context.get("active_image_id") or "",
            "current_attachment": context.get("active_attachment_name") or context.get("active_attachment_id") or "",
            "current_report": context.get("active_report") or context.get("active_document_name") or "",
            "current_topic": context.get("current_topic") or "",
            "conversation_focus": context.get("conversation_focus") or "",
            "last_route": context.get("last_route") or "",
            "last_tool": context.get("last_tool") or "",
            "last_retrieved_chunks": context.get("last_retrieval_result") or [],
            "last_summary": context.get("last_answer_preview") or "",
            "last_citations": context.get("last_citations") or [],
            "conversation_entities": context.get("conversation_entities") or {},
        }

    def _conversation_entities(
        self,
        context: dict[str, Any],
        message: str,
        documents: list[Any],
        images: list[Any],
    ) -> dict[str, list[dict[str, str]]]:
        existing = context.get("conversation_entities") if isinstance(context.get("conversation_entities"), dict) else {}
        entities: dict[str, list[dict[str, str]]] = {
            "patient": list(existing.get("patient", [])),
            "report": list(existing.get("report", [])),
            "image": list(existing.get("image", [])),
            "disease": list(existing.get("disease", [])),
            "medication": list(existing.get("medication", [])),
            "doctor": list(existing.get("doctor", [])),
            "hospital": list(existing.get("hospital", [])),
            "lab_test": list(existing.get("lab_test", [])),
            "timeline_event": list(existing.get("timeline_event", [])),
        }
        active_patient = context.get("active_patient")
        if active_patient:
            self._add_entity(entities, "patient", active_patient, context.get("active_document_id", ""))
        extracted_patient = self._extract_patient_name(message)
        if extracted_patient:
            self._add_entity(entities, "patient", extracted_patient, "")
        for document in documents[:8]:
            self._add_entity(
                entities,
                "report",
                getattr(document, "filename", ""),
                getattr(document, "document_id", ""),
            )
        for image in images[:8]:
            self._add_entity(
                entities,
                "image",
                getattr(image, "filename", ""),
                getattr(image, "image_id", ""),
            )
        for label, terms in {
            "disease": ["diabetes", "hypertension", "pneumonia", "anemia", "anaemia", "svt"],
            "medication": ["ibuprofen", "metformin", "insulin", "antibiotic", "aspirin"],
            "lab_test": ["hemoglobin", "iron", "troponin", "creatinine", "glucose"],
            "timeline_event": ["admission", "discharge", "follow-up", "episode", "event"],
        }.items():
            for term in terms:
                if term in message.lower():
                    self._add_entity(entities, label, term, "")
        return entities

    def _add_entity(self, entities: dict[str, list[dict[str, str]]], kind: str, name: str, entity_id: str) -> None:
        if not name:
            return
        bucket = entities.setdefault(kind, [])
        normalized = name.lower()
        if any(item.get("name", "").lower() == normalized for item in bucket):
            return
        bucket.append({"name": name, "id": entity_id or ""})

    def _rewrite_query(self, message: str) -> str:
        return " ".join(message.strip().split())

    def _extract_patient_name(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text)
        patterns = [
            r"\b(Mr|Mrs|Ms|Miss)\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5})\b",
            r"\bpatient\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b",
        ]
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                if match.lastindex and match.lastindex >= 2:
                    cleaned = f"{match.group(1)}. {match.group(2)}".strip()
                else:
                    cleaned = match.group(1).strip()
                cleaned = re.sub(r"\b(ID|Age|Gender|MRD|Reason|Doctor|Centre|Study|Duration)\b.*$", "", cleaned).strip()
                if len(cleaned.split()) >= 2:
                    candidates.append(cleaned)
        return max(candidates, key=lambda candidate: (len(candidate.split()), len(candidate))) if candidates else ""

    def _is_contextual_followup(self, message: str) -> bool:
        lowered = message.lower().strip()
        contextual_terms = [
            " he ",
            " him",
            " his ",
            " she ",
            " her ",
            " the patient",
            "this report",
            "that report",
            "earlier",
            "same patient",
            "those findings",
            "these findings",
            " it ",
            " this ",
            " that ",
            " last time",
            " previous one",
        ]
        padded = f" {lowered} "
        if any(term in padded for term in contextual_terms):
            return True
        if lowered in {"why chest pain?", "why chest pain", "chest pain?", "chest pain", "what happened next?", "what happened next"}:
            return True
        if lowered.startswith(("why ", "did he ", "did she ", "was he ", "was she ", "what happened next")):
            return True
        clinical_followup_terms = [
            "fatigue",
            "test performed",
            "reason for test",
            "chest pain",
            "symptom",
            "finding",
            "worse",
            "better",
            "changed",
        ]
        return len(lowered.split()) <= 6 and any(term in lowered for term in clinical_followup_terms)

    def _has_general_education_context(self, context: dict[str, Any]) -> bool:
        return (
            context.get("conversation_focus") == "general_medical_education"
            and bool(context.get("current_topic"))
            and not context.get("active_attachment_id")
        )

    def _is_general_education_followup(self, message: str) -> bool:
        if any(term in message for term in ["report", "document", "uploaded", "patient", "my results", "my report"]):
            return False
        lowered = message.lower().strip(" ?!.")
        if lowered in {
            "symptoms",
            "causes",
            "treatment",
            "treatments",
            "cure",
            "prevention",
            "risk factors",
            "diagnosis",
            "tests",
            "tablets",
            "supplements",
            "medicine",
            "medicines",
        }:
            return True
        padded = f" {lowered} "
        if any(term in padded for term in [" it ", " that ", " this "]):
            return True
        followup_terms = [
            "cure for",
            "treatment for",
            "manage it",
            "manage that",
            "prevent it",
            "prevent that",
            "otc",
            "over the counter",
            "tablet",
            "tablets",
            "capsule",
            "capsules",
            "supplement",
            "supplements",
            "medicine",
            "medicines",
        ]
        if any(term in lowered for term in followup_terms):
            return True
        return lowered.startswith(("how is it", "can it", "does it", "what causes it", "what are its", "what symptoms does it"))

    def _looks_like_contextual_comparison(self, message: str) -> bool:
        return any(term in message for term in ["compare", "changed", "difference", "trend", "worse", "better", "previous", "last time"])

    def _infer_route_hint(
        self,
        message: str,
        used_context: bool,
        session_state: dict[str, Any],
        documents: list[Any],
        images: list[Any],
    ) -> str:
        if any(term in message for term in ["what is caremind", "who are you", "what can you do", "how do you work"]):
            return "direct"
        if self._explicitly_changes_to_general_topic(message) or self._asks_general_not_patient(message):
            return "medical_education"
        if self._looks_like_contextual_comparison(message):
            if used_context or len(documents) >= 2 or session_state.get("current_document"):
                return "compare"
        if any(term in message for term in ["x-ray", "xray", "cxr", "image", "scan"]):
            if images or session_state.get("current_image") or "x-ray" in message or "xray" in message:
                return "imaging"
        if used_context or any(term in message for term in ["report", "document", "uploaded", "patient", "my results", "this result"]):
            return "retrieve"
        if any(term in message for term in ["what is", "explain", "symptoms", "causes", "nursing", "protocol"]):
            return "medical_education"
        return "clarify"

    def _mentions_previous_attachment(self, message: str) -> bool:
        return any(
            phrase in message
            for phrase in [
                " it",
                "this",
                "that",
                "attachment",
                "uploaded image",
                "uploaded file",
                "image i uploaded",
                "file i uploaded",
                "same image",
                "same file",
            ]
        )

    def _route_hint_confidence(self, route_hint: str, message: str, used_context: bool) -> float:
        if route_hint in {"direct", "medical_education", "imaging"}:
            return 0.86
        if route_hint == "compare":
            return 0.88 if used_context else 0.78
        if route_hint == "retrieve":
            return 0.84 if used_context else 0.74
        return 0.45

    def _extract_constraints(self, message: str) -> list[str]:
        constraints = []
        patterns = [
            r"\bonly\s+([^,.?]+)",
            r"\bfocus(?:ing)?\s+on\s+([^,.?]+)",
            r"\bjust\s+([^,.?]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                value = self._rewrite_query(match.group(1))
                if value and value.lower() not in {item.lower() for item in constraints}:
                    constraints.append(value)
        return constraints[:3]

    def _context_target(self, session_state: dict[str, Any]) -> str:
        if session_state.get("current_patient") and session_state.get("current_document"):
            return f"{session_state['current_patient']} / {session_state['current_document']}"
        return (
            session_state.get("current_patient")
            or session_state.get("current_document")
            or session_state.get("current_attachment")
            or session_state.get("current_image")
            or ""
        )

    def _explicitly_changes_to_general_topic(self, message: str) -> bool:
        return explicitly_general_topic(message) or "different question" in message

    def _asks_general_not_patient(self, message: str) -> bool:
        if "different question" in message:
            return True
        return asks_general_medical_not_document(message)
