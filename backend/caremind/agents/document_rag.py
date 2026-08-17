import time
import re
import logging
import hashlib
from typing import Any

from ..llm import LLMClient
from ..memory import ConversationMemory
from ..metrics import metrics
from ..schemas import ChatRequest, RetrievedChunk
from ..tools import DocumentTools, citations_from_chunks

NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER = (
    "No relevant evidence was found in the uploaded documents for this question. "
    "You can upload a more relevant document, refer to a specific report section, "
    "or ask it as a general medical education question."
)
NO_REPORT_INSTRUCTIONS_ANSWER = (
    "No instructions were found in the retrieved sections. CareMind cannot determine what this specific patient "
    "must do from the report evidence alone."
)
logger = logging.getLogger(__name__)


class DocumentRAGAgent:
    """Answers questions from uploaded workspace documents using vector retrieval."""

    def __init__(self, tools: DocumentTools, llm: LLMClient, memory: ConversationMemory):
        self.tools = tools
        self.llm = llm
        self.memory = memory

    def run(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        updates = self.prepare(request, tool_calls, trace_steps)
        if "answer" in updates:
            return updates
        chunks = updates.pop("chunks")
        history = updates.pop("history")
        return {
            **updates,
            "answer": self.llm.answer(
                question=request.message,
                chunks=chunks,
                history=history,
                response_language=request.response_language,
            ),
        }

    def prepare(self, request: ChatRequest, tool_calls: list[str], trace_steps: list[dict]) -> dict[str, Any]:
        started = time.perf_counter()
        history = self._load_history(request)
        retrieval_query = self._retrieval_query(request.message, history)
        attachment_chunks = self._attachment_chunks(request)
        if attachment_chunks:
            trace_step = {
                "node": "document_rag_agent",
                "tool": "message_attachment_lookup",
                "top_k": request.top_k,
                "standalone_query": request.message,
                "attachment_ids": request.attachment_ids,
                "retrieved_count": len(attachment_chunks),
                "filtered_count": len(attachment_chunks),
                "search_bypassed": True,
                "structured_evidence_reused": True,
                "current_message_attachment": True,
                "selection_reason": "current_message_attachments",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return {
                "chunks": attachment_chunks,
                "history": history,
                "citations": citations_from_chunks(attachment_chunks),
                "retrieved_chunks": [self._chunk_trace(chunk) for chunk in attachment_chunks],
                "retrieval_reused": True,
                "evidence_need": self._evidence_need(request.message),
                "tool_calls": [*tool_calls, "message_attachment_lookup"],
                "trace_steps": [
                    *trace_steps,
                    trace_step,
                ],
            }
        if self._is_report_guidance_query(request.message):
            return self._prepare_report_guidance(request, history, tool_calls, trace_steps, started)
        reuse_chunks, reuse_diagnostics = self._reusable_chunks(request)
        if reuse_chunks:
            metrics.record_retrieval_reuse("hit", reuse_diagnostics["reason"])
            trace_step = {
                "node": "document_rag_agent",
                "tool": "retrieval_reuse",
                "top_k": request.top_k,
                "standalone_query": request.message,
                "retrieval_query": retrieval_query,
                "history_count": len(history),
                "retrieval_reused": True,
                "retrieval_reuse_confidence": reuse_diagnostics["confidence"],
                "retrieval_reuse_reason": reuse_diagnostics["reason"],
                "retrieved_count": len(reuse_chunks),
                "kept_count": len(reuse_chunks),
                "filtered_count": 0,
                "search_bypassed": True,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            logger.info(
                "document_retrieval.reuse.done workspace_id=%s chunks=%s confidence=%.2f reason=%s",
                request.workspace_id,
                len(reuse_chunks),
                reuse_diagnostics["confidence"],
                reuse_diagnostics["reason"],
            )
            return {
                "chunks": reuse_chunks,
                "history": history,
                "citations": citations_from_chunks(reuse_chunks),
                "retrieved_chunks": [self._chunk_trace(chunk) for chunk in reuse_chunks],
                "retrieval_reused": True,
                "evidence_need": self._evidence_need(request.message),
                "tool_calls": [*tool_calls, "retrieval_reuse"],
                "trace_steps": [
                    *trace_steps,
                    trace_step,
                ],
            }
        active_match_chunks, active_match_diagnostics = self._active_document_matching_chunks(request)
        if active_match_chunks:
            trace_step = {
                "node": "document_rag_agent",
                "tool": "active_document_lookup",
                "top_k": request.top_k,
                "standalone_query": request.message,
                "retrieval_query": request.message,
                "history_count": len(history),
                "active_document_count": active_match_diagnostics["active_document_count"],
                "candidate_chunk_count": active_match_diagnostics["candidate_chunk_count"],
                "matched_chunk_count": len(active_match_chunks),
                "search_bypassed": True,
                "selection_reason": active_match_diagnostics["reason"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return {
                "chunks": active_match_chunks,
                "history": history,
                "citations": citations_from_chunks(active_match_chunks),
                "retrieved_chunks": [self._chunk_trace(chunk) for chunk in active_match_chunks],
                "retrieval_reused": True,
                "evidence_need": self._evidence_need(request.message),
                "tool_calls": [*tool_calls, "active_document_lookup"],
                "trace_steps": [
                    *trace_steps,
                    trace_step,
                ],
            }
        metrics.record_retrieval_reuse("miss", reuse_diagnostics["reason"])
        raw_chunks = self.tools.document_search(
            retrieval_query,
            workspace_id=request.workspace_id,
            top_k=request.top_k,
        )
        chunks, filter_diagnostics = self._filter_and_deduplicate(raw_chunks)
        no_relevant_evidence = not chunks
        trace_step = {
            "node": "document_rag_agent",
            "tool": "document_search",
            "top_k": request.top_k,
            "standalone_query": request.message,
            "retrieval_query": retrieval_query,
            "history_count": len(history),
            **filter_diagnostics,
            "no_relevant_document_evidence": no_relevant_evidence,
            "search_diagnostics": getattr(self.tools, "last_document_search_diagnostics", {}),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        logger.info(
            "document_retrieval.filter.done workspace_id=%s returned=%s kept=%s filtered=%s below_threshold=%s duplicate=%s "
            "min_similarity=%.4f highest_similarity=%s lowest_similarity=%s",
            request.workspace_id,
            filter_diagnostics["returned_count"],
            filter_diagnostics["kept_count"],
            filter_diagnostics["filtered_count"],
            filter_diagnostics["below_threshold_count"],
            filter_diagnostics["filter_reasons"]["duplicate"],
            filter_diagnostics["min_similarity"],
            trace_step["highest_similarity"],
            trace_step["lowest_similarity"],
        )
        if no_relevant_evidence:
            return {
                "answer": NO_RELEVANT_DOCUMENT_EVIDENCE_ANSWER,
                "citations": [],
                "retrieved_chunks": [],
                "no_relevant_document_evidence": True,
                "evidence_need": self._evidence_need(request.message),
                "tool_calls": [*tool_calls, "document_search"],
                "trace_steps": [
                    *trace_steps,
                    trace_step,
                ],
            }

        return {
            "chunks": chunks,
            "history": history,
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "evidence_need": self._evidence_need(request.message),
            "tool_calls": [*tool_calls, "document_search"],
            "trace_steps": [
                *trace_steps,
                trace_step,
            ],
        }

    def _prepare_report_guidance(
        self,
        request: ChatRequest,
        history: list,
        tool_calls: list[str],
        trace_steps: list[dict],
        started: float,
    ) -> dict[str, Any]:
        report_chunks, coverage = self._active_document_instruction_chunks(request)
        education_chunks = self._general_next_step_chunks(request) if not report_chunks else []
        chunks = [*report_chunks, *education_chunks]
        tool_names = [*tool_calls, "document_instruction_search"]
        if education_chunks:
            tool_names.append("medical_education_search")
        if not chunks:
            trace_step = self._report_guidance_trace(
                request=request,
                report_chunks=report_chunks,
                education_chunks=education_chunks,
                coverage=coverage,
                started=started,
            )
            return {
                "answer": NO_REPORT_INSTRUCTIONS_ANSWER,
                "citations": [],
                "retrieved_chunks": [],
                "no_relevant_document_evidence": True,
                "evidence_need": "report_guidance",
                "tool_calls": tool_names,
                "trace_steps": [*trace_steps, trace_step],
            }
        trace_step = self._report_guidance_trace(
            request=request,
            report_chunks=report_chunks,
            education_chunks=education_chunks,
            coverage=coverage,
            started=started,
        )
        return {
            "chunks": chunks,
            "history": history,
            "citations": citations_from_chunks(chunks),
            "retrieved_chunks": [self._chunk_trace(chunk) for chunk in chunks],
            "tool_calls": tool_names,
            "evidence_need": "report_guidance",
            "report_guidance_workflow": True,
            "trace_steps": [*trace_steps, trace_step],
        }

    def _load_history(self, request: ChatRequest) -> list:
        try:
            history = self.memory.load(request.session_id, request.workspace_id)
        except Exception:
            return []
        return history if isinstance(history, list) else []

    def _attachment_chunks(self, request: ChatRequest) -> list[RetrievedChunk]:
        raw_attachment_ids = getattr(request, "attachment_ids", []) or []
        attachment_ids = raw_attachment_ids if isinstance(raw_attachment_ids, list) else []
        if not attachment_ids:
            try:
                context = self.memory.context_get(request.session_id, request.workspace_id)
            except Exception:
                context = {}
            if isinstance(context, dict) and "previously uploaded attachment" in request.message.lower():
                attachment_ids = [
                    str(item)
                    for item in (context.get("last_attachment_ids") or [context.get("active_attachment_id")])
                    if item
                ]
        if not attachment_ids:
            return []
        lookup = getattr(self.tools, "attachment_evidence_chunks", None)
        if not callable(lookup):
            return []
        return lookup(
            attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
            top_k=max(request.top_k, len(request.attachment_ids)),
        )

    def _reusable_chunks(self, request: ChatRequest) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        if self._is_report_guidance_query(request.message):
            return [], {"confidence": 0.0, "reason": "changed_evidence_need_report_guidance"}
        try:
            context = self.memory.context_get(request.session_id, request.workspace_id)
        except Exception:
            context = {}
        if not isinstance(context, dict) or not context:
            return [], {"confidence": 0.0, "reason": "no_session_state"}
        candidates = context.get("last_retrieved_chunks") or context.get("last_retrieval_result") or []
        if not isinstance(candidates, list) or not candidates:
            return [], {"confidence": 0.0, "reason": "no_previous_chunks"}
        current_need = self._evidence_need(request.message)
        previous_need = str(context.get("last_evidence_need") or "")
        if previous_need and previous_need != current_need and not self._is_vague_contextual_followup(request.message):
            return [], {"confidence": 0.0, "reason": "changed_evidence_need"}
        active_document_id = str(context.get("active_document_id") or "")
        chunks: list[RetrievedChunk] = []
        for item in candidates[: request.top_k]:
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id") or "")
            if active_document_id and document_id and document_id != active_document_id:
                continue
            text = str(item.get("text") or item.get("preview") or item.get("quote") or "").strip()
            if not text:
                continue
            score = float(item.get("score") or 0.0)
            if score < self.tools.embeddings.settings.min_retrieval_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(item.get("chunk_id") or f"{document_id}:cached"),
                    document_id=document_id or active_document_id,
                    document_name=str(item.get("document_name") or context.get("active_document_name") or "previously retrieved document"),
                    text=text,
                    page=item.get("page"),
                    score=score,
                )
            )
        if not chunks:
            return [], {"confidence": 0.0, "reason": "no_active_document_match"}
        confidence, reason = self._reuse_confidence(request.message, chunks, context)
        if confidence < 0.62:
            return [], {"confidence": confidence, "reason": reason}
        return chunks, {"confidence": confidence, "reason": reason}

    def _reuse_confidence(self, query: str, chunks: list[RetrievedChunk], context: dict[str, Any]) -> tuple[float, str]:
        if context.get("active_attachment_id") and self._looks_like_attachment_followup(query):
            return 0.78, "active_attachment_followup_chunks"
        if self._is_vague_contextual_followup(query):
            return 0.78, "vague_contextual_followup_previous_chunks"
        query_terms = self._distinctive_terms(query)
        if not query_terms:
            return 0.0, "no_distinctive_query_terms"
        source_text = " ".join([chunk.document_name + " " + chunk.text for chunk in chunks]).lower()
        source_terms = set(re.findall(r"[a-z0-9]+", source_text))
        overlap = query_terms & source_terms
        active_patient = str(context.get("active_patient") or "").lower()
        patient_tokens = {term for term in re.findall(r"[a-z0-9]+", active_patient) if len(term) >= 3}
        patient_overlap = bool(patient_tokens & source_terms)
        if self._is_broad_document_summary_query(query) and patient_overlap:
            return 0.75, "active_patient_summary_chunks"
        if len(overlap) >= max(1, min(3, len(query_terms) // 2)):
            return min(0.95, 0.55 + (len(overlap) / max(len(query_terms), 1))), "query_terms_supported_by_previous_chunks"
        if patient_overlap and len(query_terms) <= 6:
            return 0.65, "short_followup_with_active_patient_chunks"
        return 0.25, "previous_chunks_do_not_cover_query"

    def _looks_like_attachment_followup(self, query: str) -> bool:
        lowered = f" {query.lower()} "
        return any(
            phrase in lowered
            for phrase in [
                " it ",
                " this ",
                " that ",
                "the image",
                "the file",
                "the attachment",
                "uploaded image",
                "uploaded file",
            ]
        )

    def _is_vague_contextual_followup(self, query: str) -> bool:
        lowered = query.lower()
        return any(
            phrase in lowered
            for phrase in [
                "what does that mean",
                "what does this mean",
                "what does it mean",
                "is it serious",
                "is this serious",
                "explain that",
                "explain this",
                "tell me more",
            ]
        )

    def _active_document_matching_chunks(self, request: ChatRequest) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        if not self._is_document_mention_query(request.message):
            return [], {"active_document_count": 0, "candidate_chunk_count": 0, "reason": "not_document_mention_query"}
        document_ids = self._active_document_ids(request)
        if not document_ids:
            return [], {"active_document_count": 0, "candidate_chunk_count": 0, "reason": "no_active_document"}
        candidates: list[RetrievedChunk] = []
        for document_id in document_ids[:2]:
            try:
                candidates.extend(self.tools.store.get_document_chunks(document_id))
            except Exception:
                continue
        candidates = self._deduplicate_chunks(candidates)
        query_terms = self._distinctive_terms(request.message) - {
            "according",
            "document",
            "documented",
            "mention",
            "mentioned",
            "patient",
            "report",
            "uploaded",
        }
        matches = [
            chunk.model_copy(update={"score": chunk.score if chunk.score is not None else 1.0})
            for chunk in candidates
            if query_terms and query_terms & set(re.findall(r"[a-z0-9]+", f"{chunk.document_name} {chunk.text}".lower()))
        ]
        return matches[: request.top_k], {
            "active_document_count": len(document_ids),
            "candidate_chunk_count": len(candidates),
            "reason": "active_document_term_match" if matches else "no_active_document_term_match",
        }

    def _is_document_mention_query(self, query: str) -> bool:
        lowered = query.lower()
        document_scoped = any(
            term in lowered
            for term in [
                "my report",
                "the report",
                "uploaded report",
                "uploaded document",
                "this report",
                "that report",
                "patient",
                "according to",
            ]
        )
        mention_scoped = any(term in lowered for term in ["mention", "mentioned", "say about", "documented", "show"])
        return document_scoped and mention_scoped

    def _retrieval_query(self, query: str, history: list) -> str:
        return query

    def _chunk_trace(self, chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": self._source_label(chunk),
            "page": chunk.page,
            "score": chunk.score,
            "source": chunk.metadata.get("source", "uploaded_document") if chunk.metadata else "uploaded_document",
        }

    def _source_label(self, chunk: RetrievedChunk) -> str:
        source = (chunk.metadata or {}).get("source", "")
        if source == "medical_education":
            return chunk.document_name
        if source == "medlineplus":
            return chunk.document_name
        return "Uploaded document"

    def _filter_and_deduplicate(self, raw_chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        minimum_similarity = self.tools.embeddings.settings.min_retrieval_similarity
        kept: list[RetrievedChunk] = []
        reasons = {
            "below_similarity_threshold": 0,
            "duplicate": 0,
            "wrong_index_version": 0,
            "wrong_corpus": 0,
            "topic_mismatch": 0,
            "unauthorized_workspace": 0,
            "invalid_metadata": 0,
        }
        seen: set[str] = set()
        for chunk in raw_chunks:
            score = chunk.score if chunk.score is not None else 0.0
            if score < minimum_similarity:
                reasons["below_similarity_threshold"] += 1
                continue
            key = self._dedupe_key(chunk)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            kept.append(chunk)
        filtered_count = len(raw_chunks) - len(kept)
        diagnostics = {
            "retrieved_count": len(raw_chunks),
            "returned_count": len(raw_chunks),
            "kept_count": len(kept),
            "filtered_count": filtered_count,
            "number_filtered": filtered_count,
            "below_threshold_count": reasons["below_similarity_threshold"],
            "min_similarity": minimum_similarity,
            "highest_similarity": max((chunk.score or 0.0 for chunk in raw_chunks), default=None),
            "lowest_similarity": min((chunk.score or 0.0 for chunk in raw_chunks), default=None),
            "lowest_kept_similarity": min((chunk.score or 0.0 for chunk in kept), default=None),
            "filter_reasons": reasons,
            "counter_invariant_ok": len(kept) + filtered_count == len(raw_chunks),
        }
        return kept, diagnostics

    def _dedupe_key(self, chunk: RetrievedChunk) -> str:
        normalized = re.sub(r"\s+", " ", chunk.text.lower()).strip()
        normalized = re.sub(r"\b\d+\b", "#", normalized)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{chunk.document_id}:{chunk.page}:{digest}"

    def _is_report_guidance_query(self, query: str) -> bool:
        lowered = query.lower()
        if re.search(r"\bfollow[\s-]?up\s+(?:report|document|file|lab|labs|result|results)\b", lowered):
            return False
        asks_action = any(
            phrase in lowered
            for phrase in [
                "what should",
                "what must",
                "what to do",
                "next step",
                "next steps",
                "follow up",
                "follow-up",
                "recommend",
                "recommendation",
                "instruction",
                "advice",
                "plan",
                "treatment",
            ]
        )
        report_scoped = any(term in lowered for term in ["report", "document", "patient", "according to", "uploaded"])
        return asks_action and report_scoped

    def _evidence_need(self, query: str) -> str:
        if self._is_report_guidance_query(query):
            return "report_guidance"
        if self._is_broad_document_summary_query(query):
            return "key_findings"
        lowered = query.lower()
        if any(term in lowered for term in ["compare", "difference", "changed"]):
            return "comparison"
        if any(term in lowered for term in ["symptom", "fatigue", "pain", "finding", "result", "value", "diagnosis"]):
            return "specific_report_fact"
        return "document_qa"

    def _active_document_instruction_chunks(self, request: ChatRequest) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        document_ids = self._active_document_ids(request)
        all_chunks: list[RetrievedChunk] = []
        for document_id in document_ids:
            try:
                all_chunks.extend(self.tools.store.get_document_chunks(document_id))
            except Exception:
                continue
        candidates = self._deduplicate_chunks(all_chunks)
        instruction_chunks = [chunk for chunk in candidates if self._contains_instruction_signal(chunk.text)]
        ranked = sorted(
            instruction_chunks,
            key=lambda chunk: (self._instruction_signal_score(chunk.text), chunk.score or 0.0),
            reverse=True,
        )
        selected = [
            chunk.model_copy(
                update={
                    "score": chunk.score if chunk.score is not None else 1.0,
                    "metadata": {**(chunk.metadata or {}), "source": "uploaded_document", "evidence_need": "report_guidance"},
                }
            )
            for chunk in ranked[: request.top_k]
        ]
        coverage = {
            "active_document_count": len(document_ids),
            "document_chunk_count": len(all_chunks),
            "deduplicated_document_chunk_count": len(candidates),
            "instruction_candidate_count": len(instruction_chunks),
            "selected_instruction_count": len(selected),
            "complete_active_document_scanned": bool(document_ids and all_chunks),
        }
        return selected, coverage

    def _active_document_ids(self, request: ChatRequest) -> list[str]:
        try:
            context = self.memory.context_get(request.session_id, request.workspace_id)
        except Exception:
            context = {}
        if not isinstance(context, dict):
            return []
        ids = context.get("active_document_ids") or []
        if isinstance(ids, list) and ids:
            return [str(item) for item in ids if item]
        document_id = context.get("active_document_id")
        return [str(document_id)] if document_id else []

    def _contains_instruction_signal(self, text: str) -> bool:
        return self._instruction_signal_score(text) > 0

    def _instruction_signal_score(self, text: str) -> int:
        lowered = text.lower()
        patterns = [
            r"\brecommend(?:ed|ation|ations)?\b",
            r"\bfollow[\s-]?up\b",
            r"\bplan\b",
            r"\binstruction(?:s)?\b",
            r"\badvice\b",
            r"\badvise(?:d)?\b",
            r"\bconclusion\b",
            r"\bimpression\b",
            r"\bconsult(?:ation)?\b",
            r"\breview\b",
            r"\btreatment\b",
            r"\brefer(?:ral)?\b",
            r"\breturn\b",
        ]
        return sum(1 for pattern in patterns if re.search(pattern, lowered))

    def _general_next_step_chunks(self, request: ChatRequest) -> list[RetrievedChunk]:
        queries = [
            "general medical follow-up after receiving a medical report clinician review",
            "urgent warning signs chest pain fainting difficulty breathing medical attention",
        ]
        chunks: list[RetrievedChunk] = []
        for query in queries:
            try:
                chunks.extend(self.tools.medical_education_search(query, top_k=2))
            except Exception:
                continue
        filtered = [
            chunk.model_copy(update={"metadata": {**(chunk.metadata or {}), "source": "medical_education", "evidence_need": "general_next_steps"}})
            for chunk in self._deduplicate_chunks(chunks)
            if (chunk.score or 0.0) >= self.tools.embeddings.settings.min_retrieval_similarity
        ]
        return filtered[: max(1, min(request.top_k, 4))]

    def _deduplicate_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        result: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = self._dedupe_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            result.append(chunk)
        return result

    def _report_guidance_trace(
        self,
        *,
        request: ChatRequest,
        report_chunks: list[RetrievedChunk],
        education_chunks: list[RetrievedChunk],
        coverage: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        return {
            "node": "document_rag_agent",
            "tool": "document_instruction_search+medical_education_search" if education_chunks else "document_instruction_search",
            "top_k": request.top_k,
            "standalone_query": request.message,
            "retrieval_query": request.message,
            "history_count": 0,
            "retrieval_reused": False,
            "report_guidance_workflow": True,
            "report_instruction_count": len(report_chunks),
            "education_fallback_used": bool(education_chunks),
            "education_evidence_count": len(education_chunks),
            "retrieved_count": len(report_chunks) + len(education_chunks),
            "kept_count": len(report_chunks) + len(education_chunks),
            "filtered_count": 0,
            "coverage": coverage,
            "limited_coverage": not coverage.get("complete_active_document_scanned"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _has_strong_lexical_match(self, query: str, chunk: RetrievedChunk) -> bool:
        query_terms = self._distinctive_terms(query)
        if len(query_terms) < 2:
            return False
        source_text = f"{chunk.document_name} {chunk.text}".lower()
        source_terms = set(re.findall(r"[a-z0-9]+", source_text))
        matched_terms = query_terms & source_terms
        return len(matched_terms) >= 2 and len(matched_terms) >= max(2, len(query_terms) - 1)

    def _is_broad_document_summary_query(self, query: str) -> bool:
        lowered = query.lower()
        if any(
            phrase in lowered
            for phrase in [
                "key finding",
                "key findings",
                "main finding",
                "main findings",
                "important finding",
                "important findings",
                "report finding",
                "report findings",
                "findings of the report",
                "findings in the report",
                "findings from the report",
                "summarize",
                "summary",
                "clinical summary",
                "patient summary",
                "health condition",
                "condition of the patient",
                "patient condition",
            ]
        ):
            return True
        return bool(
            re.search(r"\bwhat\s+(?:are|r|is|'?s)?\s*(?:the\s+)?(?:main\s+|key\s+)?findings?\b", lowered)
            or re.search(r"\bfindings?\s+(?:of|in|from)\s+(?:the\s+)?(?:report|document|file)\b", lowered)
        )

    def _distinctive_terms(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "attached",
            "based",
            "document",
            "file",
            "find",
            "for",
            "from",
            "give",
            "me",
            "patient",
            "please",
            "report",
            "results",
            "show",
            "tell",
            "the",
            "this",
            "uploaded",
            "what",
        }
        terms = set()
        for term in re.findall(r"[a-z0-9]+", text.lower()):
            if len(term) < 3 or term in stopwords:
                continue
            terms.add(term)
        return terms
