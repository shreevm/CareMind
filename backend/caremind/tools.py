from collections import Counter
from difflib import SequenceMatcher
import logging
import math
import re

from .education import MedicalEducationStore
from .embeddings import EmbeddingClient
from .external_literature import ExternalLiteratureSearch
from .observability import chunk_outputs, trace_block, trace_text
from .schemas import Citation, CompareResponse, RetrievedChunk
from .store import SQLiteStore
from .vectorstore import VectorStore
from .vectorstore import cosine_similarity


logger = logging.getLogger(__name__)


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations = []
    for chunk in chunks:
        quote = clean_evidence_text(chunk.text).strip().replace("\n", " ")
        citations.append(
            Citation(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                score=chunk.score,
                quote=quote[:360],
            )
        )
    return citations


def clean_evidence_text(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("Â®", "").replace("â€“", "-").replace("\u00ae", "")
    patterns = [
        r"The content in this report is not intended to be a substitute for professional medical advice,\s*diagnosis,\s*or treatment\.",
        r"Always seek the\s+advice of your physician or other qualified health provider with any questions you may have regarding a medical condition\.",
        r"Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition\.",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


class DocumentTools:
    def __init__(self, embeddings: EmbeddingClient, vectorstore: VectorStore, store: SQLiteStore):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.store = store
        self.education = MedicalEducationStore(embeddings)
        self.external_literature = ExternalLiteratureSearch(embeddings.settings)
        self.last_document_search_diagnostics: dict = {}

    def document_search(self, query: str, workspace_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.document_search",
            "retriever",
            {
                "query": trace_text(settings, query),
                "workspace_id": workspace_id,
                "top_k": top_k,
                "vector_backend": settings.vector_backend,
            },
        ) as run:
            query_embedding = self.embeddings.embed_query(query)
            query_norm = math.sqrt(sum(value * value for value in query_embedding))
            embedding_diagnostics = self.store.embedding_diagnostics(workspace_id, limit=5)
            consistency_issues = self._embedding_consistency_issues(query_embedding, embedding_diagnostics)
            known_chunk_diagnostic = self.embedding_similarity_diagnostic(
                query=query,
                workspace_id=workspace_id,
                query_embedding=query_embedding,
            )
            logger.info(
                "document_search.query_embedding provider=%s model=%s configured_model=%s dimension=%s configured_dimension=%s index_version=%s norm=%.6f",
                self.embeddings.last_provider,
                self.embeddings.last_model,
                self.embeddings.configured_model_label(),
                len(query_embedding),
                settings.embedding_dimension,
                settings.embedding_index_version,
                query_norm,
            )
            logger.info(
                "document_search.document_embedding_diagnostics workspace_id=%s current_model=%s diagnostics=%s",
                workspace_id,
                self.embeddings.last_model,
                embedding_diagnostics,
            )
            if consistency_issues:
                logger.warning(
                    "document_search.embedding_consistency_issues workspace_id=%s issues=%s reindex_required=%s",
                    workspace_id,
                    consistency_issues,
                    True,
                )
            logger.info(
                "document_search.known_chunk_similarity workspace_id=%s query=%r diagnostic=%s",
                workspace_id,
                query,
                known_chunk_diagnostic,
            )
            chunks = self.vectorstore.search(
                query_embedding,
                workspace_id=workspace_id,
                top_k=max(top_k * 3, 12),
                embedding_provider=self.embeddings.last_provider,
                embedding_model=self.embeddings.last_model or self.embeddings.configured_model_label(),
                embedding_index_version=settings.embedding_index_version,
            )
            child_results = self._rerank_document_chunks(query, chunks)[:top_k]
            results = self._expand_parent_child_context(child_results)
            scores = [chunk.score or 0.0 for chunk in results]
            self.last_document_search_diagnostics = {
                "query_embedding_provider": self.embeddings.last_provider,
                "query_embedding_model": self.embeddings.last_model,
                "query_embedding_dimension": len(query_embedding),
                "configured_embedding_dimension": settings.embedding_dimension,
                "embedding_index_version": settings.embedding_index_version,
                "query_embedding_norm": round(query_norm, 6),
                "document_embedding_diagnostics": embedding_diagnostics,
                "embedding_consistency_issues": consistency_issues,
                "reindex_required": bool(consistency_issues),
                "known_chunk_similarity": known_chunk_diagnostic,
                "parent_child_expansion": {
                    "enabled": True,
                    "child_count": len(child_results),
                    "expanded_count": len(results),
                },
                "highest_similarity": max(scores, default=None),
                "lowest_similarity": min(scores, default=None),
                "returned_count": len(results),
            }
            run.end(
                {
                    "query_embedding": {
                        "provider": self.embeddings.last_provider,
                        "model": self.embeddings.last_model,
                        "dimension": len(query_embedding),
                        "configured_dimension": settings.embedding_dimension,
                        "index_version": settings.embedding_index_version,
                        "norm": round(query_norm, 6),
                    },
                    "document_embedding_diagnostics": embedding_diagnostics,
                    "embedding_consistency_issues": consistency_issues,
                    "known_chunk_similarity": known_chunk_diagnostic,
                    "parent_child_expansion": {
                        "enabled": True,
                        "child_count": len(child_results),
                        "expanded_count": len(results),
                    },
                    "retrieved_count": len(results),
                    "highest_similarity": max(scores, default=None),
                    "lowest_similarity": min(scores, default=None),
                    "results": chunk_outputs(settings, results),
                }
            )
            return results

    def _expand_parent_child_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        expand = getattr(self.store, "expand_chunks_with_neighbors", None)
        if not callable(expand):
            return chunks
        try:
            return expand(chunks, window=1, max_chars=2400)
        except Exception as exc:
            logger.info("document_search.parent_child_expansion.skipped reason=%s", exc.__class__.__name__)
            return chunks

    def embedding_similarity_diagnostic(
        self,
        *,
        query: str,
        workspace_id: str,
        query_embedding: list[float] | None = None,
        document_name_contains: str = "",
        chunk_text_contains: str = "",
    ) -> dict:
        query_embedding = query_embedding if query_embedding is not None else self.embeddings.embed_query(query)
        rows = self.store.get_local_vector_diagnostics(workspace_id, limit=500)
        selected = self._select_known_chunk_for_embedding_probe(
            query=query,
            rows=rows,
            document_name_contains=document_name_contains,
            chunk_text_contains=chunk_text_contains,
        )
        query_model = self.embeddings.last_model or self.embeddings.configured_model_label()
        query_provider = self.embeddings.last_provider
        query_norm = math.sqrt(sum(value * value for value in query_embedding))
        if selected is None:
            diagnostic = {
                "status": "no_matching_indexed_chunk",
                "query": query,
                "workspace_id": workspace_id,
                "query_embedding_provider": query_provider,
                "query_embedding_model": query_model,
                "query_embedding_dimension": len(query_embedding),
                "query_embedding_norm": round(query_norm, 6),
                "indexed_chunk_count_checked": len(rows),
                "document_name_contains": document_name_contains,
                "chunk_text_contains": chunk_text_contains,
            }
            logger.info("embedding_similarity_diagnostic %s", diagnostic)
            return diagnostic

        stored_vector = selected["embedding"]
        similarity = cosine_similarity(query_embedding, stored_vector)
        stored_model = selected.get("embedding_model") or ""
        stored_provider = selected.get("embedding_provider") or ""
        stored_index_version = selected.get("embedding_index_version") or ""
        issues = []
        if len(query_embedding) != len(stored_vector):
            issues.append("dimension_mismatch")
        if not stored_model:
            issues.append("missing_stored_embedding_model_metadata")
        elif stored_model != query_model:
            issues.append("model_mismatch")
        if stored_provider and query_provider and stored_provider != query_provider:
            issues.append("provider_mismatch")
        if stored_index_version and stored_index_version != self.embeddings.settings.embedding_index_version:
            issues.append("index_version_mismatch")
        if selected["non_zero_count"] == 0:
            issues.append("stored_zero_vector")

        diagnostic = {
            "status": "ok",
            "query": query,
            "workspace_id": workspace_id,
            "query_embedding_provider": query_provider,
            "query_embedding_model": query_model,
            "query_embedding_dimension": len(query_embedding),
            "query_embedding_norm": round(query_norm, 6),
            "chunk_id": selected["chunk_id"],
            "document_id": selected["document_id"],
            "document_name": selected["document_name"],
            "page": selected["page"],
            "position": selected["position"],
            "stored_embedding_provider": stored_provider,
            "stored_embedding_model": stored_model,
            "stored_embedding_dimension": selected.get("embedding_dimension"),
            "stored_embedding_index_version": selected.get("embedding_index_version"),
            "stored_actual_dimension": len(stored_vector),
            "stored_non_zero_count": selected["non_zero_count"],
            "stored_embedding_norm": selected["norm"],
            "direct_local_cosine_similarity": round(similarity, 6),
            "model_matches": bool(stored_model and stored_model == query_model),
            "dimension_matches": len(query_embedding) == len(stored_vector),
            "issues": issues,
        }
        logger.info("embedding_similarity_diagnostic %s", diagnostic)
        return diagnostic

    def medical_education_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.medical_education_search",
            "retriever",
            {"query": trace_text(settings, query), "top_k": top_k},
        ) as run:
            candidates = self.education.search(query, top_k=max(top_k * 3, top_k))
            results, diagnostics = self._filter_retrieval_results(candidates, minimum_similarity=settings.min_retrieval_similarity)
            run.end({"retrieved_count": len(results), "retrieval_filter": diagnostics, "results": chunk_outputs(settings, results)})
            return [
                chunk.model_copy(update={"metadata": {**(chunk.metadata or {}), "source": "medical_education"}})
                for chunk in results[:top_k]
            ]

    def medlineplus_health_topic_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.medlineplus_health_topic_search",
            "tool",
            {"query": trace_text(settings, query), "top_k": top_k, "provider": "medlineplus"},
        ) as run:
            results = self.external_literature.search_medlineplus(query, top_k=top_k)
            run.end({"retrieved_count": len(results), "results": chunk_outputs(settings, results)})
            return results

    def external_literature_search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.external_literature_search",
            "tool",
            {"query": trace_text(settings, query), "top_k": top_k, "provider": "pubmed"},
        ) as run:
            results = self.external_literature.search_pubmed(query, top_k=top_k)
            run.end({"retrieved_count": len(results), "results": chunk_outputs(settings, results)})
            return results

    def external_literature_available(self) -> bool:
        return self.external_literature.enabled

    def medlineplus_available(self) -> bool:
        return self.external_literature.medlineplus_enabled

    def imaging_search(self, query: str, workspace_id: str, top_k: int = 3) -> list[RetrievedChunk]:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.imaging_search",
            "retriever",
            {"query": trace_text(settings, query), "workspace_id": workspace_id, "top_k": top_k},
        ) as run:
            images = self.store.list_images(workspace_id)[:top_k]
            chunks = []
            for image in images:
                report_summary = (image.report_text_summary or "").strip()
                if report_summary:
                    text = (
                        f"Clinical image asset: {image.filename}. Modality: {image.modality}. "
                        f"Paired report or caption for grounding: {report_summary}"
                    )
                else:
                    text = (
                        f"Clinical image asset: {image.filename}. Modality: {image.modality}. "
                        "No paired report or caption was provided, so CareMind cannot ground image findings yet."
                    )
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"image:{image.image_id}",
                        document_id=image.image_id,
                        document_name=image.filename,
                        text=text,
                        score=1.0 if report_summary else 0.0,
                    )
                )
            run.end({"retrieved_count": len(chunks), "results": chunk_outputs(settings, chunks)})
            return chunks

    def attachment_evidence_chunks(
        self,
        attachment_ids: list[str],
        *,
        workspace_id: str,
        conversation_id: str,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        attachments = self.store.require_message_attachments(
            attachment_ids=attachment_ids,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        chunks: list[RetrievedChunk] = []
        for attachment in attachments:
            if attachment.document_id:
                document = self.store.get_document(attachment.document_id)
                if document is not None and document.workspace_id == workspace_id:
                    chunks.extend(self.store.get_document_chunks(attachment.document_id))
                    continue
            if attachment.image_asset_id:
                image = self.store.get_image_asset(attachment.image_asset_id)
                if image is None or image.workspace_id != workspace_id:
                    continue
                if image.ocr_document_id:
                    chunks.extend(self.store.get_document_chunks(image.ocr_document_id))
                    continue
                report_summary = (image.report_text_summary or "").strip()
                text = (
                    f"Clinical image attachment: {image.filename}. Modality: {image.modality}. "
                    f"Paired report or caption for grounding: {report_summary}"
                    if report_summary
                    else (
                        f"Clinical image attachment: {image.filename}. Modality: {image.modality}. "
                        "No extracted document evidence or paired report is available for this image."
                    )
                )
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"attachment:{attachment.id}",
                        document_id=image.image_id,
                        document_name=image.filename,
                        text=text,
                        score=1.0 if report_summary else 0.0,
                        metadata={
                            "source": "message_attachment",
                            "attachment_id": attachment.id,
                            "image_id": image.image_id,
                            "modality": image.modality,
                        },
                    )
                )
        normalized: list[RetrievedChunk] = []
        for chunk in chunks[:top_k]:
            metadata = {**(chunk.metadata or {})}
            metadata.setdefault("source", "message_attachment")
            if not metadata.get("attachment_id"):
                for attachment in attachments:
                    if attachment.document_id == chunk.document_id:
                        metadata["attachment_id"] = attachment.id
                        break
            normalized.append(chunk.model_copy(update={"score": chunk.score if chunk.score is not None else 1.0, "metadata": metadata}))
        return normalized

    def attachment_document_ids(
        self,
        attachment_ids: list[str],
        *,
        workspace_id: str,
        conversation_id: str,
    ) -> list[str]:
        attachments = self.store.require_message_attachments(
            attachment_ids=attachment_ids,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        document_ids: list[str] = []
        for attachment in attachments:
            document_id = attachment.document_id
            if not document_id and attachment.image_asset_id:
                image = self.store.get_image_asset(attachment.image_asset_id)
                document_id = image.ocr_document_id if image and image.workspace_id == workspace_id else None
            if document_id:
                document = self.store.get_document(document_id)
                if document is not None and document.workspace_id == workspace_id:
                    document_ids.append(document_id)
        return document_ids

    def compare_reports(self, document_ids: list[str], workspace_id: str = "default") -> CompareResponse:
        settings = self.embeddings.settings
        with trace_block(
            settings,
            "DocumentTools.compare_reports",
            "tool",
            {
                "document_ids": document_ids[:2],
                "workspace_id": workspace_id,
            },
        ) as run:
            if len(document_ids) < 2:
                response = CompareResponse(summary="Upload or select two reports to compare.", citations=[])
                run.end({"status": "not_enough_documents", "citation_count": 0})
                return response
            left = self.store.get_document_chunks(document_ids[0])
            right = self.store.get_document_chunks(document_ids[1])
            if not left or not right:
                response = CompareResponse(summary="I could not find indexed chunks for both reports.", citations=[])
                run.end({"status": "missing_chunks", "left_chunks": len(left), "right_chunks": len(right)})
                return response

            left_text = "\n".join(chunk.text for chunk in left)
            right_text = "\n".join(chunk.text for chunk in right)
            left_terms = self._important_terms(left_text)
            right_terms = self._important_terms(right_text)
            added = [term for term, _ in (right_terms - left_terms).most_common(8)]
            removed = [term for term, _ in (left_terms - right_terms).most_common(8)]
            similarity = SequenceMatcher(None, left_text[:5000], right_text[:5000]).ratio()

            summary_parts = [
                f"The reports are {similarity:.0%} textually similar across their indexed passages.",
            ]
            if added:
                summary_parts.append("New or more prominent terms in the later report: " + ", ".join(added) + ".")
            if removed:
                summary_parts.append("Terms less prominent or absent in the later report: " + ", ".join(removed) + ".")
            if not added and not removed:
                summary_parts.append("No major term-level changes were detected in the extracted text.")

            citations = citations_from_chunks((left[:2] + right[:2])[:4])
            response = CompareResponse(summary=" ".join(summary_parts), citations=citations)
            run.end(
                {
                    "status": "ok",
                    "similarity": round(similarity, 4),
                    "left_chunks": len(left),
                    "right_chunks": len(right),
                    "citation_count": len(citations),
                }
            )
            return response

    def extract_timeline(self, document_id: str) -> list[str]:
        chunks = self.store.get_document_chunks(document_id)
        events = []
        for chunk in chunks:
            for line in chunk.text.splitlines():
                if any(marker in line.lower() for marker in ["date", "admit", "discharge", "visit", "follow-up"]):
                    events.append(line.strip())
        return events[:20]

    def _important_terms(self, text: str) -> Counter:
        stopwords = {
            "with",
            "from",
            "that",
            "this",
            "have",
            "were",
            "patient",
            "report",
            "medical",
            "page",
            "and",
            "the",
            "for",
        }
        terms = [word.lower() for word in text.replace("/", " ").split()]
        cleaned = [term.strip(".,:;()[]{}").lower() for term in terms]
        return Counter(term for term in cleaned if len(term) > 3 and term not in stopwords)

    def _rerank_document_chunks(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        broad_summary_query = any(
            phrase in query.lower()
            for phrase in [
                "key findings",
                "main findings",
                "report findings",
                "findings of the report",
                "findings in the report",
                "health condition",
                "condition of the patient",
                "summary",
                "summarize",
            ]
        ) or bool(
            re.search(r"\bwhat\s+(?:are|r|is|'?s)?\s*(?:the\s+)?(?:main\s+|key\s+)?findings?\b", query.lower())
            or re.search(r"\bfindings?\s+(?:of|in|from)\s+(?:the\s+)?(?:report|document|file)\b", query.lower())
        )
        clinical_terms = {
            "finding",
            "findings",
            "impression",
            "summary",
            "diagnosis",
            "assessment",
            "rhythm",
            "sinus",
            "tachycardia",
            "bradycardia",
            "ectopic",
            "ventricular",
            "supraventricular",
            "avg",
            "average",
            "heart",
            "hr",
            "bpm",
            "episode",
            "events",
            "patient",
            "triggered",
        }

        def score(chunk: RetrievedChunk) -> float:
            text = clean_evidence_text(chunk.text).lower()
            terms = set(re.findall(r"[a-z0-9]+", text))
            lexical = len(query_terms & terms)
            clinical = len(clinical_terms & terms)
            base = chunk.score or 0
            penalty = 4 if "substitute for professional medical advice" in chunk.text.lower() else 0
            broad_boost = clinical * 0.6 if broad_summary_query else 0
            return base + lexical + broad_boost + (clinical * 0.15) - penalty

        return sorted(chunks, key=score, reverse=True)

    def _filter_retrieval_results(
        self,
        chunks: list[RetrievedChunk],
        *,
        minimum_similarity: float,
    ) -> tuple[list[RetrievedChunk], dict]:
        kept: list[RetrievedChunk] = []
        seen: set[str] = set()
        reasons = {
            "below_similarity_threshold": 0,
            "duplicate": 0,
        }
        for chunk in chunks:
            if (chunk.score or 0.0) < minimum_similarity:
                reasons["below_similarity_threshold"] += 1
                continue
            normalized = re.sub(r"\s+", " ", clean_evidence_text(chunk.text).lower()).strip()
            key = f"{chunk.document_id}:{chunk.page}:{normalized[:700]}"
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            kept.append(chunk)
        filtered_count = len(chunks) - len(kept)
        return kept, {
            "score_direction": "higher_is_better",
            "returned_count": len(chunks),
            "kept_count": len(kept),
            "filtered_count": filtered_count,
            "below_threshold_count": reasons["below_similarity_threshold"],
            "filter_reasons": reasons,
            "counter_invariant_ok": len(kept) + filtered_count == len(chunks),
            "highest_similarity": max((chunk.score or 0.0 for chunk in chunks), default=None),
            "lowest_kept_similarity": min((chunk.score or 0.0 for chunk in kept), default=None),
            "min_similarity": minimum_similarity,
        }

    def _select_known_chunk_for_embedding_probe(
        self,
        *,
        query: str,
        rows: list[dict],
        document_name_contains: str = "",
        chunk_text_contains: str = "",
    ) -> dict | None:
        filtered = rows
        if document_name_contains:
            needle = document_name_contains.lower()
            filtered = [row for row in filtered if needle in row["document_name"].lower()]
        if chunk_text_contains:
            needle = chunk_text_contains.lower()
            filtered = [row for row in filtered if needle in row["text"].lower()]
        if not filtered:
            return None

        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in {"the", "about", "tell", "patient", "report", "document"}
        }
        if not query_terms:
            return filtered[0]

        def lexical_score(row: dict) -> tuple[int, int]:
            text = f"{row['document_name']} {row['text']}".lower()
            matched_terms = sum(1 for term in query_terms if term in text)
            matched_chars = sum(len(term) for term in query_terms if term in text)
            return matched_terms, matched_chars

        return max(filtered, key=lexical_score)

    def _embedding_consistency_issues(self, query_embedding: list[float], diagnostics: list[dict]) -> list[str]:
        issues = []
        current_model = self.embeddings.last_model or self.embeddings.configured_model_label()
        current_provider = self.embeddings.last_provider
        for item in diagnostics:
            if item["dimension"] != len(query_embedding):
                issues.append(
                    f"{item['document_name']}:{item['chunk_id']} has dimension {item['dimension']} but query has {len(query_embedding)}"
                )
            if item["non_zero_count"] == 0:
                issues.append(f"{item['document_name']}:{item['chunk_id']} has a zero embedding vector")
            stored_model = item.get("embedding_model") or ""
            stored_provider = item.get("embedding_provider") or ""
            if not stored_model:
                issues.append(
                    f"{item['document_name']}:{item['chunk_id']} has no stored embedding model metadata; re-index to verify consistency"
                )
            elif stored_model != current_model:
                issues.append(
                    f"{item['document_name']}:{item['chunk_id']} was indexed with model {stored_model} but query used {current_model}"
                )
            if stored_provider and current_provider and stored_provider != current_provider:
                issues.append(
                    f"{item['document_name']}:{item['chunk_id']} was indexed with provider {stored_provider} but query used {current_provider}"
                )
            stored_index_version = item.get("embedding_index_version") or ""
            if stored_index_version and stored_index_version != self.embeddings.settings.embedding_index_version:
                issues.append(
                    f"{item['document_name']}:{item['chunk_id']} was indexed with {stored_index_version} but retrieval is using {self.embeddings.settings.embedding_index_version}"
                )
        return issues
