from collections import Counter
from difflib import SequenceMatcher
import re

from .education import MedicalEducationStore
from .embeddings import EmbeddingClient
from .schemas import Citation, CompareResponse, RetrievedChunk
from .store import SQLiteStore
from .vectorstore import VectorStore


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

    def document_search(self, query: str, workspace_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embeddings.embed_query(query)
        chunks = self.vectorstore.search(query_embedding, workspace_id=workspace_id, top_k=max(top_k * 3, 12))
        return self._rerank_document_chunks(query, chunks)[:top_k]

    def medical_education_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self.education.search(query, top_k=top_k)

    def compare_reports(self, document_ids: list[str], workspace_id: str = "default") -> CompareResponse:
        if len(document_ids) < 2:
            return CompareResponse(summary="Upload or select two reports to compare.", citations=[])
        left = self.store.get_document_chunks(document_ids[0])
        right = self.store.get_document_chunks(document_ids[1])
        if not left or not right:
            return CompareResponse(summary="I could not find indexed chunks for both reports.", citations=[])

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
        return CompareResponse(summary=" ".join(summary_parts), citations=citations)

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
            for phrase in ["key findings", "health condition", "condition of the patient", "summary", "summarize"]
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
