import math
from typing import Any

from .config import Settings
from .schemas import RetrievedChunk
from .store import SQLiteStore


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorStore:
    def __init__(self, settings: Settings, sqlite_store: SQLiteStore):
        self.settings = settings
        self.sqlite_store = sqlite_store
        self._pinecone_index: Any | None = None

    def upsert(self, chunks: list[RetrievedChunk], embeddings: list[list[float]], workspace_id: str) -> None:
        self.sqlite_store.replace_chunks(chunks, embeddings, workspace_id)
        if not self.settings.should_use_pinecone:
            return
        index = self._get_pinecone_index(len(embeddings[0]) if embeddings else self.settings.embedding_dimension)
        if index is None:
            return
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "workspace_id": workspace_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page": chunk.page,
                        "text": chunk.text[:3500],
                    },
                }
            )
        if vectors:
            try:
                index.upsert(vectors=vectors)
            except Exception:
                # The SQLite store has already been updated; hosted vector DB failures
                # should not break local demos or document upload.
                return

    def search(self, query_embedding: list[float], workspace_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.settings.should_use_pinecone:
            index = self._get_pinecone_index(len(query_embedding))
            if index is not None:
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"workspace_id": {"$eq": workspace_id}},
                    )
                    chunks = []
                    for match in results.get("matches", []):
                        metadata = match.get("metadata", {})
                        chunks.append(
                            RetrievedChunk(
                                chunk_id=match["id"],
                                document_id=metadata.get("document_id", ""),
                                document_name=metadata.get("document_name", ""),
                                text=metadata.get("text", ""),
                                page=metadata.get("page"),
                                score=match.get("score"),
                            )
                        )
                    return chunks
                except Exception:
                    pass
        scored = []
        for chunk, embedding in self.sqlite_store.get_local_vectors(workspace_id):
            scored.append((cosine_similarity(query_embedding, embedding), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk.model_copy(update={"score": score}) for score, chunk in scored[:top_k]]

    def _get_pinecone_index(self, dimension: int):
        if self._pinecone_index is not None:
            return self._pinecone_index
        if not self.settings.pinecone_api_key:
            return None
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            existing = [index["name"] if isinstance(index, dict) else index.name for index in pc.list_indexes()]
            if self.settings.pinecone_index_name not in existing:
                pc.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                )
            self._pinecone_index = pc.Index(self.settings.pinecone_index_name)
        except Exception:
            self._pinecone_index = None
        return self._pinecone_index
