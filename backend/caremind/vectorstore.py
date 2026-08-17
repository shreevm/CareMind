import json
import logging
import math
import re
from typing import Any

from .config import Settings
from .observability import chunk_outputs, trace_block
from .schemas import RetrievedChunk
from .store import SQLiteStore

logger = logging.getLogger(__name__)


class VectorStoreUnavailable(RuntimeError):
    """Raised when the required vector backend cannot serve indexing or search."""


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
        self._pinecone_dimension: int | None = None
        self._supabase_schema_checked = False

    def upsert(
        self,
        chunks: list[RetrievedChunk],
        embeddings: list[list[float]],
        workspace_id: str,
        *,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_index_version: str = "",
    ) -> None:
        embedding_provider = embedding_provider or self.settings.embedding_provider
        embedding_model = embedding_model or self.settings.embedding_model
        embedding_index_version = embedding_index_version or self.settings.embedding_index_version
        with trace_block(
            self.settings,
            "VectorStore.upsert",
            "tool",
            {
                "backend": self.settings.vector_backend,
                "workspace_id": workspace_id,
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "dimension": len(embeddings[0]) if embeddings else self.settings.embedding_dimension,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_index_version": embedding_index_version,
                "chunks": chunk_outputs(self.settings, chunks),
            },
        ) as run:
            logger.info(
                "vectorstore.upsert.start backend=%s workspace_id=%s chunks=%s embeddings=%s",
                self.settings.vector_backend,
                workspace_id,
                len(chunks),
                len(embeddings),
            )
            if self.settings.should_use_sqlite_vectors:
                self.sqlite_store.replace_chunks(
                    chunks,
                    embeddings,
                    workspace_id,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_index_version=embedding_index_version,
                )
                logger.info("vectorstore.upsert.done backend=sqlite workspace_id=%s chunks=%s", workspace_id, len(chunks))
                run.end({"backend": "sqlite", "status": "ok", "chunk_count": len(chunks)})
                return
            if self.settings.should_use_supabase:
                self._upsert_supabase(
                    chunks,
                    embeddings,
                    workspace_id,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_index_version=embedding_index_version,
                )
                self.sqlite_store.replace_chunks(
                    chunks,
                    embeddings,
                    workspace_id,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_index_version=embedding_index_version,
                )
                logger.info("vectorstore.upsert.done backend=supabase workspace_id=%s chunks=%s", workspace_id, len(chunks))
                run.end({"backend": "supabase", "status": "ok", "chunk_count": len(chunks)})
                return
            if not self.settings.should_use_pinecone:
                raise VectorStoreUnavailable(f"Unsupported CAREMIND_VECTOR_BACKEND '{self.settings.vector_backend}'.")
            index = self._get_pinecone_index(len(embeddings[0]) if embeddings else self.settings.embedding_dimension)
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append(
                    {
                        "id": chunk.chunk_id,
                        "values": embedding,
                        "metadata": {
                            **(chunk.metadata or {}),
                            "workspace_id": workspace_id,
                            "document_id": chunk.document_id,
                            "document_name": chunk.document_name,
                            "page": chunk.page,
                            "text": chunk.text[:3500],
                            "embedding_provider": embedding_provider,
                            "embedding_model": embedding_model,
                            "embedding_index_version": embedding_index_version,
                        },
                    }
                )
            if vectors:
                try:
                    index.upsert(vectors=vectors)
                except Exception as exc:
                     logger.exception("vectorstore.pinecone.upsert.failed index=%s", self.settings.pinecone_index_name)
                     import traceback
                     traceback.print_exc()
                     print(type(exc))
                     print(exc)
                     print("=" * 80)
                     print("SUPABASE UPSERT ERROR")
                    #  print(type(exc).__name__)
                    #  print(str(exc))
                    #  print("=" * 80)
                     raise VectorStoreUnavailable(f"Supabase upsert failed: {exc.__class__.__name__}") from exc
            self.sqlite_store.replace_chunks(
                chunks,
                embeddings,
                workspace_id,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_index_version=embedding_index_version,
            )
            logger.info("vectorstore.upsert.done backend=pinecone workspace_id=%s chunks=%s", workspace_id, len(chunks))
            run.end({"backend": "pinecone", "status": "ok", "chunk_count": len(chunks)})

    def search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        top_k: int = 5,
        *,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_index_version: str = "",
    ) -> list[RetrievedChunk]:
        embedding_provider = embedding_provider or self.settings.embedding_provider
        embedding_model = embedding_model or self.settings.embedding_model
        embedding_index_version = embedding_index_version or self.settings.embedding_index_version
        with trace_block(
            self.settings,
            "VectorStore.search",
            "retriever",
            {
                "backend": self.settings.vector_backend,
                "workspace_id": workspace_id,
                "dimension": len(query_embedding),
                "top_k": top_k,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_index_version": embedding_index_version,
            },
        ) as run:
            logger.info(
                "vectorstore.search.start backend=%s workspace_id=%s dimension=%s top_k=%s",
                self.settings.vector_backend,
                workspace_id,
                len(query_embedding),
                top_k,
            )
            if self.settings.should_use_sqlite_vectors:
                scored = []
                for chunk, embedding in self.sqlite_store.get_local_vectors(
                    workspace_id,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_dimension=len(query_embedding),
                    embedding_index_version=embedding_index_version,
                ):
                    scored.append((cosine_similarity(query_embedding, embedding), chunk))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                results = [chunk.model_copy(update={"score": score}) for score, chunk in scored[:top_k]]
                logger.info("vectorstore.search.done backend=sqlite workspace_id=%s results=%s", workspace_id, len(results))
                run.end({"backend": "sqlite", "retrieved_count": len(results), "results": chunk_outputs(self.settings, results)})
                return results
            if self.settings.should_use_supabase:
                results = self._search_supabase(
                    query_embedding,
                    workspace_id,
                    top_k,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_index_version=embedding_index_version,
                )
                logger.info("vectorstore.search.done backend=supabase workspace_id=%s results=%s", workspace_id, len(results))
                run.end({"backend": "supabase", "retrieved_count": len(results), "results": chunk_outputs(self.settings, results)})
                return results
            if not self.settings.should_use_pinecone:
                raise VectorStoreUnavailable(f"Unsupported CAREMIND_VECTOR_BACKEND '{self.settings.vector_backend}'.")
            index = self._get_pinecone_index(len(query_embedding))
            self._assert_dimension(len(query_embedding))
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={
                        "workspace_id": {"$eq": workspace_id},
                        "embedding_provider": {"$eq": embedding_provider},
                        "embedding_model": {"$eq": embedding_model},
                        "embedding_dimension": {"$eq": len(query_embedding)},
                        "embedding_index_version": {"$eq": embedding_index_version},
                    },
                )
            except Exception as exc:
                logger.exception("vectorstore.pinecone.query.failed index=%s", self.settings.pinecone_index_name)
                raise VectorStoreUnavailable(f"Pinecone query failed: {exc.__class__.__name__}") from exc
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
                        metadata={key: value for key, value in metadata.items() if key != "text"},
                    )
                )
            logger.info("vectorstore.search.done backend=pinecone workspace_id=%s results=%s", workspace_id, len(chunks))
            run.end({"backend": "pinecone", "retrieved_count": len(chunks), "results": chunk_outputs(self.settings, chunks)})
            return chunks

    def delete_document(self, document_id: str, workspace_id: str, chunk_ids: list[str] | None = None) -> None:
        logger.info(
            "vectorstore.delete_document.start backend=%s workspace_id=%s document_id=%s",
            self.settings.vector_backend,
            workspace_id,
            document_id,
        )
        if self.settings.should_use_sqlite_vectors:
            logger.info("vectorstore.delete_document.skip backend=sqlite document_id=%s", document_id)
            return
        if self.settings.should_use_supabase:
            self._delete_supabase_document(document_id, workspace_id)
            return
        if self.settings.should_use_pinecone:
            ids = chunk_ids or [chunk.chunk_id for chunk in self.sqlite_store.get_document_chunks(document_id)]
            if ids:
                index = self._get_pinecone_index(self.settings.embedding_dimension)
                index.delete(ids=ids)
            logger.info("vectorstore.delete_document.done backend=pinecone document_id=%s ids=%s", document_id, len(ids))
            return
        raise VectorStoreUnavailable(f"Unsupported CAREMIND_VECTOR_BACKEND '{self.settings.vector_backend}'.")

    def status(self) -> dict[str, Any]:
        if self.settings.should_use_sqlite_vectors:
            return {
                "backend": "sqlite",
                "configured": True,
                "connected": True,
                "required": False,
            }
        if self.settings.should_use_supabase:
            return self.supabase_status()
        if self.settings.should_use_pinecone:
            return self.pinecone_status()
        return {
            "backend": self.settings.vector_backend,
            "configured": False,
            "connected": False,
            "required": False,
            "error": f"Unsupported CAREMIND_VECTOR_BACKEND '{self.settings.vector_backend}'",
        }

    def supabase_status(self) -> dict[str, Any]:
        if not self.settings.should_use_supabase:
            return {
                "backend": "supabase",
                "project_ref": self.settings.supabase_project_ref,
                "url_configured": bool(self.settings.supabase_url),
                "publishable_key_configured": bool(self.settings.supabase_publishable_key),
                "secret_key_configured": bool(self.settings.supabase_secret_key),
                "db_url_configured": bool(self.settings.supabase_db_url),
                "configured": False,
                "connected": False,
                "table_exists": False,
                "required": False,
                "error": "Supabase disabled by CAREMIND_VECTOR_BACKEND",
            }
        if not self.settings.supabase_db_url and not self.settings.has_supabase_rest_credentials:
            return {
                "backend": "supabase",
                "project_ref": self.settings.supabase_project_ref,
                "url_configured": bool(self.settings.supabase_url),
                "publishable_key_configured": bool(self.settings.supabase_publishable_key),
                "secret_key_configured": bool(self.settings.supabase_secret_key),
                "db_url_configured": False,
                "configured": False,
                "connected": False,
                "table_exists": False,
                "required": True,
                "error": "SUPABASE_DB_URL or SUPABASE_URL + SUPABASE_SECRET_KEY is required for server-side vector writes",
            }
        try:
            if self.settings.supabase_db_url:
                with self._supabase_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("select to_regclass(%s)", (self._qualified_supabase_table(),))
                        table_exists = cur.fetchone()[0] is not None
                auto_migration = None
                if not table_exists:
                    self.ensure_supabase_schema()
                    table_exists = True
                    auto_migration = "available"
                access_method = "postgres"
            else:
                table_exists = self._supabase_rest_table_exists()
                auto_migration = None
                access_method = "rest"
            status = {
                "backend": "supabase",
                "project_ref": self.settings.supabase_project_ref,
                "url_configured": bool(self.settings.supabase_url),
                "publishable_key_configured": bool(self.settings.supabase_publishable_key),
                "secret_key_configured": bool(self.settings.supabase_secret_key),
                "db_url_configured": bool(self.settings.supabase_db_url),
                "access_method": access_method,
                "configured": True,
                "connected": True,
                "table_exists": table_exists,
                "table": self._qualified_supabase_table(),
                "expected_dimension": self.settings.embedding_dimension,
                "required": True,
            }
            if auto_migration:
                status["auto_migration"] = auto_migration
            return status
        except Exception as exc:
            return {
                "backend": "supabase",
                "project_ref": self.settings.supabase_project_ref,
                "url_configured": bool(self.settings.supabase_url),
                "publishable_key_configured": bool(self.settings.supabase_publishable_key),
                "secret_key_configured": bool(self.settings.supabase_secret_key),
                "db_url_configured": bool(self.settings.supabase_db_url),
                "configured": True,
                "connected": False,
                "table_exists": False,
                "table": self._qualified_supabase_table(),
                "required": True,
                "error": exc.__class__.__name__,
            }

    def pinecone_status(self) -> dict[str, Any]:
        if not self.settings.should_use_pinecone:
            return {
                "backend": "pinecone",
                "configured": False,
                "connected": False,
                "index_exists": False,
                "required": False,
                "error": "Pinecone disabled by CAREMIND_VECTOR_BACKEND",
            }
        if not self.settings.pinecone_api_key:
            return {
                "backend": "pinecone",
                "configured": False,
                "connected": False,
                "index_exists": False,
                "required": True,
                "error": "PINECONE_API_KEY is required",
            }
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            indexes = list(pc.list_indexes())
            existing = [index["name"] if isinstance(index, dict) else index.name for index in indexes]
            index_dimension = None
            if self.settings.pinecone_index_name in existing:
                index_dimension = self._describe_index_dimension(pc, self.settings.pinecone_index_name)
            return {
                "backend": "pinecone",
                "configured": True,
                "connected": True,
                "index_exists": self.settings.pinecone_index_name in existing,
                "index_name": self.settings.pinecone_index_name,
                "dimension": index_dimension,
                "expected_dimension": self.settings.embedding_dimension,
                "dimension_matches": index_dimension in {None, self.settings.embedding_dimension},
                "required": True,
            }
        except Exception as exc:
            return {
                "backend": "pinecone",
                "configured": True,
                "connected": False,
                "index_exists": False,
                "index_name": self.settings.pinecone_index_name,
                "required": True,
                "error": exc.__class__.__name__,
            }

    def _upsert_supabase(
        self,
        chunks: list[RetrievedChunk],
        embeddings: list[list[float]],
        workspace_id: str,
        *,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_index_version: str = "",
    ) -> None:
        if not self.settings.supabase_db_url:
            self._upsert_supabase_rest(
                chunks,
                embeddings,
                workspace_id,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_index_version=embedding_index_version,
            )
            return
        if not chunks:
            return
        self.ensure_supabase_ready()
        table = self._supabase_table_identifier()
        print("=" * 80)
        print("Runtime Configuration")
        print("Backend:", self.settings.vector_backend)
        print("Embedding Dimension:", self.settings.embedding_dimension)
        print("Embedding Provider:", embedding_provider)
        print("Embedding Model:", embedding_model)
        print("Embedding Index Version:", embedding_index_version)
        print("Actual Embedding Length:", len(embeddings[0]))
        print("=" * 80)
        try:
            from psycopg import sql

            with self._supabase_connect() as conn:
                with conn.cursor() as cur:
                    print("=" * 80)
                    print("UPSERT DEBUG")
                    print("Embedding provider :", embedding_provider)
                    print("Embedding model    :", embedding_model)
                    print("Embedding index    :", embedding_index_version)
                    print("Configured dim     :", self.settings.embedding_dimension)
                    print("=" * 80)
                    cur.execute(
                        sql.SQL("delete from {} where document_id = %s and workspace_id = %s").format(table),
                        (chunks[0].document_id, workspace_id),
                    )
                    for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        self._assert_vector_dimension(len(embedding))
                        cur.execute(
                            sql.SQL(
                                """
                                insert into {}
                                (
                                    chunk_id, document_id, workspace_id, document_name, page, position, text,
                                    embedding, embedding_provider, embedding_model, embedding_dimension,
                                    embedding_index_version, metadata
                                )
                                values (%s, %s, %s, %s, %s, %s, %s, %s::extensions.vector, %s, %s, %s, %s, %s::jsonb)
                                on conflict (chunk_id) do update set
                                    document_id = excluded.document_id,
                                    workspace_id = excluded.workspace_id,
                                    document_name = excluded.document_name,
                                    page = excluded.page,
                                    position = excluded.position,
                                    text = excluded.text,
                                    embedding = excluded.embedding,
                                    embedding_provider = excluded.embedding_provider,
                                    embedding_model = excluded.embedding_model,
                                    embedding_dimension = excluded.embedding_dimension,
                                    embedding_index_version = excluded.embedding_index_version,
                                    metadata = excluded.metadata,
                                    updated_at = now()
                                """
                            ).format(table),
                            (
                                chunk.chunk_id,
                                chunk.document_id,
                                workspace_id,
                                chunk.document_name,
                                chunk.page,
                                position,
                                chunk.text,
                                self._vector_literal(embedding),
                                embedding_provider,
                                embedding_model,
                                len(embedding),
                                embedding_index_version,
                                json.dumps(
                                    {
                                        "document_id": chunk.document_id,
                                        "document_name": chunk.document_name,
                                        "workspace_id": workspace_id,
                                        "page": chunk.page,
                                        "embedding_provider": embedding_provider,
                                        "embedding_model": embedding_model,
                                        "embedding_dimension": len(embedding),
                                        "embedding_index_version": embedding_index_version,
                                        **(chunk.metadata or {}),
                                    }
                                ),
                            ),
                        )
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(f"Supabase vector upsert failed: {exc.__class__.__name__}") from exc

    def _delete_supabase_document(self, document_id: str, workspace_id: str) -> None:
        if not self.settings.supabase_db_url:
            self._delete_supabase_document_rest(document_id, workspace_id)
            return
        self.ensure_supabase_ready()
        table = self._supabase_table_identifier()
        try:
            from psycopg import sql

            with self._supabase_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("delete from {} where document_id = %s and workspace_id = %s").format(table),
                        (document_id, workspace_id),
                    )
            logger.info("vectorstore.delete_document.done backend=supabase document_id=%s", document_id)
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(f"Supabase vector delete failed: {exc.__class__.__name__}") from exc

    def _search_supabase(
        self,
        query_embedding: list[float],
        workspace_id: str,
        top_k: int,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_index_version: str,
    ) -> list[RetrievedChunk]:
        if not self.settings.supabase_db_url:
            return self._search_supabase_rest(
                query_embedding,
                workspace_id,
                top_k,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_index_version=embedding_index_version,
            )
        self._assert_vector_dimension(len(query_embedding))
        self.ensure_supabase_ready()
        table = self._supabase_table_identifier()
        try:
            from psycopg import sql

            query_vector = self._vector_literal(query_embedding)
            with self._supabase_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            select
                                chunk_id,
                                document_id,
                                document_name,
                                text,
                                page,
                                1 - (embedding <=> %s::extensions.vector) as score
                            from {}
                            where workspace_id = %s
                              and embedding_provider = %s
                              and embedding_model = %s
                              and embedding_dimension = %s
                              and embedding_index_version = %s
                            order by embedding <=> %s::extensions.vector
                            limit %s
                            """
                        ).format(table),
                        (
                            query_vector,
                            workspace_id,
                            embedding_provider,
                            embedding_model,
                            len(query_embedding),
                            embedding_index_version,
                            query_vector,
                            top_k,
                        ),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(f"Supabase vector query failed: {exc.__class__.__name__}") from exc
        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                document_name=row[2],
                text=row[3],
                page=row[4],
                score=float(row[5]) if row[5] is not None else None,
            )
            for row in rows
        ]

    def _upsert_supabase_rest(
        self,
        chunks: list[RetrievedChunk],
        embeddings: list[list[float]],
        workspace_id: str,
        *,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_index_version: str = "",
    ) -> None:
        if not self.settings.has_supabase_rest_credentials:
            raise VectorStoreUnavailable(
                "SUPABASE_URL and SUPABASE_SECRET_KEY are required for CareMind Supabase REST vector search."
            )
        if not chunks:
            logger.info("supabase.rest.upsert.skip reason=no_chunks workspace_id=%s", workspace_id)
            return
        if not self._supabase_rest_table_exists():
            raise VectorStoreUnavailable(
                "Supabase vector table is missing. Set SUPABASE_DB_URL so CareMind can create it automatically, "
                "or run the Supabase migration manually."
            )
        try:
            with self._supabase_rest_client() as client:
                logger.info(
                    "supabase.rest.delete.start project_ref=%s table=%s document_id=%s workspace_id=%s",
                    self.settings.supabase_project_ref,
                    self._unqualified_supabase_table(),
                    chunks[0].document_id,
                    workspace_id,
                )
                delete_response = client.delete(
                    self._supabase_rest_table_url(),
                    params={
                        "document_id": f"eq.{chunks[0].document_id}",
                        "workspace_id": f"eq.{workspace_id}",
                    },
                )
                delete_response.raise_for_status()


                logger.info(
                    "supabase.rest.delete.done status=%s document_id=%s workspace_id=%s",
                    delete_response.status_code,
                    chunks[0].document_id,
                    workspace_id,
                )
                rows = []
                for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    self._assert_vector_dimension(len(embedding))
                    rows.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "workspace_id": workspace_id,
                            "document_name": chunk.document_name,
                            "page": chunk.page,
                            "position": position,
                            "text": chunk.text,
                            "embedding": self._vector_literal(embedding),
                            "embedding_provider": embedding_provider,
                            "embedding_model": embedding_model,
                            "embedding_dimension": len(embedding),
                            "embedding_index_version": embedding_index_version,
                            "metadata": {
                                "document_id": chunk.document_id,
                                "document_name": chunk.document_name,
                                "workspace_id": workspace_id,
                                "page": chunk.page,
                                "embedding_provider": embedding_provider,
                                "embedding_model": embedding_model,
                                "embedding_dimension": len(embedding),
                                "embedding_index_version": embedding_index_version,
                                **(chunk.metadata or {}),
                            },
                        }
                    )
                if rows:
                    logger.info(
                        "supabase.rest.upsert.start project_ref=%s table=%s rows=%s dimension=%s",
                        self.settings.supabase_project_ref,
                        self._unqualified_supabase_table(),
                        len(rows),
                        len(embeddings[0]) if embeddings else 0,
                    )
                    upsert_response = client.post(
                        self._supabase_rest_table_url(),
                        params={"on_conflict": "chunk_id"},
                        headers={"Prefer": "resolution=merge-duplicates"},
                        json=rows,
                    )
                    upsert_response.raise_for_status()
                    logger.info(
                        "supabase.rest.upsert.done status=%s rows=%s",
                        upsert_response.status_code,
                        len(rows),
                    )
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            detail = self._http_error_detail(exc)
            logger.exception("supabase.rest.upsert.failed detail=%s", detail)
            raise VectorStoreUnavailable(f"Supabase REST vector upsert failed: {detail}") from exc

    def _delete_supabase_document_rest(self, document_id: str, workspace_id: str) -> None:
        if not self.settings.has_supabase_rest_credentials:
            raise VectorStoreUnavailable(
                "SUPABASE_URL and SUPABASE_SECRET_KEY are required for CareMind Supabase REST vector delete."
            )
        if not self._supabase_rest_table_exists():
            return
        try:
            with self._supabase_rest_client() as client:
                response = client.delete(
                    self._supabase_rest_table_url(),
                    params={
                        "document_id": f"eq.{document_id}",
                        "workspace_id": f"eq.{workspace_id}",
                    },
                )
                response.raise_for_status()
            logger.info("vectorstore.delete_document.done backend=supabase_rest document_id=%s", document_id)
        except Exception as exc:
            detail = self._http_error_detail(exc)
            logger.exception("supabase.rest.delete.failed detail=%s", detail)
            raise VectorStoreUnavailable(f"Supabase REST vector delete failed: {detail}") from exc

    def _search_supabase_rest(
        self,
        query_embedding: list[float],
        workspace_id: str,
        top_k: int,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_index_version: str,
    ) -> list[RetrievedChunk]:
        if not self.settings.has_supabase_rest_credentials:
            raise VectorStoreUnavailable(
                "SUPABASE_URL and SUPABASE_SECRET_KEY are required for CareMind Supabase REST vector search."
            )
        self._assert_vector_dimension(len(query_embedding))
        try:
            with self._supabase_rest_client() as client:
                logger.info(
                    "supabase.rest.search.start project_ref=%s table=%s workspace_id=%s dimension=%s top_k=%s",
                    self.settings.supabase_project_ref,
                    self._unqualified_supabase_table(),
                    workspace_id,
                    len(query_embedding),
                    top_k,
                )

                response = client.post(
                    self._supabase_rest_rpc_url(self.settings.supabase_match_function),
                    json={
                        "query_embedding": self._vector_literal(query_embedding),
                        "match_workspace_id": workspace_id,
                        "match_count": top_k,
                        "match_embedding_provider": embedding_provider,
                        "match_embedding_model": embedding_model,
                        "match_embedding_dimension": len(query_embedding),
                        "match_embedding_index_version": embedding_index_version,
                    },
                )
                response.raise_for_status()
                rows = response.json()
                logger.info(
                    "supabase.rest.search.done status=%s workspace_id=%s rows=%s",
                    response.status_code,
                    workspace_id,
                    len(rows),
                )
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            detail = self._http_error_detail(exc)
            logger.exception("supabase.rest.search.failed detail=%s", detail)
            raise VectorStoreUnavailable(f"Supabase REST vector query failed: {detail}") from exc
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                text=row["text"],
                page=row.get("page"),
                score=float(row["score"]) if row.get("score") is not None else None,
            )
            for row in rows
        ]

    def _supabase_rest_table_exists(self) -> bool:
        with self._supabase_rest_client() as client:
            logger.info(
                "supabase.rest.table_check.start project_ref=%s table=%s",
                self.settings.supabase_project_ref,
                self._unqualified_supabase_table(),
            )
            response = client.get(
                self._supabase_rest_table_url(),
                params={"select": "chunk_id", "limit": "1"},
            )
            if response.status_code == 404:
                logger.warning(
                    "supabase.rest.table_check.missing status=%s table=%s",
                    response.status_code,
                    self._unqualified_supabase_table(),
                )
                return False
            response.raise_for_status()
            logger.info(
                "supabase.rest.table_check.done status=%s table=%s",
                response.status_code,
                self._unqualified_supabase_table(),
            )
            return True

    def _supabase_rest_client(self):
        if not self.settings.supabase_url or not self.settings.supabase_secret_key:
            raise VectorStoreUnavailable("SUPABASE_URL and SUPABASE_SECRET_KEY are required.")
        import httpx

        return httpx.Client(
            timeout=45,
            headers={
                "apikey": self.settings.supabase_secret_key,
                "Authorization": f"Bearer {self.settings.supabase_secret_key}",
                "Content-Type": "application/json",
            },
        )

    def _supabase_rest_table_url(self) -> str:
        return f"{self.settings.supabase_url.rstrip('/')}/rest/v1/{self._unqualified_supabase_table()}"

    def _supabase_rest_rpc_url(self, function_name: str) -> str:
        return f"{self.settings.supabase_url.rstrip('/')}/rest/v1/rpc/{function_name}"

    def _http_error_detail(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return exc.__class__.__name__
        body = response.text.strip()
        if len(body) > 700:
            body = body[:700] + "..."
        return f"HTTP {response.status_code}: {body or response.reason_phrase}"

    def _supabase_connect(self):
        try:
            import psycopg
        except Exception as exc:
            raise VectorStoreUnavailable("Install psycopg[binary] to use Supabase pgvector.") from exc
        return psycopg.connect(self.settings.supabase_db_url, connect_timeout=10)

    def ensure_supabase_ready(self) -> None:
        if self._supabase_schema_checked:
            return
        self.ensure_supabase_schema()
        self._supabase_schema_checked = True

    def ensure_supabase_schema(self) -> None:
        if not self.settings.supabase_db_url:
            raise VectorStoreUnavailable("Supabase automatic table creation requires SUPABASE_DB_URL.")
        dimension = int(self.settings.embedding_dimension)
        table = self._supabase_table_identifier()
        function_name = self.settings.supabase_match_function
        try:
            from psycopg import sql

            with self._supabase_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("create extension if not exists vector with schema extensions")
                    cur.execute(
                        sql.SQL(
                            """
                            create table if not exists {} (
                                chunk_id text primary key,
                                document_id text not null,
                                workspace_id text not null,
                                document_name text not null,
                                page integer,
                                position integer not null,
                                text text not null,
                                embedding extensions.vector({}) not null,
                                embedding_provider text not null default '',
                                embedding_model text not null default '',
                                embedding_dimension integer,
                                embedding_index_version text not null default '',
                                metadata jsonb not null default '{{}}'::jsonb,
                                created_at timestamptz not null default now(),
                                updated_at timestamptz not null default now()
                            )
                            """
                        ).format(table, sql.SQL(str(dimension)))
                    )
                    cur.execute(sql.SQL("alter table {} add column if not exists embedding_provider text not null default ''").format(table))
                    cur.execute(sql.SQL("alter table {} add column if not exists embedding_model text not null default ''").format(table))
                    cur.execute(sql.SQL("alter table {} add column if not exists embedding_dimension integer").format(table))
                    cur.execute(sql.SQL("alter table {} add column if not exists embedding_index_version text not null default ''").format(table))
                    index_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", self._unqualified_supabase_table())[:48]
                    cur.execute(
                        sql.SQL("create index if not exists {} on {} (workspace_id)").format(
                            sql.Identifier(f"{index_prefix}_workspace_idx"),
                            table,
                        )
                    )
                    cur.execute(
                        sql.SQL("create index if not exists {} on {} (document_id)").format(
                            sql.Identifier(f"{index_prefix}_document_idx"),
                            table,
                        )
                    )
                    cur.execute(
                        sql.SQL("create index if not exists {} on {} (workspace_id, embedding_provider, embedding_model, embedding_dimension, embedding_index_version)").format(
                            sql.Identifier(f"{index_prefix}_embedding_version_idx"),
                            table,
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "create index if not exists {} on {} using hnsw (embedding extensions.vector_cosine_ops)"
                        ).format(sql.Identifier(f"{index_prefix}_embedding_hnsw_idx"), table)
                    )
                    cur.execute(
                        sql.SQL(
                            """
                            create or replace function public.{}(
                                query_embedding extensions.vector({}),
                                match_workspace_id text,
                                match_count int default 5,
                                match_embedding_provider text default '',
                                match_embedding_model text default '',
                                match_embedding_dimension int default null,
                                match_embedding_index_version text default ''
                            )
                            returns table (
                                chunk_id text,
                                document_id text,
                                document_name text,
                                text text,
                                page integer,
                                score double precision
                            )
                            language sql
                            stable
                            as $$
                                select
                                    c.chunk_id,
                                    c.document_id,
                                    c.document_name,
                                    c.text,
                                    c.page,
                                    1 - (c.embedding <=> query_embedding) as score
                                from {} c
                                where c.workspace_id = match_workspace_id
                                  and c.embedding_provider = match_embedding_provider
                                  and c.embedding_model = match_embedding_model
                                  and c.embedding_dimension = match_embedding_dimension
                                  and c.embedding_index_version = match_embedding_index_version
                                order by c.embedding <=> query_embedding
                                limit match_count;
                            $$;
                            """
                        ).format(sql.Identifier(function_name), sql.SQL(str(dimension)), table)
                    )
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(f"Supabase schema setup failed: {exc.__class__.__name__}") from exc

    def _supabase_table_identifier(self):
        from psycopg import sql

        parts = [part for part in self.settings.supabase_vector_table.split(".") if part]
        if len(parts) == 1:
            return sql.Identifier(parts[0])
        if len(parts) == 2:
            return sql.Identifier(parts[0], parts[1])
        raise VectorStoreUnavailable("SUPABASE_VECTOR_TABLE must be a table name or schema-qualified table name.")

    def _qualified_supabase_table(self) -> str:
        table = self.settings.supabase_vector_table.strip()
        return table if "." in table else f"public.{table}"

    def _unqualified_supabase_table(self) -> str:
        return self.settings.supabase_vector_table.strip().split(".")[-1]

    def _vector_literal(self, vector: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def _assert_vector_dimension(self, requested_dimension: int) -> None:
        if requested_dimension == self.settings.embedding_dimension:
            return
        raise VectorStoreUnavailable(
            "Supabase vector dimension mismatch: "
            f"CAREMIND_EMBEDDING_DIM is {self.settings.embedding_dimension}, "
            f"but the current embedding vector has dimension {requested_dimension}. "
            "Update CAREMIND_EMBEDDING_DIM and the Supabase pgvector column dimension together."
        )

    def _get_pinecone_index(self, dimension: int):
        if self._pinecone_index is not None:
            self._assert_dimension(dimension)
            return self._pinecone_index
        if not self.settings.pinecone_api_key:
            raise VectorStoreUnavailable("PINECONE_API_KEY is required for CareMind vector search.")
        from pinecone import Pinecone, ServerlessSpec

        try:
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
                self._pinecone_dimension = dimension
            else:
                self._pinecone_dimension = self._describe_index_dimension(pc, self.settings.pinecone_index_name)
                self._assert_dimension(dimension)
            self._pinecone_index = pc.Index(self.settings.pinecone_index_name)
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(f"Pinecone is required but unavailable: {exc.__class__.__name__}") from exc
        return self._pinecone_index

    def _assert_dimension(self, requested_dimension: int) -> None:
        if self._pinecone_dimension is None or self._pinecone_dimension == requested_dimension:
            return
        raise VectorStoreUnavailable(
            "Pinecone index dimension mismatch: "
            f"index '{self.settings.pinecone_index_name}' has dimension {self._pinecone_dimension}, "
            f"but the current embedding vector has dimension {requested_dimension}. "
            "Create a new index or set CAREMIND_EMBEDDING_DIM/NVIDIA_EMBEDDING_MODEL to match the existing index."
        )

    def _describe_index_dimension(self, pc, index_name: str) -> int | None:
        try:
            description = pc.describe_index(index_name)
            if isinstance(description, dict):
                value = description.get("dimension")
            else:
                value = getattr(description, "dimension", None)
            return int(value) if value is not None else None
        except Exception:
            return None
