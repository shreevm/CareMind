import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .schemas import ChatMessage, DocumentMetadata, ImageMetadata, MessageAttachment, RetrievedChunk, TranscriptMetadata

logger = logging.getLogger(__name__)


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 15000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        print(f"[DB] Initializing SQLite database at {self.db_path}")
        logger.info("database.init.start path=%s", self.db_path)
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists documents (
                    document_id text primary key,
                    workspace_id text not null,
                    filename text not null,
                    content_type text,
                    file_path text not null,
                    uploaded_at text not null,
                    chunk_count integer not null default 0,
                    summary text,
                    source_image_id text,
                    document_type text,
                    understanding_confidence real
                );

                create table if not exists image_assets (
                    image_id text primary key,
                    workspace_id text not null,
                    filename text not null,
                    content_type text,
                    file_path text not null,
                    modality text not null,
                    uploaded_at text not null,
                    report_text_summary text,
                    ocr_document_id text
                );

                create table if not exists chunks (
                    chunk_id text primary key,
                    document_id text not null,
                    workspace_id text not null,
                    document_name text not null,
                    page integer,
                    position integer not null,
                    text text not null,
                    embedding_json text,
                    embedding_provider text,
                    embedding_model text,
                    embedding_dimension integer,
                    embedding_index_version text,
                    metadata_json text,
                    foreign key(document_id) references documents(document_id)
                );

                create table if not exists chunk_embeddings (
                    chunk_id text not null,
                    document_id text not null,
                    workspace_id text not null,
                    embedding_json text not null,
                    embedding_provider text not null,
                    embedding_model text not null,
                    embedding_dimension integer not null,
                    embedding_index_version text not null,
                    created_at text not null,
                    updated_at text not null,
                    primary key(chunk_id, embedding_index_version),
                    foreign key(chunk_id) references chunks(chunk_id)
                );

                create table if not exists messages (
                    id integer primary key autoincrement,
                    session_id text not null,
                    workspace_id text not null,
                    role text not null,
                    content text not null,
                    created_at text not null
                );

                create table if not exists message_attachments (
                    id text primary key,
                    message_id integer,
                    conversation_id text not null,
                    workspace_id text not null,
                    user_id text,
                    attachment_type text not null,
                    filename text not null,
                    mime_type text,
                    file_size integer not null default 0,
                    storage_bucket text not null default 'local-uploads',
                    storage_path text not null,
                    image_asset_id text,
                    document_id text,
                    processing_status text not null default 'completed',
                    processing_error text,
                    created_at text not null,
                    expires_at text,
                    persistence_mode text not null default 'saved_compat',
                    foreign key(message_id) references messages(id) on delete set null,
                    foreign key(image_asset_id) references image_assets(image_id),
                    foreign key(document_id) references documents(document_id)
                );

                create table if not exists voice_transcripts (
                    id integer primary key autoincrement,
                    message_id integer,
                    session_id text not null,
                    workspace_id text not null,
                    transcript text not null,
                    confidence real,
                    language text,
                    detected_language text,
                    language_confidence real,
                    timestamps_json text not null default '[]',
                    modality text,
                    input_modality text,
                    audio_storage_path text,
                    started_at text,
                    ended_at text,
                    created_at text not null,
                    foreign key(message_id) references messages(id) on delete set null
                );

                create index if not exists idx_voice_transcripts_message
                    on voice_transcripts(message_id);
                create index if not exists idx_voice_transcripts_session
                    on voice_transcripts(workspace_id, session_id, created_at);

                create index if not exists idx_message_attachments_message
                    on message_attachments(message_id);
                create index if not exists idx_message_attachments_conversation
                    on message_attachments(workspace_id, conversation_id, created_at);
                create index if not exists idx_message_attachments_document
                    on message_attachments(document_id);
                create index if not exists idx_message_attachments_image
                    on message_attachments(image_asset_id);

                create index if not exists idx_chunk_embeddings_version
                    on chunk_embeddings(
                        workspace_id, embedding_provider, embedding_model,
                        embedding_dimension, embedding_index_version
                    );

                create table if not exists workspace_revisions (
                    workspace_id text primary key,
                    revision integer not null default 0,
                    updated_at text not null
                );

                create table if not exists conversation_context (
                    session_id text not null,
                    workspace_id text not null,
                    payload_json text not null,
                    updated_at text not null,
                    primary key(session_id, workspace_id)
                );

                create table if not exists document_understanding (
                    document_id text primary key,
                    workspace_id text not null,
                    filename text not null,
                    content_type text,
                    document_type text,
                    summary text,
                    confidence real,
                    model_name text,
                    extraction_source text,
                    raw_json text not null,
                    index_text text not null,
                    created_at text not null,
                    updated_at text not null,
                    foreign key(document_id) references documents(document_id)
                );
                """
            )
            self._ensure_column(conn, "chunks", "embedding_provider", "text")
            self._ensure_column(conn, "chunks", "embedding_model", "text")
            self._ensure_column(conn, "chunks", "embedding_dimension", "integer")
            self._ensure_column(conn, "chunks", "embedding_index_version", "text")
            self._ensure_column(conn, "chunks", "metadata_json", "text")
            self._ensure_column(conn, "documents", "source_image_id", "text")
            self._ensure_column(conn, "documents", "document_type", "text")
            self._ensure_column(conn, "documents", "understanding_confidence", "real")
            self._ensure_column(conn, "image_assets", "content_type", "text")
            self._ensure_column(conn, "image_assets", "storage_path", "text")
            self._ensure_column(conn, "image_assets", "file_path", "text")
            self._ensure_column(conn, "image_assets", "modality", "text")
            self._ensure_column(conn, "image_assets", "uploaded_at", "text")
            self._ensure_column(conn, "image_assets", "summary", "text")
            self._ensure_column(conn, "image_assets", "report_text_summary", "text")
            self._ensure_column(conn, "image_assets", "ocr_document_id", "text")
            self._ensure_column(conn, "message_attachments", "message_id", "integer")
            self._ensure_column(conn, "message_attachments", "user_id", "text")
            self._ensure_column(conn, "message_attachments", "expires_at", "text")
            self._ensure_column(conn, "message_attachments", "persistence_mode", "text")
            self._ensure_column(conn, "voice_transcripts", "detected_language", "text")
            self._ensure_column(conn, "voice_transcripts", "language_confidence", "real")
            self._ensure_column(conn, "voice_transcripts", "timestamps_json", "text")
            self._ensure_column(conn, "voice_transcripts", "modality", "text")
            self._ensure_column(conn, "voice_transcripts", "input_modality", "text")
        print("[DB] Database tables created successfully: documents, image_assets, chunks, messages, message_attachments, workspace_revisions, conversation_context, document_understanding")
        logger.info("database.init.done tables_created=7")

    def save_image_asset(
        self,
        *,
        image_id: str,
        workspace_id: str,
        filename: str,
        content_type: str | None,
        file_path: Path,
        modality: str,
        report_text_summary: str | None = None,
        ocr_document_id: str | None = None,
    ) -> None:
        print(f"[DATA] Saving image asset - ID: {image_id}, Filename: {filename}, Workspace: {workspace_id}, Modality: {modality}")
        logger.info("store.save_image_asset.start image_id=%s workspace_id=%s filename=%s modality=%s", image_id, workspace_id, filename, modality)
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into image_assets
                (
                    image_id, workspace_id, filename, content_type, modality,
                    storage_path, file_path, uploaded_at, summary, report_text_summary,
                    ocr_document_id
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    workspace_id,
                    filename,
                    content_type,
                    modality,
                    str(file_path),
                    str(file_path),
                    datetime.now(timezone.utc).isoformat(),
                    report_text_summary,
                    report_text_summary,
                    ocr_document_id,
                ),
            )
            self._bump_workspace_revision(conn, workspace_id)
        print(f"[DATA] Image asset saved successfully")
        logger.info("store.save_image_asset.done image_id=%s status=ok", image_id)

    def list_images(self, workspace_id: str = "default") -> list[ImageMetadata]:
        logger.info("store.list_images.start workspace_id=%s", workspace_id)
        with self.connect() as conn:
            rows = conn.execute(
                "select * from image_assets where workspace_id = ? order by uploaded_at desc",
                (workspace_id,),
            ).fetchall()
        images = [
            ImageMetadata(
                image_id=row["image_id"],
                workspace_id=row["workspace_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                file_path=row["file_path"] or "",
                modality=row["modality"] or "clinical_image",
                uploaded_at=datetime.fromisoformat(row["uploaded_at"] or datetime.now(timezone.utc).isoformat()),
                report_text_summary=row["report_text_summary"],
                ocr_document_id=row["ocr_document_id"],
            )
            for row in rows
        ]
        print(f"[DATA] Retrieved {len(images)} images from workspace {workspace_id}")
        logger.info("store.list_images.done workspace_id=%s count=%s", workspace_id, len(images))
        return images

    def link_image_ocr_document(
        self,
        *,
        image_id: str,
        ocr_document_id: str,
        report_text_summary: str | None = None,
    ) -> None:
        logger.info("store.link_image_ocr_document.start image_id=%s document_id=%s", image_id, ocr_document_id)
        with self.connect() as conn:
            row = conn.execute("select workspace_id from image_assets where image_id = ?", (image_id,)).fetchone()
            if row is None:
                return
            conn.execute(
                """
                update image_assets
                set ocr_document_id = ?, report_text_summary = coalesce(?, report_text_summary)
                where image_id = ?
                """,
                (ocr_document_id, report_text_summary, image_id),
            )
            self._bump_workspace_revision(conn, row["workspace_id"])
        logger.info("store.link_image_ocr_document.done image_id=%s document_id=%s", image_id, ocr_document_id)

    def save_document(
        self,
        *,
        document_id: str,
        workspace_id: str,
        filename: str,
        content_type: str | None,
        file_path: Path,
        summary: str | None,
        source_image_id: str | None = None,
        document_type: str | None = None,
        understanding_confidence: float | None = None,
    ) -> None:
        print(f"[DATA] Saving document - ID: {document_id}, Filename: {filename}, Type: {content_type}, Workspace: {workspace_id}")
        logger.info("store.save_document.start document_id=%s workspace_id=%s filename=%s content_type=%s", document_id, workspace_id, filename, content_type)
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into documents
                (
                    document_id, workspace_id, filename, content_type, file_path, uploaded_at,
                    summary, source_image_id, document_type, understanding_confidence
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    workspace_id,
                    filename,
                    content_type,
                    str(file_path),
                    datetime.now(timezone.utc).isoformat(),
                    summary,
                    source_image_id,
                    document_type,
                    understanding_confidence,
                ),
            )
            self._bump_workspace_revision(conn, workspace_id)
        print(f"[DATA] Document saved successfully")
        logger.info("store.save_document.done document_id=%s status=ok", document_id)

    def save_document_understanding(
        self,
        *,
        document_id: str,
        workspace_id: str,
        filename: str,
        content_type: str | None,
        document_type: str,
        summary: str,
        confidence: float,
        model_name: str,
        extraction_source: str,
        raw_json: dict,
        index_text: str,
    ) -> None:
        logger.info(
            "store.save_document_understanding.start document_id=%s document_type=%s confidence=%.2f source=%s",
            document_id,
            document_type,
            confidence,
            extraction_source,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                insert into document_understanding
                (
                    document_id, workspace_id, filename, content_type, document_type, summary,
                    confidence, model_name, extraction_source, raw_json, index_text, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                    workspace_id = excluded.workspace_id,
                    filename = excluded.filename,
                    content_type = excluded.content_type,
                    document_type = excluded.document_type,
                    summary = excluded.summary,
                    confidence = excluded.confidence,
                    model_name = excluded.model_name,
                    extraction_source = excluded.extraction_source,
                    raw_json = excluded.raw_json,
                    index_text = excluded.index_text,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    workspace_id,
                    filename,
                    content_type,
                    document_type,
                    summary,
                    confidence,
                    model_name,
                    extraction_source,
                    json.dumps(raw_json),
                    index_text,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                update documents
                set document_type = ?, understanding_confidence = ?, summary = coalesce(?, summary)
                where document_id = ?
                """,
                (document_type, confidence, summary, document_id),
            )
            self._bump_workspace_revision(conn, workspace_id)
        logger.info("store.save_document_understanding.done document_id=%s", document_id)

    def get_document_understanding(self, document_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "select * from document_understanding where document_id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            raw_json = json.loads(row["raw_json"] or "{}")
        except Exception:
            raw_json = {}
        return {
            "document_id": row["document_id"],
            "workspace_id": row["workspace_id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "document_type": row["document_type"],
            "summary": row["summary"],
            "confidence": row["confidence"],
            "model_name": row["model_name"],
            "extraction_source": row["extraction_source"],
            "raw_json": raw_json,
            "index_text": row["index_text"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def replace_chunks(
        self,
        chunks: list[RetrievedChunk],
        embeddings: list[list[float]],
        workspace_id: str,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_dimension: int | None = None,
        embedding_index_version: str = "",
    ) -> None:
        print(f"[DATA] Storing chunks - Document: {chunks[0].document_id if chunks else 'N/A'}, Count: {len(chunks)}, Provider: {embedding_provider}, Model: {embedding_model}, Index: {embedding_index_version}")
        logger.info("store.replace_chunks.start document_id=%s chunk_count=%s embedding_provider=%s embedding_model=%s embedding_index_version=%s", chunks[0].document_id if chunks else "N/A", len(chunks), embedding_provider, embedding_model, embedding_index_version)
        with self.connect() as conn:
            if chunks:
                conn.execute("delete from chunks where document_id = ?", (chunks[0].document_id,))
            stored_dimension = embedding_dimension or (len(embeddings[0]) if embeddings else None)
            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    insert or replace into chunks
                    (
                        chunk_id, document_id, workspace_id, document_name, page, position, text,
                        embedding_json, embedding_provider, embedding_model, embedding_dimension,
                        embedding_index_version, metadata_json
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        workspace_id,
                        chunk.document_name,
                        chunk.page,
                        position,
                        chunk.text,
                        json.dumps(embedding),
                        embedding_provider,
                        embedding_model,
                        stored_dimension,
                        embedding_index_version,
                        json.dumps(chunk.metadata or {}),
                    ),
                )
                if embedding_index_version:
                    conn.execute(
                        """
                        insert into chunk_embeddings
                        (
                            chunk_id, document_id, workspace_id, embedding_json, embedding_provider,
                            embedding_model, embedding_dimension, embedding_index_version,
                            created_at, updated_at
                        )
                        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        on conflict(chunk_id, embedding_index_version) do update set
                            document_id = excluded.document_id,
                            workspace_id = excluded.workspace_id,
                            embedding_json = excluded.embedding_json,
                            embedding_provider = excluded.embedding_provider,
                            embedding_model = excluded.embedding_model,
                            embedding_dimension = excluded.embedding_dimension,
                            updated_at = excluded.updated_at
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            workspace_id,
                            json.dumps(embedding),
                            embedding_provider,
                            embedding_model,
                            stored_dimension,
                            embedding_index_version,
                            now,
                            now,
                        ),
                    )
            if chunks:
                conn.execute(
                    "update documents set chunk_count = ? where document_id = ?",
                    (len(chunks), chunks[0].document_id),
                )
                self._bump_workspace_revision(conn, workspace_id)
        print(f"[DATA] Chunks stored successfully - {len(chunks)} chunks with dimension {stored_dimension}")
        logger.info("store.replace_chunks.done chunk_count=%s dimension=%s status=ok", len(chunks), stored_dimension)

    def delete_document(self, document_id: str) -> dict[str, str] | None:
        print(f"[DATA] Deleting document - ID: {document_id}")
        logger.info("store.delete_document.start document_id=%s", document_id)
        with self.connect() as conn:
            row = conn.execute("select workspace_id, file_path from documents where document_id = ?", (document_id,)).fetchone()
            conn.execute("delete from chunk_embeddings where document_id = ?", (document_id,))
            conn.execute("delete from chunks where document_id = ?", (document_id,))
            conn.execute("delete from document_understanding where document_id = ?", (document_id,))
            conn.execute("delete from documents where document_id = ?", (document_id,))
            conn.execute("update image_assets set ocr_document_id = null where ocr_document_id = ?", (document_id,))
            if row is not None:
                self._bump_workspace_revision(conn, row["workspace_id"])
        print(f"[DATA] Document deleted successfully")
        logger.info("store.delete_document.done document_id=%s status=ok", document_id)
        if row is None:
            return None
        return {"workspace_id": row["workspace_id"], "file_path": row["file_path"]}

    def delete_image_asset(self, image_id: str) -> dict[str, str] | None:
        print(f"[DATA] Deleting image asset - ID: {image_id}")
        logger.info("store.delete_image_asset.start image_id=%s", image_id)
        with self.connect() as conn:
            row = conn.execute("select workspace_id, file_path, ocr_document_id from image_assets where image_id = ?", (image_id,)).fetchone()
            conn.execute("delete from image_assets where image_id = ?", (image_id,))
            if row is not None:
                self._bump_workspace_revision(conn, row["workspace_id"])
        print(f"[DATA] Image asset deleted successfully")
        logger.info("store.delete_image_asset.done image_id=%s status=ok", image_id)
        if row is None:
            return None
        return {"workspace_id": row["workspace_id"], "file_path": row["file_path"], "ocr_document_id": row["ocr_document_id"] or ""}

    def list_documents(self, workspace_id: str = "default") -> list[DocumentMetadata]:
        logger.info("store.list_documents.start workspace_id=%s", workspace_id)
        with self.connect() as conn:
            rows = conn.execute(
                "select * from documents where workspace_id = ? order by uploaded_at desc",
                (workspace_id,),
            ).fetchall()
        documents = [
            DocumentMetadata(
                document_id=row["document_id"],
                workspace_id=row["workspace_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                file_path=row["file_path"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                chunk_count=row["chunk_count"],
                summary=row["summary"],
                source_image_id=row["source_image_id"],
                document_type=row["document_type"],
                understanding_confidence=row["understanding_confidence"],
            )
            for row in rows
        ]
        print(f"[DATA] Retrieved {len(documents)} documents from workspace {workspace_id}")
        logger.info("store.list_documents.done workspace_id=%s count=%s", workspace_id, len(documents))
        return documents

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        logger.info("store.get_document.start document_id=%s", document_id)
        with self.connect() as conn:
            row = conn.execute("select * from documents where document_id = ?", (document_id,)).fetchone()
        if row is None:
            print(f"[DATA] Document not found - ID: {document_id}")
            logger.info("store.get_document.done document_id=%s status=not_found", document_id)
            return None
        doc = DocumentMetadata(
            document_id=row["document_id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            file_path=row["file_path"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            chunk_count=row["chunk_count"],
            summary=row["summary"],
            source_image_id=row["source_image_id"],
            document_type=row["document_type"],
            understanding_confidence=row["understanding_confidence"],
        )
        print(f"[DATA] Retrieved document - ID: {document_id}, Chunks: {doc.chunk_count}")
        logger.info("store.get_document.done document_id=%s chunk_count=%s status=ok", document_id, doc.chunk_count)
        return doc

    def get_image_asset(self, image_id: str) -> ImageMetadata | None:
        with self.connect() as conn:
            row = conn.execute("select * from image_assets where image_id = ?", (image_id,)).fetchone()
        if row is None:
            return None
        return ImageMetadata(
            image_id=row["image_id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            file_path=row["file_path"] or "",
            modality=row["modality"] or "clinical_image",
            uploaded_at=datetime.fromisoformat(row["uploaded_at"] or datetime.now(timezone.utc).isoformat()),
            report_text_summary=row["report_text_summary"],
            ocr_document_id=row["ocr_document_id"],
        )

    def get_document_chunks(self, document_id: str) -> list[RetrievedChunk]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from chunks where document_id = ? order by position asc",
                (document_id,),
            ).fetchall()
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                text=row["text"],
                page=row["page"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    def expand_chunks_with_neighbors(
        self,
        chunks: list[RetrievedChunk],
        *,
        window: int = 1,
        max_chars: int = 2400,
    ) -> list[RetrievedChunk]:
        expanded: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            document_chunks = self.get_document_chunks(chunk.document_id)
            index = self._chunk_position(chunk.chunk_id)
            if index is None:
                expanded.append(chunk)
                continue
            start = max(0, index - window)
            end = min(len(document_chunks), index + window + 1)
            parent_parts = [candidate.text for candidate in document_chunks[start:end] if candidate.text]
            parent_text = "\n\n".join(parent_parts).strip()
            if not parent_text:
                expanded.append(chunk)
                continue
            expanded.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    text=parent_text[:max_chars],
                    page=chunk.page,
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
            )
        return expanded

    def get_local_vectors(
        self,
        workspace_id: str,
        *,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_dimension: int | None = None,
        embedding_index_version: str = "",
    ) -> list[tuple[RetrievedChunk, list[float]]]:
        with self.connect() as conn:
            filters = ["e.workspace_id = ?"]
            params: list[object] = [workspace_id]
            if embedding_provider:
                filters.append("e.embedding_provider = ?")
                params.append(embedding_provider)
            if embedding_model:
                filters.append("e.embedding_model = ?")
                params.append(embedding_model)
            if embedding_dimension is not None:
                filters.append("e.embedding_dimension = ?")
                params.append(embedding_dimension)
            if embedding_index_version:
                filters.append("e.embedding_index_version = ?")
                params.append(embedding_index_version)
            rows = conn.execute(
                f"""
                select
                    c.chunk_id, c.document_id, c.document_name, c.text, c.page, c.metadata_json,
                    e.embedding_json, e.embedding_provider, e.embedding_model,
                    e.embedding_dimension, e.embedding_index_version
                from chunk_embeddings e
                join chunks c on c.chunk_id = e.chunk_id
                where {' and '.join(filters)}
                """,
                tuple(params),
            ).fetchall()
        vectors = []
        for row in rows:
            vectors.append(
                (
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        document_name=row["document_name"],
                        text=row["text"],
                        page=row["page"],
                        metadata=json.loads(row["metadata_json"] or "{}"),
                    ),
                    json.loads(row["embedding_json"] or "[]"),
                )
            )
        return vectors

    def get_local_vector_diagnostics(self, workspace_id: str, limit: int = 200) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select document_id, document_name, chunk_id, page, position, text, embedding_json,
                       embedding_provider, embedding_model, embedding_dimension, embedding_index_version
                from chunks
                where workspace_id = ?
                order by document_name, position asc
                limit ?
                """,
                (workspace_id, limit),
            ).fetchall()
        diagnostics = []
        for row in rows:
            vector = json.loads(row["embedding_json"] or "[]")
            norm = math.sqrt(sum(float(value) * float(value) for value in vector))
            diagnostics.append(
                {
                    "document_id": row["document_id"],
                    "document_name": row["document_name"],
                    "chunk_id": row["chunk_id"],
                    "page": row["page"],
                    "position": row["position"],
                    "text": row["text"],
                    "embedding": vector,
                    "embedding_provider": row["embedding_provider"] or "",
                    "embedding_model": row["embedding_model"] or "",
                    "embedding_dimension": row["embedding_dimension"],
                    "embedding_index_version": row["embedding_index_version"] or "",
                    "actual_dimension": len(vector),
                    "non_zero_count": sum(1 for value in vector if value),
                    "norm": round(norm, 6),
                }
            )
        return diagnostics

    def embedding_diagnostics(self, workspace_id: str, limit: int = 5) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select document_id, document_name, chunk_id, embedding_json,
                       embedding_provider, embedding_model, embedding_dimension, embedding_index_version
                from chunks
                where workspace_id = ?
                order by document_name, position asc
                limit ?
                """,
                (workspace_id, limit),
            ).fetchall()
        diagnostics = []
        for row in rows:
            vector = json.loads(row["embedding_json"] or "[]")
            non_zero = sum(1 for value in vector if value)
            norm = math.sqrt(sum(float(value) * float(value) for value in vector))
            diagnostics.append(
                {
                    "document_id": row["document_id"],
                    "document_name": row["document_name"],
                    "chunk_id": row["chunk_id"],
                    "dimension": len(vector),
                    "stored_dimension": row["embedding_dimension"],
                    "non_zero_count": non_zero,
                    "norm": round(norm, 6),
                    "embedding_provider": row["embedding_provider"] or "",
                    "embedding_model": row["embedding_model"] or "",
                    "embedding_index_version": row["embedding_index_version"] or "",
                }
            )
        return diagnostics

    def workspace_revision(self, workspace_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "select revision from workspace_revisions where workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        return str(row["revision"] if row is not None else 0)

    def get_conversation_context(self, session_id: str, workspace_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                """
                select payload_json from conversation_context
                where session_id = ? and workspace_id = ?
                """,
                (session_id, workspace_id),
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def set_conversation_context(self, session_id: str, workspace_id: str, payload: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into conversation_context (session_id, workspace_id, payload_json, updated_at)
                values (?, ?, ?, ?)
                on conflict(session_id, workspace_id) do update set
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    workspace_id,
                    json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def append_message(self, session_id: str, workspace_id: str, role: str, content: str) -> int:
        print(f"[CHAT] Appending message - Session: {session_id}, Role: {role}, Length: {len(content)} chars")
        logger.info("store.append_message.start session_id=%s workspace_id=%s role=%s content_length=%s", session_id, workspace_id, role, len(content))
        with self.connect() as conn:
            cursor = conn.execute(
                """
                insert into messages (session_id, workspace_id, role, content, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (session_id, workspace_id, role, content, datetime.now(timezone.utc).isoformat()),
            )
        logger.info("store.append_message.done session_id=%s status=ok", session_id)
        return int(cursor.lastrowid)

    def save_voice_transcript(
        self,
        *,
        message_id: int | None,
        session_id: str,
        workspace_id: str,
        transcript: TranscriptMetadata,
        audio_storage_path: str | None = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                insert into voice_transcripts
                (
                    message_id, session_id, workspace_id, transcript, confidence, language,
                    detected_language, language_confidence, timestamps_json, modality,
                    input_modality, audio_storage_path, started_at, ended_at, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    workspace_id,
                    transcript.text,
                    transcript.confidence,
                    transcript.language,
                    transcript.detected_language,
                    transcript.language_confidence,
                    json.dumps(transcript.timestamps),
                    transcript.modality,
                    transcript.input_modality,
                    audio_storage_path,
                    transcript.started_at.isoformat() if transcript.started_at else None,
                    transcript.ended_at.isoformat() if transcript.ended_at else None,
                    now,
                ),
            )
        logger.info("store.save_voice_transcript.done session_id=%s message_id=%s", session_id, message_id)
        return int(cursor.lastrowid)

    def create_message_attachment(
        self,
        *,
        attachment_id: str,
        conversation_id: str,
        workspace_id: str,
        attachment_type: str,
        filename: str,
        mime_type: str | None,
        file_size: int,
        storage_path: str,
        storage_bucket: str = "local-uploads",
        image_asset_id: str | None = None,
        document_id: str | None = None,
        user_id: str | None = None,
        processing_status: str = "completed",
        processing_error: str | None = None,
        expires_at: str | None = None,
        persistence_mode: str = "saved_compat",
    ) -> MessageAttachment:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                insert into message_attachments
                (
                    id, message_id, conversation_id, workspace_id, user_id, attachment_type,
                    filename, mime_type, file_size, storage_bucket, storage_path, image_asset_id,
                    document_id, processing_status, processing_error, created_at, expires_at, persistence_mode
                )
                values (?, null, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    conversation_id,
                    workspace_id,
                    user_id,
                    attachment_type,
                    filename,
                    mime_type,
                    file_size,
                    storage_bucket,
                    storage_path,
                    image_asset_id,
                    document_id,
                    processing_status,
                    processing_error,
                    now,
                    expires_at,
                    persistence_mode,
                ),
            )
        attachment = self.get_message_attachment(attachment_id)
        if attachment is None:
            raise RuntimeError("Attachment was not saved")
        return attachment

    def get_message_attachment(self, attachment_id: str) -> MessageAttachment | None:
        with self.connect() as conn:
            row = conn.execute("select * from message_attachments where id = ?", (attachment_id,)).fetchone()
        return self._attachment_from_row(row) if row is not None else None

    def list_message_attachments(
        self,
        *,
        attachment_ids: list[str],
        workspace_id: str,
        conversation_id: str,
    ) -> list[MessageAttachment]:
        if not attachment_ids:
            return []
        unique_ids = list(dict.fromkeys(attachment_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                select * from message_attachments
                where id in ({placeholders})
                  and workspace_id = ?
                  and conversation_id = ?
                """,
                (*unique_ids, workspace_id, conversation_id),
            ).fetchall()
        by_id = {row["id"]: self._attachment_from_row(row) for row in rows}
        return [by_id[item] for item in unique_ids if item in by_id]

    def require_message_attachments(
        self,
        *,
        attachment_ids: list[str],
        workspace_id: str,
        conversation_id: str,
    ) -> list[MessageAttachment]:
        requested = list(dict.fromkeys([item for item in attachment_ids if item]))
        attachments = self.list_message_attachments(
            attachment_ids=requested,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if len(attachments) != len(requested):
            raise PermissionError("Attachment is not available to this conversation.")
        return attachments

    def bind_attachments_to_message(
        self,
        *,
        attachment_ids: list[str],
        message_id: int,
        workspace_id: str,
        conversation_id: str,
    ) -> int:
        if not attachment_ids:
            return 0
        attachments = self.require_message_attachments(
            attachment_ids=attachment_ids,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        with self.connect() as conn:
            for attachment in attachments:
                conn.execute(
                    """
                    update message_attachments
                    set message_id = ?
                    where id = ? and workspace_id = ? and conversation_id = ?
                    """,
                    (message_id, attachment.id, workspace_id, conversation_id),
                )
        return len(attachments)

    def delete_message_attachment(self, attachment_id: str) -> MessageAttachment | None:
        attachment = self.get_message_attachment(attachment_id)
        if attachment is None:
            return None
        with self.connect() as conn:
            conn.execute("delete from message_attachments where id = ?", (attachment_id,))
        return attachment

    def delete_expired_temporary_attachments(self, now: datetime | None = None) -> list[MessageAttachment]:
        cutoff = (now or datetime.now(timezone.utc)).isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select * from message_attachments
                where persistence_mode = 'temporary'
                  and expires_at is not null
                  and expires_at <= ?
                """,
                (cutoff,),
            ).fetchall()
            attachments = [self._attachment_from_row(row) for row in rows]
            for attachment in attachments:
                conn.execute("delete from message_attachments where id = ?", (attachment.id,))
        for attachment in attachments:
            if attachment.document_id or attachment.image_asset_id:
                continue
            if attachment.storage_path:
                Path(attachment.storage_path).unlink(missing_ok=True)
        return attachments

    def load_messages(self, session_id: str, workspace_id: str, limit: int = 20) -> list[ChatMessage]:
        logger.info("store.load_messages.start session_id=%s workspace_id=%s limit=%s", session_id, workspace_id, limit)
        with self.connect() as conn:
            rows = conn.execute(
                """
                select role, content, created_at from messages
                where session_id = ? and workspace_id = ?
                order by id desc limit ?
                """,
                (session_id, workspace_id, limit),
            ).fetchall()
        messages = [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]
        print(f"[CHAT] Loaded {len(messages)} messages for session {session_id}")
        logger.info("store.load_messages.done session_id=%s message_count=%s", session_id, len(messages))
        return messages

    def delete_session(self, session_id: str, workspace_id: str) -> dict:
        logger.info("store.delete_session.start session_id=%s workspace_id=%s", session_id, workspace_id)
        with self.connect() as conn:
            message_count = conn.execute(
                "select count(*) from messages where session_id = ? and workspace_id = ?",
                (session_id, workspace_id),
            ).fetchone()[0]
            context_count = conn.execute(
                "select count(*) from conversation_context where session_id = ? and workspace_id = ?",
                (session_id, workspace_id),
            ).fetchone()[0]
            conn.execute("delete from messages where session_id = ? and workspace_id = ?", (session_id, workspace_id))
            conn.execute(
                "delete from message_attachments where conversation_id = ? and workspace_id = ?",
                (session_id, workspace_id),
            )
            conn.execute(
                "delete from conversation_context where session_id = ? and workspace_id = ?",
                (session_id, workspace_id),
            )
        logger.info(
            "store.delete_session.done session_id=%s workspace_id=%s messages=%s contexts=%s",
            session_id,
            workspace_id,
            message_count,
            context_count,
        )
        return {"messages": int(message_count), "contexts": int(context_count)}

    def _attachment_from_row(self, row: sqlite3.Row) -> MessageAttachment:
        return MessageAttachment(
            id=row["id"],
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            workspace_id=row["workspace_id"],
            user_id=row["user_id"],
            attachment_type=row["attachment_type"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            file_size=int(row["file_size"] or 0),
            storage_bucket=row["storage_bucket"] or "local-uploads",
            storage_path=row["storage_path"] or "",
            image_asset_id=row["image_asset_id"],
            document_id=row["document_id"],
            processing_status=row["processing_status"] or "completed",
            processing_error=row["processing_error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            persistence_mode=row["persistence_mode"] or "saved_compat",
        )

    def _bump_workspace_revision(self, conn: sqlite3.Connection, workspace_id: str) -> None:
        conn.execute(
            """
            insert into workspace_revisions (workspace_id, revision, updated_at)
            values (?, 1, ?)
            on conflict(workspace_id) do update set
                revision = revision + 1,
                updated_at = excluded.updated_at
            """,
            (workspace_id, datetime.now(timezone.utc).isoformat()),
        )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        rows = conn.execute(f"pragma table_info({table})").fetchall()
        if any(row["name"] == column for row in rows):
            return
        conn.execute(f"alter table {table} add column {column} {column_type}")

    def _chunk_position(self, chunk_id: str) -> int | None:
        try:
            return int(chunk_id.rsplit(":", 1)[1])
        except (IndexError, TypeError, ValueError):
            return None
