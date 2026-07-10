import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .schemas import ChatMessage, DocumentMetadata, RetrievedChunk


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
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
                    summary text
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
                    foreign key(document_id) references documents(document_id)
                );

                create table if not exists messages (
                    id integer primary key autoincrement,
                    session_id text not null,
                    workspace_id text not null,
                    role text not null,
                    content text not null,
                    created_at text not null
                );
                """
            )

    def save_document(
        self,
        *,
        document_id: str,
        workspace_id: str,
        filename: str,
        content_type: str | None,
        file_path: Path,
        summary: str | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into documents
                (document_id, workspace_id, filename, content_type, file_path, uploaded_at, summary)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    workspace_id,
                    filename,
                    content_type,
                    str(file_path),
                    datetime.utcnow().isoformat(),
                    summary,
                ),
            )

    def replace_chunks(
        self,
        chunks: list[RetrievedChunk],
        embeddings: list[list[float]],
        workspace_id: str,
    ) -> None:
        with self.connect() as conn:
            if chunks:
                conn.execute("delete from chunks where document_id = ?", (chunks[0].document_id,))
            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                conn.execute(
                    """
                    insert or replace into chunks
                    (chunk_id, document_id, workspace_id, document_name, page, position, text, embedding_json)
                    values (?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
            if chunks:
                conn.execute(
                    "update documents set chunk_count = ? where document_id = ?",
                    (len(chunks), chunks[0].document_id),
                )

    def list_documents(self, workspace_id: str = "default") -> list[DocumentMetadata]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from documents where workspace_id = ? order by uploaded_at desc",
                (workspace_id,),
            ).fetchall()
        return [
            DocumentMetadata(
                document_id=row["document_id"],
                workspace_id=row["workspace_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                chunk_count=row["chunk_count"],
                summary=row["summary"],
            )
            for row in rows
        ]

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        with self.connect() as conn:
            row = conn.execute("select * from documents where document_id = ?", (document_id,)).fetchone()
        if row is None:
            return None
        return DocumentMetadata(
            document_id=row["document_id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            chunk_count=row["chunk_count"],
            summary=row["summary"],
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
            )
            for row in rows
        ]

    def get_local_vectors(self, workspace_id: str) -> list[tuple[RetrievedChunk, list[float]]]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from chunks where workspace_id = ?",
                (workspace_id,),
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
                    ),
                    json.loads(row["embedding_json"] or "[]"),
                )
            )
        return vectors

    def workspace_revision(self, workspace_id: str) -> str:
        digest = hashlib.sha256()
        with self.connect() as conn:
            document_rows = conn.execute(
                """
                select document_id, filename, uploaded_at, chunk_count, coalesce(summary, '') as summary
                from documents
                where workspace_id = ?
                order by document_id
                """,
                (workspace_id,),
            ).fetchall()
            chunk_rows = conn.execute(
                """
                select chunk_id, document_id, position, coalesce(text, '') as text
                from chunks
                where workspace_id = ?
                order by document_id, position, chunk_id
                """,
                (workspace_id,),
            ).fetchall()

        for row in document_rows:
            digest.update(
                json.dumps(
                    {
                        "document_id": row["document_id"],
                        "filename": row["filename"],
                        "uploaded_at": row["uploaded_at"],
                        "chunk_count": row["chunk_count"],
                        "summary": row["summary"],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
        for row in chunk_rows:
            digest.update(
                json.dumps(
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "position": row["position"],
                        "text": row["text"],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
        return digest.hexdigest()

    def append_message(self, session_id: str, workspace_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into messages (session_id, workspace_id, role, content, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (session_id, workspace_id, role, content, datetime.utcnow().isoformat()),
            )

    def load_messages(self, session_id: str, workspace_id: str, limit: int = 20) -> list[ChatMessage]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select role, content, created_at from messages
                where session_id = ? and workspace_id = ?
                order by id desc limit ?
                """,
                (session_id, workspace_id, limit),
            ).fetchall()
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]
