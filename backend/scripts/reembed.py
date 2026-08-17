from __future__ import annotations

import argparse
import logging

from backend.caremind.config import get_settings
from backend.caremind.embeddings import EmbeddingClient
from backend.caremind.store import SQLiteStore
from backend.caremind.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CareMind document embeddings for the configured index.")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of documents to inspect.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override CAREMIND_EMBEDDING_BATCH_SIZE.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect work without writing vectors.")
    parser.add_argument("--resume", action="store_true", help="Skip chunks already embedded for the target index.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.batch_size:
        settings.embedding_batch_size = max(1, args.batch_size)
    store = SQLiteStore(settings.sqlite_path)
    embeddings = EmbeddingClient(settings)
    vectorstore = VectorStore(settings, store)

    documents = store.list_documents(args.workspace_id)
    if args.limit > 0:
        documents = documents[: args.limit]

    inspected = 0
    skipped = 0
    written = 0
    failed = 0

    for document in documents:
        inspected += 1
        chunks = store.get_document_chunks(document.document_id)
        if not chunks:
            skipped += 1
            continue
        existing = store.get_local_vectors(
            args.workspace_id,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            embedding_index_version=settings.embedding_index_version,
        )
        existing_chunk_ids = {chunk.chunk_id for chunk, _ in existing}
        if args.resume and all(chunk.chunk_id in existing_chunk_ids for chunk in chunks):
            skipped += 1
            continue
        if args.dry_run:
            logger.info(
                "reembed.dry_run document_id=%s chunk_count=%s provider=%s model=%s index_version=%s",
                document.document_id,
                len(chunks),
                settings.embedding_provider,
                settings.embedding_model,
                settings.embedding_index_version,
            )
            skipped += 1
            continue
        try:
            vectors = embeddings.embed_texts([chunk.text for chunk in chunks])
            vectorstore.upsert(
                chunks,
                vectors,
                args.workspace_id,
                embedding_provider=embeddings.last_provider,
                embedding_model=embeddings.last_model,
                embedding_index_version=settings.embedding_index_version,
            )
            written += 1
        except Exception as exc:
            failed += 1
            logger.exception(
                "reembed.document_failed document_id=%s error=%s",
                document.document_id,
                exc.__class__.__name__,
            )

    print(
        "reembed.summary "
        f"workspace_id={args.workspace_id} inspected={inspected} skipped={skipped} "
        f"written={written} failed={failed} provider={settings.embedding_provider} "
        f"model={settings.embedding_model} dimension={settings.embedding_dimension} "
        f"index_version={settings.embedding_index_version}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
