import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from .embeddings import EmbeddingClient
from .schemas import DocumentMetadata, RetrievedChunk
from .store import SQLiteStore
from .vectorstore import VectorStore


class DocumentIngestionService:
    def __init__(
        self,
        upload_dir: Path,
        store: SQLiteStore,
        embeddings: EmbeddingClient,
        vectorstore: VectorStore,
    ):
        self.upload_dir = upload_dir
        self.store = store
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    async def ingest_upload(self, file: UploadFile, workspace_id: str = "default") -> DocumentMetadata:
        document_id = str(uuid.uuid4())
        safe_name = self._safe_filename(file.filename or "uploaded-document")
        document_dir = self.upload_dir / workspace_id
        document_dir.mkdir(parents=True, exist_ok=True)
        file_path = document_dir / f"{document_id}-{safe_name}"
        with file_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        extracted = self.extract_text(file_path)
        chunks = self.chunk_text(
            text=extracted,
            document_id=document_id,
            document_name=safe_name,
        )
        summary = self._summarize(extracted)
        self.store.save_document(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=safe_name,
            content_type=file.content_type,
            file_path=file_path,
            summary=summary,
        )
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        self.vectorstore.upsert(chunks, vectors, workspace_id)
        document = self.store.get_document(document_id)
        if document is None:
            raise RuntimeError("Document was not saved")
        return document

    def ingest_text(
        self,
        *,
        text: str,
        filename: str,
        workspace_id: str = "default",
        content_type: str = "text/plain",
    ) -> DocumentMetadata:
        document_id = str(uuid.uuid4())
        safe_name = self._safe_filename(filename)
        document_dir = self.upload_dir / workspace_id
        document_dir.mkdir(parents=True, exist_ok=True)
        file_path = document_dir / f"{document_id}-{safe_name}"
        file_path.write_text(text, encoding="utf-8")
        chunks = self.chunk_text(text=text, document_id=document_id, document_name=safe_name)
        self.store.save_document(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=safe_name,
            content_type=content_type,
            file_path=file_path,
            summary=self._summarize(text),
        )
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        self.vectorstore.upsert(chunks, vectors, workspace_id)
        document = self.store.get_document(document_id)
        if document is None:
            raise RuntimeError("Document was not saved")
        return document

    def extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                pages = []
                for index, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append(f"[Page {index}]\n{text}")
                return "\n\n".join(pages).strip()
            except Exception as exc:
                return f"Unable to extract PDF text: {exc}"
        return path.read_text(encoding="utf-8", errors="ignore")

    def chunk_text(
        self,
        *,
        text: str,
        document_id: str,
        document_name: str,
        chunk_size: int = 900,
        overlap: int = 120,
    ) -> list[RetrievedChunk]:
        cleaned = self._clean_extracted_text(text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not cleaned:
            cleaned = "No extractable text found."
        chunks: list[RetrievedChunk] = []
        start = 0
        index = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            if end < len(cleaned):
                boundary = max(cleaned.rfind("\n", start, end), cleaned.rfind(". ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunk_text = cleaned[start:end].strip()
            if chunk_text:
                page = self._page_for_chunk(chunk_text)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{document_id}:{index}",
                        document_id=document_id,
                        document_name=document_name,
                        text=chunk_text,
                        page=page,
                    )
                )
                index += 1
            if end >= len(cleaned):
                break
            start = max(0, end - overlap)
        return chunks

    def _page_for_chunk(self, text: str) -> int | None:
        match = re.search(r"\[Page (\d+)\]", text)
        return int(match.group(1)) if match else None

    def _summarize(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", self._clean_extracted_text(text)).strip()
        return compact[:500] if compact else "No extractable text."

    def _safe_filename(self, filename: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "document.txt"

    def _clean_extracted_text(self, text: str) -> str:
        replacements = {
            "\u00ae": "",
            "\u00a9": "",
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
            "Â®": "",
            "â€“": "-",
        }
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        boilerplate_patterns = [
            r"The content in this report is not intended to be a substitute for professional medical advice,\s*diagnosis,\s*or treatment\.",
            r"Always seek the\s+advice of your physician or other qualified health provider with any questions you may have regarding a medical condition\.",
            r"Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition\.",
            r"Note:\s*(?=\[Page|\n|$)",
        ]
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        return re.sub(r"[ \t]{2,}", " ", cleaned)
