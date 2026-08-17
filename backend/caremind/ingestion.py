import logging
import json
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from .agents.document_understanding import DocumentUnderstandingAgent, DocumentUnderstandingResult
from .embeddings import EmbeddingClient
from .ocr import ClinicalTextStructurer, ImageOCRService, OCR_DERIVED_CONTENT_TYPE
from .schemas import DocumentMetadata, ImageMetadata, RetrievedChunk
from .store import SQLiteStore
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """Raised when an upload is too large or not an accepted type."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class TextExtractionError(RuntimeError):
    """Raised when text extraction fails for an uploaded file."""


class DocumentIngestionService:
    def __init__(
        self,
        upload_dir: Path,
        store: SQLiteStore,
        embeddings: EmbeddingClient,
        vectorstore: VectorStore,
        max_upload_bytes: int,
        allowed_content_types: set[str],
        ocr: ImageOCRService | None = None,
        structurer: ClinicalTextStructurer | None = None,
        llm: Any | None = None,
    ):
        self.upload_dir = upload_dir
        self.store = store
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.max_upload_bytes = max_upload_bytes
        self.allowed_content_types = allowed_content_types
        self.ocr = ocr or ImageOCRService(embeddings.settings)
        self.structurer = structurer or ClinicalTextStructurer(embeddings.settings)
        self.llm = llm
        self.document_understanding = DocumentUnderstandingAgent(embeddings.settings, llm=llm)

    async def ingest_upload(
        self,
        file: UploadFile,
        workspace_id: str = "default",
        modality: str = "clinical_image",
        report_text: str = "",
    ) -> DocumentMetadata | ImageMetadata:
        document_id = str(uuid.uuid4())
        safe_name = self._safe_filename(file.filename or "uploaded-document")
        content_type = self._normalized_content_type(file.content_type, safe_name)
        if self._is_image_content_type(content_type):
            return await self.ingest_image_upload(
                file,
                workspace_id=workspace_id,
                modality=modality,
                report_text=report_text,
            )
        logger.info(
            "upload.start document_id=%s workspace_id=%s filename=%s content_type=%s",
            document_id,
            workspace_id,
            safe_name,
            content_type,
        )
        if content_type not in self.allowed_content_types:
            allowed = ", ".join(sorted(self.allowed_content_types))
            logger.warning(
                "upload.rejected document_id=%s reason=unsupported_content_type content_type=%s allowed=%s",
                document_id,
                content_type,
                allowed,
            )
            raise UploadValidationError(
                f"Unsupported upload type '{content_type}'. Allowed types: {allowed}.",
                status_code=415,
            )
        document_dir = self.upload_dir / workspace_id
        document_dir.mkdir(parents=True, exist_ok=True)
        file_path = document_dir / f"{document_id}-{safe_name}"
        bytes_written = 0
        too_large = False
        logger.info("upload.write_file.start document_id=%s path=%s", document_id, file_path)
        with file_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > self.max_upload_bytes:
                    too_large = True
                    break
                output.write(chunk)
        if too_large:
            file_path.unlink(missing_ok=True)
            logger.warning(
                "upload.rejected document_id=%s reason=too_large bytes=%s max_bytes=%s",
                document_id,
                bytes_written,
                self.max_upload_bytes,
            )
            raise UploadValidationError(
                f"Upload exceeds maximum size of {self.max_upload_bytes} bytes.",
                status_code=413,
            )
        logger.info("upload.write_file.done document_id=%s bytes=%s", document_id, bytes_written)
        print(f"[INGEST] ✓ File written: {bytes_written} bytes")

        logger.info("upload.extract.start document_id=%s", document_id)
        print(f"[INGEST] Extracting text...")
        extracted = self.extract_text(file_path)
        logger.info("upload.extract.done document_id=%s chars=%s", document_id, len(extracted))
        print(f"[INGEST] ✓ Text extracted: {len(extracted)} characters")
        
        understanding: DocumentUnderstandingResult | None = None
        indexed_text = extracted
        chunk_metadata = None
        if self.embeddings.settings.document_understanding_enabled:
            understanding = self.document_understanding.understand(
                document_id=document_id,
                workspace_id=workspace_id,
                filename=safe_name,
                content_type=content_type,
                file_path=file_path,
                extracted_text=extracted,
                modality="document",
            )
            indexed_text = understanding.index_text
            chunk_metadata = self._document_understanding_metadata(understanding)
        chunks = self._chunks_for_understanding(
            understanding=understanding,
            fallback_text=indexed_text,
            document_id=document_id,
            document_name=safe_name,
            metadata=chunk_metadata,
        )
        logger.info("upload.chunk.done document_id=%s chunks=%s", document_id, len(chunks))
        print(f"[INGEST] ✓ Created {len(chunks)} chunks")
        
        summary = understanding.summary if understanding is not None else self._summarize(extracted)
        logger.info("upload.sqlite.document.start document_id=%s", document_id)
        self.store.save_document(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=safe_name,
            content_type=content_type,
            file_path=file_path,
            summary=summary,
            document_type=understanding.document_type if understanding is not None else None,
            understanding_confidence=understanding.confidence if understanding is not None else None,
        )
        logger.info("upload.sqlite.document.done document_id=%s", document_id)
        print(f"[INGEST] ✓ Document metadata saved to database")
        
        logger.info("upload.embedding.start document_id=%s chunks=%s", document_id, len(chunks))
        print(f"[INGEST] Computing embeddings for {len(chunks)} chunks...")
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        dimension = len(vectors[0]) if vectors else 0
        logger.info(
            "upload.embedding.done document_id=%s vectors=%s dimension=%s provider=%s model=%s",
            document_id,
            len(vectors),
            dimension,
            self.embeddings.last_provider,
            self.embeddings.last_model,
        )
        print(f"[INGEST] ✓ Embeddings computed - Provider: {self.embeddings.last_provider}, Model: {self.embeddings.last_model}, Dimension: {dimension}")
        
        try:
            logger.info("upload.vectorstore.upsert.start document_id=%s backend=%s", document_id, self.vectorstore.settings.vector_backend)
            print(f"[INGEST] Storing vectors in {self.vectorstore.settings.vector_backend} backend...")
            self.vectorstore.upsert(
                chunks,
                vectors,
                workspace_id,
                embedding_provider=self.embeddings.last_provider,
                embedding_model=self.embeddings.last_model,
                embedding_index_version=self.embeddings.settings.embedding_index_version,
            )
            if understanding is not None:
                self._store_document_understanding(understanding)
            logger.info("upload.vectorstore.upsert.done document_id=%s", document_id)
            print(f"[INGEST] ✓ Vectors stored in vectorstore")
        except Exception as exc:
            logger.exception(
                "upload.vectorstore.upsert.failed document_id=%s error=%s",
                document_id,
                exc,
            )
            self.store.delete_document(document_id)
            if file_path.exists():
                file_path.unlink()
            logger.info("upload.cleanup.done document_id=%s", document_id)
            raise
        document = self.store.get_document(document_id)
        if document is None:
            raise RuntimeError("Document was not saved")
        logger.info("upload.done document_id=%s chunk_count=%s", document_id, document.chunk_count)
        print(f"[INGEST] ✓ Upload complete - Document ID: {document_id}, Chunks: {document.chunk_count}\n")
        return document

    async def ingest_image_upload(
        self,
        file: UploadFile,
        workspace_id: str = "default",
        modality: str = "clinical_image",
        report_text: str = "",
    ) -> ImageMetadata:
        image_id = str(uuid.uuid4())
        safe_name = self._safe_filename(file.filename or "uploaded-image")
        content_type = self._normalized_content_type(file.content_type, safe_name)
        logger.info(
            "image_upload.start image_id=%s workspace_id=%s filename=%s content_type=%s modality=%s",
            image_id,
            workspace_id,
            safe_name,
            content_type,
            modality,
        )
        if not self._is_image_content_type(content_type) or content_type not in self.allowed_content_types:
            allowed = ", ".join(sorted(item for item in self.allowed_content_types if item.startswith("image/")))
            raise UploadValidationError(
                f"Unsupported image upload type '{content_type}'. Allowed image types: {allowed or 'none configured'}.",
                status_code=415,
            )
        image_dir = self.upload_dir / workspace_id / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        file_path = image_dir / f"{image_id}-{safe_name}"
        bytes_written = 0
        too_large = False
        logger.info("image_upload.write_file.start image_id=%s path=%s", image_id, file_path)
        with file_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > self.max_upload_bytes:
                    too_large = True
                    break
                output.write(chunk)
        if too_large:
            file_path.unlink(missing_ok=True)
            raise UploadValidationError(
                f"Upload exceeds maximum size of {self.max_upload_bytes} bytes.",
                status_code=413,
            )
        logger.info("image_upload.write_file.done image_id=%s bytes=%s", image_id, bytes_written)
        report_text_summary = self._summarize(report_text) if report_text.strip() else None
        logger.info("image_upload.ocr.start image_id=%s engine=%s", image_id, self.ocr.__class__.__name__)
        ocr_result = self.ocr.extract(file_path)
        logger.info(
            "image_upload.ocr.done image_id=%s engine=%s chars=%s confidence=%s warnings=%s",
            image_id,
            ocr_result.engine,
            len(ocr_result.text),
            ocr_result.confidence,
            ocr_result.warnings,
        )
        source_text = self._image_grounding_text(report_text, ocr_result.text)
        ocr_document: DocumentMetadata | None = None
        logger.info("image_upload.sqlite.asset.start image_id=%s", image_id)
        self.store.save_image_asset(
            image_id=image_id,
            workspace_id=workspace_id,
            filename=safe_name,
            content_type=content_type,
            file_path=file_path,
            modality=modality or "clinical_image",
            report_text_summary=report_text_summary or (self._summarize(source_text) if source_text.strip() else None),
        )
        logger.info("image_upload.sqlite.asset.done image_id=%s", image_id)
        if source_text.strip():
            try:
                logger.info("image_upload.ocr_index.start image_id=%s chars=%s", image_id, len(source_text))
                ocr_document_id = str(uuid.uuid4())
                understanding: DocumentUnderstandingResult | None = None
                if self.embeddings.settings.document_understanding_enabled:
                    understanding = self.document_understanding.understand(
                        document_id=ocr_document_id,
                        workspace_id=workspace_id,
                        filename=f"{safe_name}.ocr.json",
                        content_type=OCR_DERIVED_CONTENT_TYPE,
                        file_path=file_path,
                        extracted_text=source_text,
                        source_image_id=image_id,
                        modality=modality or "clinical_image",
                        ocr_confidence=ocr_result.confidence,
                    )
                    indexed_text = understanding.index_text
                    understanding_metadata = self._document_understanding_metadata(understanding)
                else:
                    structured = self.structurer.structure(
                        text=source_text,
                        filename=safe_name,
                        modality=modality or "clinical_image",
                        source="user_report_text" if report_text.strip() and not ocr_result.text.strip() else "ocr",
                        llm=self.llm,
                    )
                    indexed_text = self.structurer.to_index_text(structured)
                    understanding_metadata = {}
                ocr_document = self.ingest_text(
                    text=indexed_text,
                    filename=f"{safe_name}.ocr.json",
                    workspace_id=workspace_id,
                    content_type=OCR_DERIVED_CONTENT_TYPE,
                    source_image_id=image_id,
                    document_id=ocr_document_id,
                    understanding=understanding,
                    metadata={
                        **understanding_metadata,
                        "source": "ocr",
                        "image_id": image_id,
                        "image_filename": safe_name,
                        "ocr_engine": ocr_result.engine,
                        "ocr_confidence": ocr_result.confidence,
                        "ocr_warnings": ocr_result.warnings,
                    },
                )
                self.store.link_image_ocr_document(
                    image_id=image_id,
                    ocr_document_id=ocr_document.document_id,
                    report_text_summary=self._summarize(source_text),
                )
                logger.info(
                    "image_upload.ocr_index.done image_id=%s document_id=%s chunks=%s",
                    image_id,
                    ocr_document.document_id,
                    ocr_document.chunk_count,
                )
            except Exception as exc:
                logger.exception(
                    "image_upload.ocr_index.failed image_id=%s filename=%s error=%s",
                    image_id,
                    safe_name,
                    exc,
                )
        image = next((item for item in self.store.list_images(workspace_id) if item.image_id == image_id), None)
        if image is None:
            logger.error("image_upload.missing_saved_asset image_id=%s workspace_id=%s", image_id, workspace_id)
            raise RuntimeError("Image was not saved")
        logger.info("image_upload.done image_id=%s bytes=%s workspace_id=%s", image_id, bytes_written, workspace_id)
        return image

    def ingest_text(
        self,
        *,
        text: str,
        filename: str,
        workspace_id: str = "default",
        content_type: str = "text/plain",
        source_image_id: str | None = None,
        document_id: str | None = None,
        understanding: DocumentUnderstandingResult | None = None,
        metadata: dict | None = None,
    ) -> DocumentMetadata:
        document_id = document_id or str(uuid.uuid4())
        safe_name = self._safe_filename(filename)
        logger.info(
            "ingest_text.start document_id=%s workspace_id=%s filename=%s content_type=%s",
            document_id,
            workspace_id,
            safe_name,
            content_type,
        )
        document_dir = self.upload_dir / workspace_id
        document_dir.mkdir(parents=True, exist_ok=True)
        file_path = document_dir / f"{document_id}-{safe_name}"
        file_path.write_text(self._stored_text_payload(text, content_type), encoding="utf-8")
        if understanding is None and self.embeddings.settings.document_understanding_enabled:
            understanding = self.document_understanding.understand(
                document_id=document_id,
                workspace_id=workspace_id,
                filename=safe_name,
                content_type=content_type,
                file_path=file_path,
                extracted_text=text,
                source_image_id=source_image_id,
                modality="document",
            )
            text = understanding.index_text
        chunk_metadata = {**(metadata or {})}
        if understanding is not None:
            understanding_metadata = self._document_understanding_metadata(understanding)
            if "source" in chunk_metadata:
                understanding_metadata.pop("source", None)
            chunk_metadata.update(understanding_metadata)
        chunks = self._chunks_for_understanding(
            understanding=understanding,
            fallback_text=text,
            document_id=document_id,
            document_name=safe_name,
            metadata=chunk_metadata,
        )
        logger.info("ingest_text.chunk.done document_id=%s chunks=%s", document_id, len(chunks))
        self.store.save_document(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=safe_name,
            content_type=content_type,
            file_path=file_path,
            summary=understanding.summary if understanding is not None else self._summarize(text),
            source_image_id=source_image_id,
            document_type=understanding.document_type if understanding is not None else None,
            understanding_confidence=understanding.confidence if understanding is not None else None,
        )
        logger.info("ingest_text.sqlite.document.done document_id=%s", document_id)
        logger.info("ingest_text.embedding.start document_id=%s chunks=%s", document_id, len(chunks))
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        logger.info(
            "ingest_text.embedding.done document_id=%s vectors=%s dimension=%s provider=%s model=%s",
            document_id,
            len(vectors),
            len(vectors[0]) if vectors else 0,
            self.embeddings.last_provider,
            self.embeddings.last_model,
        )
        try:
            logger.info("ingest_text.vectorstore.upsert.start document_id=%s backend=%s", document_id, self.vectorstore.settings.vector_backend)
            self.vectorstore.upsert(
                chunks,
                vectors,
                workspace_id,
                embedding_provider=self.embeddings.last_provider,
                embedding_model=self.embeddings.last_model,
                embedding_index_version=self.embeddings.settings.embedding_index_version,
            )
            if understanding is not None:
                self._store_document_understanding(understanding)
            logger.info("ingest_text.vectorstore.upsert.done document_id=%s", document_id)
        except Exception as exc:
            logger.exception(
                "ingest_text.vectorstore.upsert.failed document_id=%s error=%s",
                document_id,
                exc,
            )
            self.store.delete_document(document_id)
            if file_path.exists():
                file_path.unlink()
            logger.info("ingest_text.cleanup.done document_id=%s", document_id)
            raise
        document = self.store.get_document(document_id)
        if document is None:
            raise RuntimeError("Document was not saved")
        logger.info("ingest_text.done document_id=%s chunk_count=%s", document_id, document.chunk_count)
        return document

    def extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                logger.info("extract.pdf.pages path=%s pages=%s", path, len(reader.pages))
                pages = []
                for index, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append(f"[Page {index}]\n{text}")
                extracted = "\n\n".join(pages).strip()
                if not extracted:
                    raise TextExtractionError("PDF contains no extractable text.")
                return extracted
            except TextExtractionError:
                raise
            except Exception as exc:
                raise TextExtractionError(f"Unable to extract PDF text: {exc}") from exc
        return path.read_text(encoding="utf-8", errors="ignore")

    def chunk_text(
        self,
        *,
        text: str,
        document_id: str,
        document_name: str,
        chunk_size: int = 900,
        overlap: int = 120,
        metadata: dict | None = None,
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
                        metadata=metadata or {},
                    )
                )
                index += 1
            if end >= len(cleaned):
                break
            start = max(0, end - overlap)
        return chunks

    def _chunks_for_understanding(
        self,
        *,
        understanding: DocumentUnderstandingResult | None,
        fallback_text: str,
        document_id: str,
        document_name: str,
        metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        if understanding is not None and understanding.semantic_chunks:
            chunks: list[RetrievedChunk] = []
            for index, item in enumerate(understanding.semantic_chunks):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                text = str(item.get("text") or "").strip()
                if not title or not text:
                    continue
                item_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                chunk_metadata = {**(metadata or {}), **item_metadata}
                chunk_metadata.setdefault("chunk_title", title)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{document_id}:{len(chunks)}",
                        document_id=document_id,
                        document_name=document_name,
                        text=text,
                        metadata=chunk_metadata,
                    )
                )
            if chunks:
                return chunks
        return self.chunk_text(text=fallback_text, document_id=document_id, document_name=document_name, metadata=metadata)

    def _page_for_chunk(self, text: str) -> int | None:
        match = re.search(r"\[Page (\d+)\]", text)
        return int(match.group(1)) if match else None

    def _summarize(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", self._clean_extracted_text(text)).strip()
        return compact[:500] if compact else "No extractable text."

    def _image_grounding_text(self, report_text: str, ocr_text: str) -> str:
        parts = []
        if report_text.strip():
            parts.append(f"User-provided paired report or caption:\n{report_text.strip()}")
        if ocr_text.strip():
            parts.append(f"OCR text extracted from uploaded image:\n{ocr_text.strip()}")
        return "\n\n".join(parts).strip()

    def _stored_text_payload(self, text: str, content_type: str) -> str:
        if content_type != OCR_DERIVED_CONTENT_TYPE:
            return text
        if self._looks_like_json(text):
            try:
                return json.dumps(json.loads(text), indent=2)
            except json.JSONDecodeError:
                pass
        return json.dumps({"text": text}, indent=2)

    def _store_document_understanding(self, understanding: DocumentUnderstandingResult) -> None:
        self.store.save_document_understanding(
            document_id=understanding.document_id,
            workspace_id=understanding.workspace_id,
            filename=understanding.filename,
            content_type=understanding.content_type,
            document_type=understanding.document_type,
            summary=understanding.summary,
            confidence=understanding.confidence,
            model_name=understanding.model_name,
            extraction_source=understanding.extraction_source,
            raw_json=understanding.raw_json,
            index_text=understanding.index_text,
        )

    def _document_understanding_metadata(self, understanding: DocumentUnderstandingResult) -> dict[str, Any]:
        return {
            "source": "document_understanding",
            "document_type": understanding.document_type,
            "understanding_confidence": understanding.confidence,
            "understanding_model": understanding.model_name,
            "understanding_source": understanding.extraction_source,
        }

    def _looks_like_json(self, text: str) -> bool:
        stripped = text.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))

    def _safe_filename(self, filename: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "document.txt"

    def _normalized_content_type(self, content_type: str | None, filename: str) -> str:
        if content_type:
            return content_type.split(";")[0].strip().lower()
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        if suffix in {".md", ".markdown"}:
            return "text/markdown"
        if suffix in {".png"}:
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix in {".webp"}:
            return "image/webp"
        return "text/plain"

    def _is_image_content_type(self, content_type: str) -> bool:
        return content_type.startswith("image/")

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
