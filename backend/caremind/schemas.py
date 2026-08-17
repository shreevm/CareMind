from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None = None
    score: float | None = None
    quote: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page: int | None = None
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    document_id: str
    workspace_id: str
    filename: str
    content_type: str | None = None
    file_path: str | None = Field(default=None, exclude=True)
    uploaded_at: datetime
    chunk_count: int = 0
    summary: str | None = None
    source_image_id: str | None = None
    document_type: str | None = None
    understanding_confidence: float | None = None


class ImageMetadata(BaseModel):
    image_id: str
    workspace_id: str
    filename: str
    content_type: str | None = None
    file_path: str = Field(exclude=True)
    modality: str = "clinical_image"
    uploaded_at: datetime
    report_text_summary: str | None = None
    ocr_document_id: str | None = None


class MessageAttachment(BaseModel):
    id: str
    message_id: int | None = None
    conversation_id: str
    workspace_id: str
    user_id: str | None = None
    attachment_type: str
    filename: str
    mime_type: str | None = None
    file_size: int = 0
    storage_bucket: str = "local-uploads"
    storage_path: str = Field(exclude=True)
    image_asset_id: str | None = None
    document_id: str | None = None
    processing_status: str = "completed"
    processing_error: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    persistence_mode: Literal["temporary", "saved", "saved_compat"] = "saved_compat"


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class TranscriptMetadata(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    language: str | None = None
    detected_language: str | None = None
    language_confidence: float | None = Field(default=None, ge=0, le=1)
    timestamps: list[dict[str, Any]] = Field(default_factory=list)
    modality: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    input_modality: str | None = None


class ChatRequest(BaseModel):
    message: str
    attachment_ids: list[str] = Field(default_factory=list)
    session_id: str = "default"
    workspace_id: str = "default"
    top_k: int = Field(default=5, ge=1, le=12)
    modality: Literal["text", "voice", "image", "document"] = "text"
    transcript: TranscriptMetadata | None = None
    response_language: str | None = None
    response_mode: Literal["patient", "agent"] = "agent"
    debug: bool = False
    bypass_cache: bool = False


class ChatResponse(BaseModel):
    session_id: str
    route: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)


class CompareRequest(BaseModel):
    document_ids: list[str]
    workspace_id: str = "default"


class CompareResponse(BaseModel):
    summary: str
    citations: list[Citation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
