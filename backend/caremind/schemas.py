from datetime import datetime
from pydantic import BaseModel, Field


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


class DocumentMetadata(BaseModel):
    document_id: str
    workspace_id: str
    filename: str
    content_type: str | None = None
    uploaded_at: datetime
    chunk_count: int = 0
    summary: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    workspace_id: str = "default"
    top_k: int = Field(default=5, ge=1, le=12)


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
