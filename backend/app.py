from functools import lru_cache
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import secrets
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.caremind.agent import CareMindAgent
from backend.caremind.config import Settings, get_settings
from backend.caremind.embeddings import EmbeddingClient, EmbeddingServiceUnavailable
from backend.caremind.ingestion import DocumentIngestionService
from backend.caremind.ingestion import TextExtractionError, UploadValidationError
from backend.caremind.llm import LLMClient
from backend.caremind.memory import ConversationMemory
from backend.caremind.metrics import metrics
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest, ChatResponse, CompareRequest, CompareResponse, SearchResponse
from backend.caremind.speech import SpeechToTextService, SpeechTranscriptionError
from backend.caremind.store import SQLiteStore
from backend.caremind.tools import DocumentTools
from backend.caremind.validation import validate_request_middleware
from backend.caremind.vectorstore import VectorStore, VectorStoreUnavailable


logger = logging.getLogger(__name__)

security = HTTPBasic(auto_error=False)


class Services:
    def __init__(self, settings: Settings):
        print("\n" + "="*80)
        print("[INIT] CareMind Services Initialization")
        print("="*80)
        print(f"[INIT] App: {settings.app_name} v{settings.app_version}")
        print(f"[INIT] Environment: {settings.environment}")
        print(f"[INIT] Database: {settings.sqlite_path}")
        print(f"[INIT] Vector Backend: {settings.vector_backend}")
        print(f"[INIT] Embedding Provider: {settings.embedding_provider}")
        print(f"[INIT] Embedding Dimension: {settings.embedding_dimension}")
        print(f"[INIT] LLM Provider: NVIDIA" if settings.nvidia_api_key else "[INIT] LLM Provider: Local")
        print(f"[INIT] Medical LLM Provider: {settings.medical_llm_provider}")
        print(f"[INIT] Cache Enabled: Redis={settings.response_cache_enabled}")
        print("="*80 + "\n")
        logger.info(
            "services.init app=%s version=%s environment=%s vector_backend=%s embedding_provider=%s llm_provider=%s log_path=%s",
            settings.app_name,
            settings.app_version,
            settings.environment,
            settings.vector_backend,
            settings.embedding_provider,
            "nvidia" if settings.nvidia_api_key else "local",
            settings.log_path,
        )
        self.settings = settings
        self.store = SQLiteStore(settings.sqlite_path)
        self.embeddings = EmbeddingClient(settings)
        self.vectorstore = VectorStore(settings, self.store)
        self.llm = LLMClient(settings)
        self.ingestion = DocumentIngestionService(
            settings.upload_dir,
            self.store,
            self.embeddings,
            self.vectorstore,
            settings.max_upload_bytes,
            settings.allowed_upload_types,
            llm=self.llm,
        )
        self.tools = DocumentTools(self.embeddings, self.vectorstore, self.store)
        self.memory = ConversationMemory(settings, self.store)
        self.speech = SpeechToTextService(settings)
        self.agent = CareMindAgent(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            safety=SafetyLayer(),
        )
        print(f"[INIT] ✓ All services initialized successfully\n")


@lru_cache(maxsize=1)
def get_services() -> Services:
    return Services(get_settings())


def require_auth(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if not settings.basic_auth_username and not settings.basic_auth_password:
        return
    if credentials is None:
        raise_auth_error("Authentication required")
    username_ok = secrets.compare_digest(credentials.username, settings.basic_auth_username or "")
    password_ok = secrets.compare_digest(credentials.password, settings.basic_auth_password or "")
    if not (username_ok and password_ok):
        raise_auth_error("Invalid credentials")


def raise_auth_error(detail: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": 'Basic realm="CareMind"'},
    )


app = FastAPI(
    title="CareMind API",
    version=get_settings().app_version,
    description="Agentic RAG assistant for medical and research documents.",
    dependencies=[Depends(require_auth)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add validation middleware
app.middleware("http")(validate_request_middleware)


def _hide_agent_details(request: ChatRequest) -> bool:
    return request.response_mode == "patient" and not request.debug


def _authorized_attachment_ids(request: ChatRequest, services: Services) -> list[str]:
    attachment_ids = list(dict.fromkeys([item for item in request.attachment_ids if item]))
    if not attachment_ids:
        return []
    try:
        attachments = services.store.require_message_attachments(
            attachment_ids=attachment_ids,
            workspace_id=request.workspace_id,
            conversation_id=request.session_id,
        )
    except PermissionError as exc:
        logger.warning(
            "api.chat.attachment_unauthorized session_id=%s workspace_id=%s attachment_count=%s",
            request.session_id,
            request.workspace_id,
            len(attachment_ids),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Attachment is not available to this conversation.") from exc
    unsupported = [item for item in attachments if item.processing_status not in {"completed", "ready"}]
    if unsupported:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="One or more attachments are still processing.")
    return [item.id for item in attachments]


def _patient_safe_payload(payload: dict) -> dict:
    return {
        **payload,
        "route": "caremind_answer",
        "tool_calls": [],
        "trace": {},
    }


def _patient_safe_response(response: ChatResponse) -> ChatResponse:
    return response.model_copy(update={"route": "caremind_answer", "tool_calls": [], "trace": {}})


@app.exception_handler(VectorStoreUnavailable)
async def vector_store_unavailable_handler(_, exc: VectorStoreUnavailable) -> JSONResponse:
    logger.error("api.vector_store_unavailable detail=%s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "vector_backend": get_settings().vector_backend,
            "hint": "Check the configured vector backend credentials, schema, and network reachability.",
        },
    )


@app.exception_handler(EmbeddingServiceUnavailable)
async def embedding_service_unavailable_handler(_, exc: EmbeddingServiceUnavailable) -> JSONResponse:
    logger.error("api.embedding_service_unavailable detail=%s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "embedding_provider": get_settings().embedding_provider,
            "hint": "Check NVIDIA_API_KEY/NVIDIA_BASE_URL or explicitly enable local embeddings only for offline development.",
        },
    )


@app.exception_handler(UploadValidationError)
async def upload_validation_handler(_, exc: UploadValidationError) -> JSONResponse:
    logger.warning("api.upload_validation_failed detail=%s status_code=%s", exc, exc.status_code)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(TextExtractionError)
async def text_extraction_handler(_, exc: TextExtractionError) -> JSONResponse:
    logger.error("api.text_extraction_failed detail=%s", exc)
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)})


@app.exception_handler(SpeechTranscriptionError)
async def speech_transcription_handler(_, exc: SpeechTranscriptionError) -> JSONResponse:
    logger.warning("api.speech_transcription_failed detail=%s", exc)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})


@app.middleware("http")
async def collect_metrics(request, call_next):
    started_at = time.perf_counter()
    logger.info("http.request.start method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started_at) * 1000
        metrics.record_request(
            path=request.url.path,
            method=request.method,
            status_code=500,
            latency_ms=latency_ms,
        )
        logger.exception(
            "http.request.failed method=%s path=%s status_code=500 latency_ms=%.2f",
            request.method,
            request.url.path,
            latency_ms,
        )
        raise
    latency_ms = (time.perf_counter() - started_at) * 1000
    metrics.record_request(
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    logger.info(
        "http.request.done method=%s path=%s status_code=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response

web_dir = Path(__file__).resolve().parent.parent / "frontend"
next_out_dir = web_dir / "out"
if (next_out_dir / "_next").exists():
    app.mount("/_next", StaticFiles(directory=next_out_dir / "_next"), name="next_static")
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index = next_out_dir / "index.html" if (next_out_dir / "index.html").exists() else web_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(index)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    services = get_services()
    vector_store = services.vectorstore.status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "vector_backend": settings.vector_backend,
        "vector_backend_required": settings.vector_backend_requires_network,
        "vector_store": vector_store,
        "supabase": services.vectorstore.supabase_status(),
        "pinecone": services.vectorstore.pinecone_status(),
        "llm": settings.nvidia_chat_model if settings.nvidia_api_key else "local-grounded-fallback",
        "medical_llm": (
            settings.medical_hf_model
            if settings.medical_llm_provider.lower() == "transformers"
            else
            settings.medical_llm_model
            if settings.medical_llm_base_url and settings.medical_llm_model
            else "same-as-main-llm"
        ),
        "medical_llm_provider": settings.medical_llm_provider,
        "medical_llm_configured": bool(
            settings.medical_llm_provider.lower() == "transformers"
            or (settings.medical_llm_base_url and settings.medical_llm_model)
        ),
        "embedding_provider": settings.embedding_provider,
        "embedding_dimension": settings.embedding_dimension,
        "embedding_index_version": settings.embedding_index_version,
        "embedding_device": settings.embedding_device,
        "embedding_degraded": services.embeddings.degraded,
        "embedding_degraded_reason": services.embeddings.degraded_reason,
        "ocr": {
            "enabled": settings.image_ocr_enabled,
            "engine": settings.image_ocr_engine,
            "model": settings.nvidia_ocr_model if settings.image_ocr_engine.lower().strip() in {"auto", "nvidia"} else None,
            "nvidia_configured": bool(settings.nvidia_api_key),
        },
        "embeddings": (
            settings.embedding_model
            if settings.embedding_provider.lower() in {"qwen", "sentence-transformers", "sentence_transformers"}
            else settings.ollama_embedding_model
            if settings.embedding_provider.lower() == "ollama"
            else settings.nvidia_embedding_model
            if settings.nvidia_api_key
            else f"local-hashing-{settings.embedding_dimension}d"
        ),
        "redis": "connected" if services.memory.redis_enabled else "fallback-sqlite",
        "redis_configured": bool(settings.redis_url or settings.redis_host),
        "redis_error": services.memory.redis_error,
        "langsmith": "enabled" if settings.langsmith_enabled else "disabled",
        "langsmith_project": settings.langchain_project if settings.langsmith_enabled else None,
    }


@app.get("/metrics")
def get_metrics() -> Response:
    return Response(metrics.to_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/metrics.json")
def get_metrics_json() -> dict:
    return metrics.snapshot()


@app.get("/cache/debug")
def cache_debug(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
    session_id: str | None = None,
) -> dict:
    return services.memory.cache_debug_summary(workspace_id, session_id)


@app.post("/upload")
async def upload_document(
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File(...)],
    workspace_id: Annotated[str, Form()] = "default",
    session_id: Annotated[str, Form()] = "default",
    modality: Annotated[str, Form()] = "clinical_image",
    report_text: Annotated[str, Form()] = "",
):
    logger.info("api.upload.start filename=%s content_type=%s workspace_id=%s", file.filename, file.content_type, workspace_id)
    try:
        document = await services.ingestion.ingest_upload(
            file,
            workspace_id=workspace_id,
            modality=modality,
            report_text=report_text,
        )
    except Exception as exc:
        logger.exception(
            "api.upload.failed filename=%s content_type=%s workspace_id=%s error=%s",
            file.filename,
            file.content_type,
            workspace_id,
            exc,
        )
        raise
    services.memory.cache_clear_workspace(workspace_id)
    _activate_uploaded_asset(services, document, workspace_id=workspace_id, session_id=session_id)
    attachment = _create_chat_attachment(services, document, workspace_id=workspace_id, session_id=session_id)
    logger.info(
        "api.upload.done asset_id=%s attachment_id=%s filename=%s chunk_count=%s workspace_id=%s",
        getattr(document, "document_id", getattr(document, "image_id", "")),
        attachment.id,
        document.filename,
        getattr(document, "chunk_count", 0),
        workspace_id,
    )
    payload = document.model_dump(mode="json")
    payload["attachment_id"] = attachment.id
    payload["attachment"] = {
        "id": attachment.id,
        "attachment_type": attachment.attachment_type,
        "filename": attachment.filename,
        "mime_type": attachment.mime_type,
        "file_size": attachment.file_size,
        "image_asset_id": attachment.image_asset_id,
        "document_id": attachment.document_id,
        "processing_status": attachment.processing_status,
        "persistence_mode": attachment.persistence_mode,
        "created_at": attachment.created_at.isoformat(),
        "expires_at": attachment.expires_at.isoformat() if attachment.expires_at else None,
    }
    return payload


@app.post("/speech/transcribe")
async def transcribe_speech(
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File(...)],
    workspace_id: Annotated[str, Form()] = "default",
    session_id: Annotated[str, Form()] = "default",
    started_at: Annotated[str | None, Form()] = None,
    ended_at: Annotated[str | None, Form()] = None,
) -> dict:
    logger.info(
        "api.speech_transcribe.start filename=%s content_type=%s workspace_id=%s session_id=%s",
        file.filename,
        file.content_type,
        workspace_id,
        session_id,
    )
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio upload is empty.")

    def parse_dt(value: str | None):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    transcript = services.speech.transcribe(
        audio,
        started_at=parse_dt(started_at),
        ended_at=parse_dt(ended_at),
    )
    logger.info(
        "api.speech_transcribe.done chars=%s detected_language=%s language_confidence=%s",
        len(transcript.text),
        transcript.detected_language,
        transcript.language_confidence,
    )
    return transcript.model_dump(mode="json")


def _create_chat_attachment(services: Services, asset, *, workspace_id: str, session_id: str):
    file_path = Path(getattr(asset, "file_path", "") or "")
    if not file_path and hasattr(asset, "document_id"):
        stored_document = services.store.get_document(asset.document_id)
        if stored_document is not None:
            file_path = Path(getattr(stored_document, "file_path", "") or "")
    storage_path = str(file_path) if file_path else getattr(asset, "filename", "uploaded")
    file_size = file_path.stat().st_size if file_path.exists() else 0
    is_image = hasattr(asset, "image_id")
    return services.store.create_message_attachment(
        attachment_id=f"attachment_{uuid.uuid4()}",
        conversation_id=session_id or "default",
        workspace_id=workspace_id,
        attachment_type="image" if is_image else "document",
        filename=asset.filename,
        mime_type=getattr(asset, "content_type", None),
        file_size=file_size,
        storage_bucket=services.settings.attachment_storage_bucket,
        storage_path=storage_path,
        image_asset_id=getattr(asset, "image_id", None),
        document_id=getattr(asset, "ocr_document_id", None) if is_image else getattr(asset, "document_id", None),
        persistence_mode="saved_compat",
    )


def _activate_uploaded_asset(services: Services, asset, *, workspace_id: str, session_id: str) -> None:
    if not session_id:
        return
    context = services.memory.context_get(session_id, workspace_id)
    now = datetime.now(timezone.utc).isoformat()
    if hasattr(asset, "document_id"):
        understanding = services.store.get_document_understanding(asset.document_id) or {}
        raw_json = understanding.get("raw_json") if isinstance(understanding, dict) else {}
        patient = raw_json.get("patient", {}) if isinstance(raw_json, dict) else {}
        active_patient = patient.get("name") or context.get("active_patient") or ""
        entities = _merge_upload_entities(
            context.get("conversation_entities") or {},
            document_id=asset.document_id,
            document_name=asset.filename,
            patient_name=active_patient,
            raw_json=raw_json if isinstance(raw_json, dict) else {},
        )
        updated = {
            **context,
            "active_patient": active_patient,
            "active_document_id": asset.document_id,
            "active_document_name": asset.filename,
            "active_report": asset.filename,
            "active_document_ids": [asset.document_id],
            "current_patient": active_patient,
            "current_document": asset.filename,
            "current_report": asset.filename,
            "last_route": "upload",
            "last_tool": "DocumentUnderstandingAgent",
            "conversation_entities": entities,
            "document_memory": {
                **(context.get("document_memory") or {}),
                asset.document_id: {
                    "document_id": asset.document_id,
                    "document_name": asset.filename,
                    "document_type": getattr(asset, "document_type", None),
                    "summary": asset.summary or "",
                    "understanding_confidence": getattr(asset, "understanding_confidence", None),
                    "updated_at": now,
                },
            },
            "updated_at": now,
        }
        services.memory.context_set(session_id, workspace_id, updated)
        return
    if hasattr(asset, "image_id"):
        updated = {
            **context,
            "active_image_id": asset.image_id,
            "active_image_name": asset.filename,
            "current_image": asset.filename,
            "last_route": "upload",
            "last_tool": "DocumentUnderstandingAgent" if asset.ocr_document_id else "image_upload",
            "updated_at": now,
        }
        if asset.ocr_document_id:
            updated["active_document_id"] = asset.ocr_document_id
            updated["active_document_name"] = f"{asset.filename}.ocr.json"
            updated["active_report"] = updated["active_document_name"]
            updated["active_document_ids"] = [asset.ocr_document_id]
        services.memory.context_set(session_id, workspace_id, updated)


def _merge_upload_entities(
    entities: dict,
    *,
    document_id: str,
    document_name: str,
    patient_name: str,
    raw_json: dict,
) -> dict:
    merged = {key: list(value) for key, value in entities.items() if isinstance(value, list)}

    def add(kind: str, name: str, entity_id: str = "") -> None:
        if not name:
            return
        bucket = merged.setdefault(kind, [])
        if not any(str(item.get("name", "")).lower() == name.lower() for item in bucket if isinstance(item, dict)):
            bucket.append({"name": name, "id": entity_id})

    add("report", document_name, document_id)
    add("patient", patient_name, document_id)
    for item in raw_json.get("lab_values") or []:
        if isinstance(item, dict):
            add("lab_test", str(item.get("name") or ""), document_id)
    for item in raw_json.get("medications") or []:
        if isinstance(item, dict):
            add("medication", str(item.get("name") or ""), document_id)
    for item in raw_json.get("diagnoses") or []:
        add("disease", str(item), document_id)
    return merged


@app.get("/images")
def list_images(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
):
    print(f"[ROUTER] /images endpoint - Listing images from workspace: {workspace_id}")
    logger.info("api.list_images.start router=/images workspace_id=%s", workspace_id)
    images = services.store.list_images(workspace_id)
    logger.info("api.list_images.done router=/images workspace_id=%s count=%s", workspace_id, len(images))
    return images


@app.delete("/images/{image_id}")
def delete_image(
    image_id: str,
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
) -> dict:
    images = services.store.list_images(workspace_id)
    image = next((item for item in images if item.image_id == image_id), None)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in this workspace.")
    if image.ocr_document_id:
        chunks = services.store.get_document_chunks(image.ocr_document_id)
        services.vectorstore.delete_document(image.ocr_document_id, workspace_id, [chunk.chunk_id for chunk in chunks])
        ocr_deleted = services.store.delete_document(image.ocr_document_id)
        if ocr_deleted and ocr_deleted.get("file_path"):
            Path(ocr_deleted["file_path"]).unlink(missing_ok=True)
    deleted = services.store.delete_image_asset(image_id)
    if deleted and deleted.get("file_path"):
        Path(deleted["file_path"]).unlink(missing_ok=True)
    services.memory.cache_clear_workspace(workspace_id)
    logger.info("api.delete_image.done image_id=%s workspace_id=%s", image_id, workspace_id)
    return {"image_id": image_id, "workspace_id": workspace_id, "deleted": True}


@app.get("/documents")
def list_documents(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
):
    print(f"[ROUTER] /documents endpoint - Listing documents from workspace: {workspace_id}")
    logger.info("api.list_documents.start router=/documents workspace_id=%s", workspace_id)
    documents = services.store.list_documents(workspace_id)
    logger.info("api.list_documents.done router=/documents workspace_id=%s count=%s", workspace_id, len(documents))
    return documents


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
) -> dict:
    document = services.store.get_document(document_id)
    if document is None or document.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Document not found in this workspace.")
    chunks = services.store.get_document_chunks(document_id)
    services.vectorstore.delete_document(document_id, workspace_id, [chunk.chunk_id for chunk in chunks])
    deleted = services.store.delete_document(document_id)
    if deleted and deleted.get("file_path"):
        Path(deleted["file_path"]).unlink(missing_ok=True)
    services.memory.cache_clear_workspace(workspace_id)
    logger.info("api.delete_document.done document_id=%s workspace_id=%s", document_id, workspace_id)
    return {"document_id": document_id, "workspace_id": workspace_id, "deleted": True}


@app.delete("/cache")
def clear_cache(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
    session_id: str | None = None,
) -> dict:
    print(f"[ROUTER] /cache endpoint (DELETE) - Clearing workspace: {workspace_id}, Session: {session_id}")
    logger.info("api.clear_cache.start router=/cache workspace_id=%s session_id=%s", workspace_id, session_id)
    deleted = services.memory.cache_clear_workspace(workspace_id)
    session_deleted = services.memory.session_clear(session_id, workspace_id) if session_id else 0
    print(f"[ROUTER] /cache cleared {deleted} workspace items and {session_deleted} session items")
    logger.info("api.clear_cache.done router=/cache deleted=%s session_deleted=%s", deleted, session_deleted)
    return {
        "workspace_id": workspace_id,
        "session_id": session_id,
        "redis_enabled": services.memory.redis_enabled,
        "deleted": deleted,
        "session_deleted": session_deleted,
    }


@app.delete("/cache/{workspace_id}")
def clear_workspace_cache(
    workspace_id: str,
    services: Annotated[Services, Depends(get_services)],
    session_id: str | None = None,
) -> dict:
    print(f"[ROUTER] /cache/{{workspace_id}} endpoint (DELETE) - Clearing workspace: {workspace_id}")
    logger.info("api.clear_workspace_cache.start router=/cache/{workspace_id} workspace_id=%s session_id=%s", workspace_id, session_id)
    deleted = services.memory.cache_clear_workspace(workspace_id)
    session_deleted = services.memory.session_clear(session_id, workspace_id) if session_id else 0
    print(f"[ROUTER] /cache/{workspace_id} cleared {deleted} workspace items")
    logger.info("api.clear_workspace_cache.done router=/cache/{workspace_id} deleted=%s session_deleted=%s", deleted, session_deleted)
    return {
        "workspace_id": workspace_id,
        "session_id": session_id,
        "redis_enabled": services.memory.redis_enabled,
        "deleted": deleted,
        "session_deleted": session_deleted,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    services: Annotated[Services, Depends(get_services)],
) -> ChatResponse:
    print(f"[ROUTER] /chat endpoint - Session: {request.session_id}, Workspace: {request.workspace_id}, Query: {request.message[:100]}")
    attachment_ids = _authorized_attachment_ids(request, services)
    request = request.model_copy(update={"attachment_ids": attachment_ids})
    logger.info(
        "api.chat.start router=/chat session_id=%s workspace_id=%s query_length=%s attachment_count=%s",
        request.session_id,
        request.workspace_id,
        len(request.message),
        len(attachment_ids),
    )
    response = services.agent.answer(request)
    logger.info("api.chat.done router=/chat session_id=%s route=%s citations=%s", request.session_id, response.route, len(response.citations))
    return _patient_safe_response(response) if _hide_agent_details(request) else response


@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    services: Annotated[Services, Depends(get_services)],
) -> StreamingResponse:
    print(f"[ROUTER] /chat/stream endpoint - Session: {request.session_id}, Workspace: {request.workspace_id}, Query: {request.message[:100]}")
    attachment_ids = _authorized_attachment_ids(request, services)
    request = request.model_copy(update={"attachment_ids": attachment_ids})
    logger.info(
        "api.chat_stream.start router=/chat/stream session_id=%s workspace_id=%s query_length=%s attachment_count=%s",
        request.session_id,
        request.workspace_id,
        len(request.message),
        len(attachment_ids),
    )

    def events():
        try:
            for item in services.agent.stream_events(request):
                event = item.get("event", "message")
                if _hide_agent_details(request) and event in {"route", "metric"}:
                    continue
                payload = item.get("data", {})
                if _hide_agent_details(request) and event == "final":
                    payload = _patient_safe_payload(payload)
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: {event}\ndata: {data}\n\n"
        except Exception as exc:
            logger.exception("api.chat_stream.error router=/chat/stream session_id=%s", request.session_id)
            data = json.dumps({"error": "I could not generate a final answer. Please try again."}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"
        finally:
            logger.info("api.chat_stream.done router=/chat/stream session_id=%s", request.session_id)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/inspect")
def chat_inspect(
    request: ChatRequest,
    services: Annotated[Services, Depends(get_services)],
) -> dict:
    print(f"[ROUTER] /chat/inspect endpoint (DEBUG) - Session: {request.session_id}, Workspace: {request.workspace_id}")
    logger.info("api.chat_inspect.start router=/chat/inspect session_id=%s workspace_id=%s debug=true", request.session_id, request.workspace_id)
    attachment_ids = _authorized_attachment_ids(request, services)
    request = request.model_copy(update={"debug": True, "bypass_cache": True, "attachment_ids": attachment_ids})
    response = services.agent.answer(request)
    logger.info("api.chat_inspect.done router=/chat/inspect session_id=%s route=%s", request.session_id, response.route)
    return {
        "answer": response.answer,
        "route": response.route,
        "tool_calls": response.tool_calls,
        "citations": response.citations,
        "safety_notes": response.safety_notes,
        "trace": response.trace,
    }


@app.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
) -> dict:
    result = services.memory.session_delete(session_id, workspace_id)
    logger.info("api.delete_chat_session.done session_id=%s workspace_id=%s result=%s", session_id, workspace_id, result)
    return {"deleted": True, **result}


@app.get("/search", response_model=SearchResponse)
def search(
    query: str,
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
    top_k: int = 5,
) -> SearchResponse:
    print(f"[ROUTER] /search endpoint - Query: {query[:100]}, Workspace: {workspace_id}, Top-K: {top_k}")
    logger.info("api.search.start router=/search query=%r workspace_id=%s top_k=%s", query, workspace_id, top_k)
    results = services.tools.document_search(query, workspace_id=workspace_id, top_k=top_k)
    logger.info("api.search.done router=/search results_count=%s", len(results))
    print(f"[ROUTER] /search returned {len(results)} results")
    return SearchResponse(query=query, results=results)


@app.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    services: Annotated[Services, Depends(get_services)],
) -> CompareResponse:
    print(f"[ROUTER] /compare endpoint - Documents: {request.document_ids}, Workspace: {request.workspace_id}")
    logger.info("api.compare.start router=/compare workspace_id=%s document_ids=%s", request.workspace_id, request.document_ids)
    result = services.tools.compare_reports(request.document_ids, workspace_id=request.workspace_id)
    logger.info("api.compare.done router=/compare status=ok")
    return result


@app.post("/demo/seed")
def seed_demo(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
):
    print(f"[ROUTER] /demo/seed endpoint - Workspace: {workspace_id}")
    logger.info("api.demo_seed.start router=/demo/seed workspace_id=%s", workspace_id)
    first = services.ingestion.ingest_text(
        filename="synthetic-lab-report-baseline.txt",
        text=(
            "Synthetic medical report. Patient reports fatigue and mild shortness of breath. "
            "Hemoglobin is 11.2 g/dL. White blood cell count is 8.1. Chest x-ray shows no acute infiltrate. "
            "Plan notes iron studies and routine follow-up."
        ),
        workspace_id=workspace_id,
    )
    second = services.ingestion.ingest_text(
        filename="synthetic-lab-report-followup.txt",
        text=(
            "Synthetic follow-up report. Fatigue has improved. Hemoglobin is 12.6 g/dL. "
            "White blood cell count is 7.9. Chest x-ray remains without acute infiltrate. "
            "Plan notes continued monitoring and no urgent intervention documented."
        ),
        workspace_id=workspace_id,
    )
    services.memory.cache_clear_workspace(workspace_id)
    print(f"[ROUTER] /demo/seed created 2 synthetic documents successfully")
    logger.info("api.demo_seed.done router=/demo/seed workspace_id=%s status=ok documents_created=2", workspace_id)
    return {"documents": [first, second]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
