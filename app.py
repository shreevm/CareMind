from functools import lru_cache
from pathlib import Path
import time
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from caremind.agent import CareMindAgent
from caremind.config import Settings, get_settings
from caremind.embeddings import EmbeddingClient
from caremind.ingestion import DocumentIngestionService
from caremind.llm import LLMClient
from caremind.memory import ConversationMemory
from caremind.metrics import metrics
from caremind.safety import SafetyLayer
from caremind.schemas import ChatRequest, ChatResponse, CompareRequest, CompareResponse, SearchResponse
from caremind.store import SQLiteStore
from caremind.tools import DocumentTools
from caremind.vectorstore import VectorStore



security = HTTPBasic(auto_error=False)


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = SQLiteStore(settings.sqlite_path)
        self.embeddings = EmbeddingClient(settings)
        self.vectorstore = VectorStore(settings, self.store)
        self.ingestion = DocumentIngestionService(
            settings.upload_dir,
            self.store,
            self.embeddings,
            self.vectorstore,
        )
        self.tools = DocumentTools(self.embeddings, self.vectorstore, self.store)
        self.memory = ConversationMemory(settings, self.store)
        self.agent = CareMindAgent(
            tools=self.tools,
            llm=LLMClient(settings),
            memory=self.memory,
            safety=SafetyLayer(),
        )


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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if (
        credentials.username != settings.basic_auth_username
        or credentials.password != settings.basic_auth_password
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


app = FastAPI(
    title="CareMind API",
    version="0.4.0",
    description="Agentic RAG assistant for medical and research documents.",
    dependencies=[Depends(require_auth)],
)


@app.middleware("http")
async def collect_metrics(request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    metrics.record_request(
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=(time.perf_counter() - started_at) * 1000,
    )
    return response

web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index = web_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(index)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    services = get_services()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "vector_backend": "pinecone" if settings.should_use_pinecone else "local",
        "llm": settings.nvidia_chat_model if settings.nvidia_api_key else "local-grounded-fallback",
        "embeddings": settings.nvidia_embedding_model if settings.nvidia_api_key else "local-hashing-fallback",
        "redis": "connected" if services.memory.redis_enabled else "fallback-sqlite",
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.post("/upload")
async def upload_document(
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File(...)],
    workspace_id: Annotated[str, Form()] = "default",
):
    return await services.ingestion.ingest_upload(file, workspace_id=workspace_id)


@app.get("/documents")
def list_documents(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
):
    return services.store.list_documents(workspace_id)


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    services: Annotated[Services, Depends(get_services)],
) -> ChatResponse:
    return services.agent.answer(request)


@app.get("/search", response_model=SearchResponse)
def search(
    query: str,
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
    top_k: int = 5,
) -> SearchResponse:
    results = services.tools.document_search(query, workspace_id=workspace_id, top_k=top_k)
    return SearchResponse(query=query, results=results)


@app.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    services: Annotated[Services, Depends(get_services)],
) -> CompareResponse:
    return services.tools.compare_reports(request.document_ids, workspace_id=request.workspace_id)


@app.post("/demo/seed")
def seed_demo(
    services: Annotated[Services, Depends(get_services)],
    workspace_id: str = "default",
):
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
    return {"documents": [first, second]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
