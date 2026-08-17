from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CareMind"
    app_version: str = "0.6.0"
    environment: str = Field(default="local", alias="CAREMIND_ENV")

    data_dir: Path = Field(default=Path("data"), alias="CAREMIND_DATA_DIR")
    upload_dir: Path = Field(default=Path("data/uploads"), alias="CAREMIND_UPLOAD_DIR")
    attachment_storage_bucket: str = Field(default="local-uploads", alias="CAREMIND_ATTACHMENT_STORAGE_BUCKET")
    temporary_attachment_ttl_seconds: int = Field(default=7 * 24 * 3600, alias="CAREMIND_TEMP_ATTACHMENT_TTL_SECONDS")
    sqlite_path: Path = Field(default=Path("data/caremind.db"), alias="CAREMIND_SQLITE_PATH")
    log_path: Path = Field(default=Path("data/caremind-debug.log"), alias="CAREMIND_LOG_PATH")
    log_level: str = Field(default="INFO", alias="CAREMIND_LOG_LEVEL")
    capture_prints: bool = Field(default=True, alias="CAREMIND_CAPTURE_PRINTS")

    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    nvidia_chat_model: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b",
        alias="NVIDIA_CHAT_MODEL",
    )
    nvidia_reasoning_enabled: bool = Field(default=True, alias="NVIDIA_REASONING_ENABLED")
    nvidia_reasoning_budget: int | None = Field(default=2048, alias="NVIDIA_REASONING_BUDGET")
    nvidia_enable_thinking: bool | None = Field(default=None, alias="NVIDIA_ENABLE_THINKING")
    nvidia_embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", alias="NVIDIA_EMBEDDING_MODEL")
    nvidia_query_input_type: str = Field(default="query", alias="NVIDIA_QUERY_INPUT_TYPE")
    nvidia_document_input_type: str = Field(default="passage", alias="NVIDIA_DOCUMENT_INPUT_TYPE")
    embedding_provider: str = Field(default="qwen", alias="CAREMIND_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="CAREMIND_EMBEDDING_MODEL")
    embedding_index_version: str = Field(default="qwen3-0.6b-v1-1024", alias="CAREMIND_EMBEDDING_INDEX_VERSION")
    embedding_batch_size: int = Field(default=4, alias="CAREMIND_EMBEDDING_BATCH_SIZE")
    embedding_device: str = Field(default="auto", alias="CAREMIND_EMBEDDING_DEVICE")
    embedding_normalize: bool = Field(default=True, alias="CAREMIND_EMBEDDING_NORMALIZE")
    local_embedding_fallback_enabled: bool = Field(default=True, alias="CAREMIND_LOCAL_EMBEDDING_FALLBACK_ENABLED")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_embedding_model: str | None = Field(default=None, alias="OLLAMA_EMBEDDING_MODEL")
    medical_llm_base_url: str | None = Field(default=None, alias="MEDICAL_LLM_BASE_URL")
    medical_llm_api_key: str | None = Field(default=None, alias="MEDICAL_LLM_API_KEY")
    medical_llm_model: str | None = Field(default=None, alias="MEDICAL_LLM_MODEL")
    medical_llm_provider: str = Field(default="openai-compatible", alias="MEDICAL_LLM_PROVIDER")
    medical_hf_model: str = Field(
        default="EpistemeAI/Reasoning-Medical0.1-27B",
        alias="MEDICAL_HF_MODEL",
    )
    llm_temperature: float = Field(default=0.0, alias="CAREMIND_LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, alias="CAREMIND_LLM_TOP_P")
    llm_max_tokens: int = Field(default=2048, alias="CAREMIND_LLM_MAX_TOKENS")
    llm_frequency_penalty: float = Field(default=0.0, alias="CAREMIND_LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(default=0.0, alias="CAREMIND_LLM_PRESENCE_PENALTY")
    llm_repetition_penalty: float | None = Field(default=None, alias="CAREMIND_LLM_REPETITION_PENALTY")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="caremind-index", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    supabase_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
    )
    supabase_publishable_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPABASE_PUBLISHABLE_KEY",
            "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
            "SUPABASE_ANON_KEY",
        ),
    )
    supabase_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
    )
    supabase_jwks_url: str | None = Field(default=None, alias="SUPABASE_JWKS_URL")
    supabase_db_url: str | None = Field(default=None, alias="SUPABASE_DB_URL")
    supabase_vector_table: str = Field(default="caremind_document_chunks_qwen3_1024", alias="SUPABASE_VECTOR_TABLE")
    supabase_match_function: str = Field(default="match_caremind_document_chunks_qwen3_1024", alias="SUPABASE_MATCH_FUNCTION")
    vector_backend: str = Field(default="supabase", alias="CAREMIND_VECTOR_BACKEND")
    embedding_dimension: int = Field(default=1024, alias="CAREMIND_EMBEDDING_DIM")

    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    session_ttl_seconds: int = Field(default=3600, alias="CAREMIND_SESSION_TTL")
    response_cache_enabled: bool = Field(default=True, alias="CAREMIND_RESPONSE_CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=86400, alias="CAREMIND_CACHE_TTL_SECONDS")
    semantic_cache_enabled: bool = Field(default=True, alias="CAREMIND_SEMANTIC_CACHE_ENABLED")
    semantic_cache_threshold: float = Field(default=0.92, alias="CAREMIND_SEMANTIC_CACHE_THRESHOLD")
    min_retrieval_similarity: float = Field(default=0.5, alias="CAREMIND_MIN_RETRIEVAL_SIMILARITY")

    external_literature_search_enabled: bool = Field(default=True, alias="CAREMIND_EXTERNAL_SEARCH_ENABLED")
    ncbi_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        alias="NCBI_EUTILS_BASE_URL",
    )
    ncbi_tool: str = Field(default="caremind", alias="NCBI_TOOL")
    ncbi_email: str | None = Field(default=None, alias="NCBI_EMAIL")
    ncbi_api_key: str | None = Field(default=None, alias="NCBI_API_KEY")
    external_search_timeout_seconds: float = Field(default=12.0, alias="CAREMIND_EXTERNAL_SEARCH_TIMEOUT_SECONDS")
    external_search_max_results: int = Field(default=4, alias="CAREMIND_EXTERNAL_SEARCH_MAX_RESULTS")
    medlineplus_health_topics_enabled: bool = Field(default=True, alias="CAREMIND_MEDLINEPLUS_ENABLED")
    medlineplus_base_url: str = Field(
        default="https://wsearch.nlm.nih.gov/ws/query",
        alias="MEDLINEPLUS_BASE_URL",
    )
    medlineplus_tool: str = Field(default="caremind", alias="MEDLINEPLUS_TOOL")
    medlineplus_email: str | None = Field(default=None, alias="MEDLINEPLUS_EMAIL")
    medlineplus_timeout_seconds: float = Field(default=8.0, alias="CAREMIND_MEDLINEPLUS_TIMEOUT_SECONDS")
    medlineplus_max_results: int = Field(default=3, alias="CAREMIND_MEDLINEPLUS_MAX_RESULTS")
    medical_education_model_context_enabled: bool = Field(
        default=True,
        alias="CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_ENABLED",
    )
    medical_education_model_context_min_chars: int = Field(
        default=900,
        alias="CAREMIND_MEDICAL_EDUCATION_MODEL_CONTEXT_MIN_CHARS",
    )

    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")

    guardrails_enabled: bool = Field(default=True, alias="CAREMIND_GUARDRAILS_ENABLED")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, alias="CAREMIND_MAX_UPLOAD_BYTES")
    allowed_upload_content_types: str = Field(
        default="application/pdf,text/plain,text/markdown,image/png,image/jpeg,image/webp",
        alias="CAREMIND_ALLOWED_UPLOAD_CONTENT_TYPES",
    )
    image_ocr_enabled: bool = Field(default=True, alias="CAREMIND_IMAGE_OCR_ENABLED")
    image_ocr_engine: str = Field(default="auto", alias="CAREMIND_IMAGE_OCR_ENGINE")
    image_ocr_min_chars: int = Field(default=30, alias="CAREMIND_IMAGE_OCR_MIN_CHARS")
    image_ocr_structuring_enabled: bool = Field(default=True, alias="CAREMIND_IMAGE_OCR_STRUCTURING_ENABLED")
    nvidia_ocr_endpoint: str = Field(
        default="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2",
        alias="NVIDIA_OCR_ENDPOINT",
    )
    nvidia_ocr_model: str = Field(default="nvidia/nemotron-ocr-v2", alias="NVIDIA_OCR_MODEL")
    nvidia_ocr_timeout_seconds: float = Field(default=60.0, alias="NVIDIA_OCR_TIMEOUT_SECONDS")
    document_understanding_enabled: bool = Field(default=True, alias="CAREMIND_DOCUMENT_UNDERSTANDING_ENABLED")
    document_vlm_base_url: str | None = Field(default=None, alias="CAREMIND_DOCUMENT_VLM_BASE_URL")
    document_vlm_api_key: str | None = Field(default=None, alias="CAREMIND_DOCUMENT_VLM_API_KEY")
    document_vlm_model: str = Field(
        default="llama-4-scout-17b-16e-instruct",
        alias="CAREMIND_DOCUMENT_VLM_MODEL",
    )
    document_vlm_min_confidence: float = Field(default=0.55, alias="CAREMIND_DOCUMENT_VLM_MIN_CONFIDENCE")
    document_vlm_timeout_seconds: float = Field(default=60.0, alias="CAREMIND_DOCUMENT_VLM_TIMEOUT_SECONDS")
    radiology_vision_base_url: str | None = Field(default=None, alias="CAREMIND_RADIOLOGY_VISION_BASE_URL")
    radiology_vision_model: str = Field(default="google/medgemma-4b-it", alias="CAREMIND_RADIOLOGY_VISION_MODEL")
    transcript_min_confidence: float = Field(default=0.65, alias="CAREMIND_TRANSCRIPT_MIN_CONFIDENCE")

    eval_fail_threshold: float = Field(default=0.05, alias="CAREMIND_EVAL_FAIL_THRESHOLD")
    eval_warn_threshold: float = Field(default=0.02, alias="CAREMIND_EVAL_WARN_THRESHOLD")
    eval_baseline_path: Path = Field(
        default=Path("backend/eval_runs/baseline.json"),
        alias="CAREMIND_EVAL_BASELINE_PATH",
    )

    basic_auth_username: str | None = Field(default=None, alias="CAREMIND_USERNAME")
    basic_auth_password: str | None = Field(default=None, alias="CAREMIND_PASSWORD")

    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    )
    langchain_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    )
    langchain_project: str = Field(
        default="caremind-dev",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    )
    langchain_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    )
    langsmith_redact_inputs: bool = Field(default=True, alias="CAREMIND_LANGSMITH_REDACT_INPUTS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def should_use_pinecone(self) -> bool:
        return self.vector_backend.lower() == "pinecone"

    @property
    def should_use_supabase(self) -> bool:
        return self.vector_backend.lower() in {"supabase", "pgvector"}

    @property
    def should_use_sqlite_vectors(self) -> bool:
        return self.vector_backend.lower() in {"sqlite", "local"}

    @property
    def vector_backend_requires_network(self) -> bool:
        return self.should_use_pinecone or self.should_use_supabase

    @property
    def supabase_project_ref(self) -> str | None:
        if not self.supabase_url:
            return None
        host = urlparse(self.supabase_url).hostname or ""
        if host.endswith(".supabase.co"):
            return host.split(".")[0]
        return None

    @property
    def has_supabase_rest_credentials(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key)


    @property
    def redis_dsn(self) -> str:
        if self.redis_url:
            return self.redis_url
        password = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password}{self.redis_host}:{self.redis_port}/0"

    @property
    def langsmith_enabled(self) -> bool:
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)

    @property
    def allowed_upload_types(self) -> set[str]:
        return {item.strip().lower() for item in self.allowed_upload_content_types.split(",") if item.strip()}

    def apply_langsmith_environment(self) -> None:
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_enabled else "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.langsmith_enabled else "false"
        if not self.langsmith_enabled:
            return
        os.environ["LANGSMITH_API_KEY"] = self.langchain_api_key or ""
        os.environ["LANGSMITH_PROJECT"] = self.langchain_project
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key or ""
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        if self.langchain_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self.langchain_endpoint
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(settings)
    settings.apply_langsmith_environment()
    return settings


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler_exists = any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == str(settings.log_path.resolve())
        for handler in root.handlers
    )
    if not file_handler_exists:
        file_handler = RotatingFileHandler(
            settings.log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    console_handler_exists = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, RotatingFileHandler)
        and getattr(handler, "_caremind_console", False)
        for handler in root.handlers
    )
    if not console_handler_exists:
        console_handler = logging.StreamHandler(sys.__stderr__)
        console_handler.setFormatter(formatter)
        console_handler._caremind_console = True
        root.addHandler(console_handler)
    if settings.capture_prints:
        capture_print_streams()


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""
        self._caremind_stream_redirect = True

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, "print.%s", line)
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, "print.%s", self._buffer.strip())
        self._buffer = ""


def capture_print_streams() -> None:
    print_logger = logging.getLogger("caremind.print")
    if not getattr(sys.stdout, "_caremind_stream_redirect", False):
        sys.stdout = StreamToLogger(print_logger, logging.INFO)
    if not getattr(sys.stderr, "_caremind_stream_redirect", False):
        sys.stderr = StreamToLogger(print_logger, logging.ERROR)
