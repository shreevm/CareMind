from functools import lru_cache
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CareMind"
    app_version: str = "0.4.0"
    environment: str = Field(default="local", alias="CAREMIND_ENV")

    data_dir: Path = Field(default=Path("data"), alias="CAREMIND_DATA_DIR")
    upload_dir: Path = Field(default=Path("data/uploads"), alias="CAREMIND_UPLOAD_DIR")
    sqlite_path: Path = Field(default=Path("data/caremind.db"), alias="CAREMIND_SQLITE_PATH")

    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    nvidia_chat_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        alias="NVIDIA_CHAT_MODEL",
    )
    nvidia_embedding_model: str = Field(default="nvolveqa_40k", alias="NVIDIA_EMBEDDING_MODEL")
    medical_llm_base_url: str | None = Field(default=None, alias="MEDICAL_LLM_BASE_URL")
    medical_llm_api_key: str | None = Field(default=None, alias="MEDICAL_LLM_API_KEY")
    medical_llm_model: str | None = Field(default=None, alias="MEDICAL_LLM_MODEL")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="caremind-index", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    vector_backend: str = Field(default="local", alias="CAREMIND_VECTOR_BACKEND")
    embedding_dimension: int = Field(default=384, alias="CAREMIND_EMBEDDING_DIM")

    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    session_ttl_seconds: int = Field(default=3600, alias="CAREMIND_SESSION_TTL")

    basic_auth_username: str | None = Field(default=None, alias="CAREMIND_USERNAME")
    basic_auth_password: str | None = Field(default=None, alias="CAREMIND_PASSWORD")

    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="caremind-dev", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str | None = Field(default=None, alias="LANGCHAIN_ENDPOINT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def should_use_pinecone(self) -> bool:
        return self.vector_backend.lower() == "pinecone" and bool(self.pinecone_api_key)

    @property
    def redis_dsn(self) -> str:
        if self.redis_url:
            return self.redis_url
        password = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password}{self.redis_host}:{self.redis_port}/0"

    @property
    def langsmith_enabled(self) -> bool:
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)

    def apply_langsmith_environment(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.langsmith_enabled else "false"
        if not self.langsmith_enabled:
            return
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key or ""
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        if self.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.apply_langsmith_environment()
    return settings
