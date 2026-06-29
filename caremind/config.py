from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
