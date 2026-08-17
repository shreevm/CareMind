import hashlib
import logging
import math
import re
from typing import Any

import httpx

from .config import Settings
from .metrics import metrics
from .observability import trace_block

TOKEN_RE = re.compile(r"[a-zA-Z0-9_'-]+")
logger = logging.getLogger(__name__)


class EmbeddingServiceUnavailable(RuntimeError):
    """Raised when the configured embedding provider cannot produce vectors."""


class EmbeddingClient:
    _sentence_transformer_model: Any | None = None
    _sentence_transformer_model_key: tuple[str, str] | None = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimension = settings.embedding_dimension
        self.last_provider = ""
        self.last_model = ""
        self.degraded = False
        self.degraded_reason = ""

    def embed_texts(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        provider = self.settings.embedding_provider.lower()
        input_type = input_type or self.settings.nvidia_document_input_type
        self.degraded = False
        self.degraded_reason = ""
        with trace_block(
            self.settings,
            "EmbeddingClient.embed_texts",
            "embedding",
            {
                "requested_provider": provider,
                "text_count": len(texts),
                "total_chars": sum(len(text) for text in texts),
                "configured_dimension": self.settings.embedding_dimension,
                "embedding_model": self.settings.embedding_model,
                "embedding_index_version": self.settings.embedding_index_version,
                "nvidia_model": self.settings.nvidia_embedding_model if self.settings.nvidia_api_key else None,
                "ollama_model": self.settings.ollama_embedding_model,
            },
        ) as run:
            if provider in {"qwen", "sentence-transformers", "sentence_transformers"}:
                vectors = self._embed_with_sentence_transformers(texts, input_type=input_type)
                self._record_embedding_source("qwen", self.settings.embedding_model, vectors)
                run.end(
                    {
                        "provider": self.last_provider,
                        "model": self.last_model,
                        "index_version": self.settings.embedding_index_version,
                        "vector_count": len(vectors),
                        "dimension": self.dimension,
                    }
                )
                return vectors
            if provider in {"local", "local_hashing", "local-hashing"}:
                return self._embed_with_local_fallback(texts, run, "explicit_local_provider")
            if provider == "nvidia" and not self.settings.nvidia_api_key:
                raise EmbeddingServiceUnavailable("NVIDIA_API_KEY is required when CAREMIND_EMBEDDING_PROVIDER=nvidia.")
            if provider == "ollama" or (provider == "auto" and self.settings.ollama_embedding_model and not self.settings.nvidia_api_key):
                try:
                    vectors = self._embed_with_ollama(texts)
                    effective_provider = "ollama"
                    self._record_embedding_source(effective_provider, self.settings.ollama_embedding_model or "ollama", vectors)
                    run.end({"provider": effective_provider, "model": self.last_model, "vector_count": len(vectors), "dimension": self.dimension})
                    return vectors
                except Exception as exc:
                    metrics.record_fallback("embedding", "ollama", exc.__class__.__name__)
                    if provider == "ollama":
                        raise EmbeddingServiceUnavailable(
                            f"Ollama embedding provider failed: {exc.__class__.__name__}"
                        ) from exc
                    return self._embed_with_local_fallback(texts, run, f"ollama_failed:{exc.__class__.__name__}")
            if provider in {"auto", "nvidia"} and self.settings.nvidia_api_key:
                try:
                    vectors = self._embed_with_nvidia(texts, input_type=input_type)
                    effective_provider = "nvidia"
                    self._record_embedding_source(effective_provider, self.settings.nvidia_embedding_model, vectors)
                    run.end({"provider": effective_provider, "model": self.last_model, "vector_count": len(vectors), "dimension": self.dimension})
                    return vectors
                except Exception as exc:
                    metrics.record_fallback("embedding", "nvidia", exc.__class__.__name__)
                    raise EmbeddingServiceUnavailable(
                        f"NVIDIA embedding provider failed; refusing local hashing fallback: {exc.__class__.__name__}"
                    ) from exc
            return self._embed_with_local_fallback(texts, run, "no_remote_embedding_provider_configured")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text], input_type=self.settings.nvidia_query_input_type)[0]

    def configured_model_label(self) -> str:
        provider = self.settings.embedding_provider.lower()
        if provider in {"qwen", "sentence-transformers", "sentence_transformers"}:
            return self.settings.embedding_model
        if provider == "ollama" or (provider == "auto" and self.settings.ollama_embedding_model and not self.settings.nvidia_api_key):
            return self.settings.ollama_embedding_model or "ollama"
        if provider in {"auto", "nvidia"} and self.settings.nvidia_api_key:
            return self.settings.nvidia_embedding_model
        return f"local-hashing-{self.dimension}d"

    def _embed_with_local_fallback(self, texts: list[str], run, reason: str) -> list[list[float]]:
        if not self.settings.local_embedding_fallback_enabled:
            raise EmbeddingServiceUnavailable(
                "No remote embedding provider is available and local hashing fallback is disabled."
            )
        vectors = [self._embed_locally(text) for text in texts]
        self.degraded = True
        self.degraded_reason = reason
        logger.warning(
            "EMBEDDING DEGRADED: using local hashing vectors provider=local_hashing model=local-hashing-%sd reason=%s. "
            "Semantic similarity scores are not reliable; configure NVIDIA_API_KEY and re-index documents.",
            self.dimension,
            reason,
        )
        metrics.record_fallback("embedding", "local_hashing", reason)
        self._record_embedding_source("local_hashing", f"local-hashing-{self.dimension}d", vectors)
        run.end(
            {
                "provider": self.last_provider,
                "model": self.last_model,
                "vector_count": len(vectors),
                "dimension": self.dimension,
                "degraded": True,
                "degraded_reason": reason,
            }
        )
        return vectors

    def _embed_with_nvidia(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        url = f"{self.settings.nvidia_base_url.rstrip('/')}/embeddings"
        payload = {"input": texts, "model": self.settings.nvidia_embedding_model}
        if input_type:
            payload["input_type"] = input_type
        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=45) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def _embed_with_sentence_transformers(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        model = self._load_sentence_transformer()
        is_query = (input_type or "").lower() == self.settings.nvidia_query_input_type.lower()
        encoder_name = "encode_query" if is_query and hasattr(model, "encode_query") else "encode_document"
        if not is_query and not hasattr(model, encoder_name):
            encoder_name = "encode"
        if is_query and not hasattr(model, encoder_name):
            encoder_name = "encode"
        encoder = getattr(model, encoder_name)
        vectors: list[list[float]] = []
        batch_size = max(1, int(self.settings.embedding_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                encoded = encoder(
                    batch,
                    normalize_embeddings=self.settings.embedding_normalize,
                    batch_size=len(batch),
                    show_progress_bar=False,
                )
            except TypeError:
                encoded = encoder(batch)
            rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            if batch and rows and isinstance(rows[0], (float, int)):
                rows = [rows]
            for offset, vector in enumerate(rows or []):
                vectors.append(self._validate_vector(vector, start + offset))
        if len(vectors) != len(texts):
            raise EmbeddingServiceUnavailable(
                f"SentenceTransformer returned {len(vectors)} vectors for {len(texts)} texts."
            )
        return vectors

    def _load_sentence_transformer(self):
        provider_key = (self.settings.embedding_model, self._selected_device())
        if (
            EmbeddingClient._sentence_transformer_model is not None
            and EmbeddingClient._sentence_transformer_model_key == provider_key
        ):
            return EmbeddingClient._sentence_transformer_model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingServiceUnavailable("sentence-transformers is required for the qwen embedding provider.") from exc
        try:
            model = SentenceTransformer(
                self.settings.embedding_model,
                device=provider_key[1],
                trust_remote_code=True,
            )
        except Exception as exc:
            raise EmbeddingServiceUnavailable(
                f"Unable to load embedding model {self.settings.embedding_model}: {exc.__class__.__name__}"
            ) from exc
        EmbeddingClient._sentence_transformer_model = model
        EmbeddingClient._sentence_transformer_model_key = provider_key
        return model

    def _selected_device(self) -> str:
        configured = self.settings.embedding_device.lower().strip()
        if configured and configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _validate_vector(self, vector: Any, index: int) -> list[float]:
        if vector is None:
            raise EmbeddingServiceUnavailable(f"Embedding vector {index} is empty.")
        try:
            values = [float(value) for value in vector]
        except Exception as exc:
            raise EmbeddingServiceUnavailable(f"Embedding vector {index} contains non-numeric values.") from exc
        if not values:
            raise EmbeddingServiceUnavailable(f"Embedding vector {index} is empty.")
        if len(values) != self.settings.embedding_dimension:
            raise EmbeddingServiceUnavailable(
                f"Embedding vector {index} has dimension {len(values)}; expected {self.settings.embedding_dimension}."
            )
        if any(not math.isfinite(value) for value in values):
            raise EmbeddingServiceUnavailable(f"Embedding vector {index} contains NaN or infinite values.")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise EmbeddingServiceUnavailable(f"Embedding vector {index} has zero norm.")
        return values

    def _embed_with_ollama(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.ollama_embedding_model:
            raise ValueError("OLLAMA_EMBEDDING_MODEL is required when CAREMIND_EMBEDDING_PROVIDER=ollama")
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/embed"
        payload = {
            "model": self.settings.ollama_embedding_model,
            "input": texts,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload)
            if response.status_code == 404:
                return self._embed_with_ollama_legacy(texts, client)
            response.raise_for_status()
            body = response.json()
        vectors = body.get("embeddings") or []
        if not vectors:
            raise ValueError("Ollama returned no embeddings")
        self.dimension = len(vectors[0])
        return vectors

    def _embed_with_ollama_legacy(self, texts: list[str], client: httpx.Client) -> list[list[float]]:
        vectors = []
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/embeddings"
        for text in texts:
            response = client.post(
                url,
                json={"model": self.settings.ollama_embedding_model, "prompt": text},
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
            if not vector:
                raise ValueError("Ollama returned no embedding")
            vectors.append(vector)
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def _embed_locally(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _record_embedding_source(self, provider: str, model: str, vectors: list[list[float]]) -> None:
        self.last_provider = provider
        self.last_model = model
        if vectors:
            self.dimension = len(vectors[0])
