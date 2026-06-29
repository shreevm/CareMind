import hashlib
import math
import re

import httpx

from .config import Settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9_'-]+")


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimension = settings.embedding_dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.settings.nvidia_api_key:
            try:
                return self._embed_with_nvidia(texts)
            except Exception:
                # Keep local demos alive when the hosted endpoint is unavailable.
                pass
        return [self._embed_locally(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_with_nvidia(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.settings.nvidia_base_url.rstrip('/')}/embeddings"
        payload = {"input": texts, "model": self.settings.nvidia_embedding_model}
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
