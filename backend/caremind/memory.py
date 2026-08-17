import json
import logging

from .config import Settings
from .schemas import ChatMessage
from .store import SQLiteStore
from .vectorstore import cosine_similarity


logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, settings: Settings, sqlite_store: SQLiteStore):
        self.settings = settings
        self.sqlite_store = sqlite_store
        self.redis_error: str | None = None
        self._redis = self._connect_redis()

    def load(self, session_id: str, workspace_id: str, limit: int = 20) -> list[ChatMessage]:
        if self._redis is not None:
            raw = self._redis.get(self._key(session_id, workspace_id))
            if raw:
                try:
                    rows = json.loads(raw)
                    return [ChatMessage(**row) for row in rows[-limit:]]
                except Exception:
                    pass
        return self.sqlite_store.load_messages(session_id, workspace_id, limit=limit)

    def append(self, session_id: str, workspace_id: str, role: str, content: str) -> int:
        message_id = self.sqlite_store.append_message(session_id, workspace_id, role, content)
        if self._redis is None:
            return message_id
        # SQLite is the source of truth: reading back through Redis here would
        # re-write the stale cached list and drop the message just appended.
        history = self.sqlite_store.load_messages(session_id, workspace_id, limit=40)
        payload = [message.model_dump(mode="json") for message in history]
        self._redis.setex(
            self._key(session_id, workspace_id),
            self.settings.session_ttl_seconds,
            json.dumps(payload),
        )
        return message_id

    def context_get(self, session_id: str, workspace_id: str) -> dict:
        return self.sqlite_store.get_conversation_context(session_id, workspace_id)

    def context_set(self, session_id: str, workspace_id: str, payload: dict) -> None:
        self.sqlite_store.set_conversation_context(session_id, workspace_id, payload)

    def cache_get(self, workspace_id: str, cache_key: str, current_query: str = "") -> dict | None:
        if not self.settings.response_cache_enabled:
            logger.info(
                "cache.lookup key=%s hit=false reason=disabled current_query_chars=%s",
                cache_key,
                len(current_query),
            )
            return None
        if self._redis is None:
            logger.info(
                "cache.lookup key=%s hit=false reason=redis_unavailable current_query_chars=%s",
                cache_key,
                len(current_query),
            )
            return None
        raw = self._redis.get(self._cache_key(workspace_id, cache_key))
        if not raw:
            logger.info(
                "cache.lookup key=%s hit=false current_query_chars=%s",
                cache_key,
                len(current_query),
            )
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            logger.info(
                "cache.lookup key=%s hit=false reason=decode_error current_query_chars=%s",
                cache_key,
                len(current_query),
            )
            return None
        metadata = payload.get("trace", {}).get("cache_metadata", {})
        logger.info(
            "cache.lookup key=%s hit=true cached_at=%r current_query_chars=%s",
            cache_key,
            metadata.get("cached_at"),
            len(current_query),
        )
        return payload

    def cache_set(self, workspace_id: str, cache_key: str, value: dict) -> None:
        if not self.settings.response_cache_enabled:
            return
        if self._redis is None:
            return
        self._redis.setex(
            self._cache_key(workspace_id, cache_key),
            self.settings.cache_ttl_seconds,
            json.dumps(value),
        )

    @property
    def semantic_cache_enabled(self) -> bool:
        return self._redis is not None and self.settings.response_cache_enabled and self.settings.semantic_cache_enabled

    def semantic_cache_get(
        self,
        workspace_id: str,
        cache_namespace: str,
        query_embedding: list[float],
        current_query: str = "",
    ) -> tuple[dict | None, float]:
        """Return the cached response whose stored query embedding is most similar,
        provided it clears the configured cosine-similarity threshold."""
        if not self.semantic_cache_enabled or not query_embedding:
            logger.info(
                "semantic_cache.lookup namespace=%s hit=false reason=disabled_or_empty_embedding current_query_chars=%s",
                cache_namespace,
                len(current_query),
            )
            return None, 0.0
        best_payload: dict | None = None
        best_similarity = 0.0
        try:
            for key in self._redis.scan_iter(self._semantic_key(workspace_id, f"{cache_namespace}:*")):
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except Exception:
                    continue
                if (
                    entry.get("embedding_provider") != self.settings.embedding_provider
                    or entry.get("embedding_model") != self.settings.embedding_model
                    or entry.get("embedding_dimension") != len(query_embedding)
                    or entry.get("embedding_index_version") != self.settings.embedding_index_version
                ):
                    continue
                similarity = cosine_similarity(query_embedding, entry.get("embedding", []))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_payload = entry.get("response")
        except Exception:
            logger.info(
                "semantic_cache.lookup namespace=%s hit=false reason=scan_error current_query_chars=%s",
                cache_namespace,
                len(current_query),
            )
            return None, 0.0
        if best_payload is not None and best_similarity >= self.settings.semantic_cache_threshold:
            metadata = best_payload.get("trace", {}).get("cache_metadata", {})
            logger.info(
                "semantic_cache.lookup namespace=%s hit=true similarity=%.4f cached_at=%r current_query_chars=%s",
                cache_namespace,
                best_similarity,
                metadata.get("cached_at"),
                len(current_query),
            )
            return best_payload, best_similarity
        logger.info(
            "semantic_cache.lookup namespace=%s hit=false similarity=%.4f current_query_chars=%s",
            cache_namespace,
            best_similarity,
            len(current_query),
        )
        return None, best_similarity

    def semantic_cache_set(
        self,
        workspace_id: str,
        cache_namespace: str,
        cache_key: str,
        query_embedding: list[float],
        response: dict,
    ) -> None:
        if not self.semantic_cache_enabled or not query_embedding:
            return
        try:
            self._redis.setex(
                self._semantic_key(workspace_id, f"{cache_namespace}:{cache_key}"),
                self.settings.cache_ttl_seconds,
                json.dumps(
                    {
                        "embedding": query_embedding,
                        "embedding_provider": self.settings.embedding_provider,
                        "embedding_model": self.settings.embedding_model,
                        "embedding_dimension": len(query_embedding),
                        "embedding_index_version": self.settings.embedding_index_version,
                        "response": response,
                    }
                ),
            )
        except Exception:
            pass

    def cache_clear_workspace(self, workspace_id: str) -> int:
        if self._redis is None:
            return 0
        keys = list(self._redis.scan_iter(self._cache_key(workspace_id, "*")))
        keys.extend(self._redis.scan_iter(self._semantic_key(workspace_id, "*")))
        if not keys:
            return 0
        return int(self._redis.delete(*keys))

    def session_clear(self, session_id: str, workspace_id: str) -> int:
        if self._redis is None:
            return 0
        return int(self._redis.delete(self._key(session_id, workspace_id)))

    def session_delete(self, session_id: str, workspace_id: str) -> dict:
        sqlite_deleted = self.sqlite_store.delete_session(session_id, workspace_id)
        redis_deleted = self.session_clear(session_id, workspace_id)
        return {
            "session_id": session_id,
            "workspace_id": workspace_id,
            "sqlite_deleted": sqlite_deleted,
            "redis_deleted": redis_deleted,
        }

    def cache_debug_info(self, workspace_id: str, cache_key: str, cache_namespace: str) -> dict:
        return {
            "response_cache_enabled": self.settings.response_cache_enabled,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "redis_enabled": self.redis_enabled,
            "redis_error": self.redis_error,
            "cache_ttl_seconds": self.settings.cache_ttl_seconds,
            "exact_redis_key": self._cache_key(workspace_id, cache_key),
            "semantic_redis_key": self._semantic_key(workspace_id, f"{cache_namespace}:{cache_key}"),
            "semantic_redis_prefix": self._semantic_key(workspace_id, f"{cache_namespace}:*"),
        }

    def cache_debug_summary(self, workspace_id: str, session_id: str | None = None) -> dict:
        summary = {
            "workspace_id": workspace_id,
            "session_id": session_id,
            "response_cache_enabled": self.settings.response_cache_enabled,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "redis_enabled": self.redis_enabled,
            "redis_error": self.redis_error,
            "cache_ttl_seconds": self.settings.cache_ttl_seconds,
            "session_redis_key": self._key(session_id, workspace_id) if session_id else None,
            "exact_cache_pattern": self._cache_key(workspace_id, "*"),
            "semantic_cache_pattern": self._semantic_key(workspace_id, "*"),
            "counts": {
                "session": 0,
                "exact_response_cache": 0,
                "semantic_response_cache": 0,
            },
        }
        if self._redis is None:
            return summary

        try:
            if session_id and self._redis.exists(self._key(session_id, workspace_id)):
                summary["counts"]["session"] = 1
            summary["counts"]["exact_response_cache"] = sum(
                1 for _ in self._redis.scan_iter(self._cache_key(workspace_id, "*"))
            )
            summary["counts"]["semantic_response_cache"] = sum(
                1 for _ in self._redis.scan_iter(self._semantic_key(workspace_id, "*"))
            )
        except Exception as exc:
            summary["redis_error"] = exc.__class__.__name__
        return summary

    @property
    def redis_enabled(self) -> bool:
        return self._redis is not None

    def _key(self, session_id: str, workspace_id: str) -> str:
        return f"caremind:{workspace_id}:session:{session_id}"

    def _cache_key(self, workspace_id: str, cache_key: str) -> str:
        return f"caremind:{workspace_id}:cache:{cache_key}"

    def _semantic_key(self, workspace_id: str, cache_key: str) -> str:
        return f"caremind:{workspace_id}:semcache:{cache_key}"

    def _connect_redis(self):
        try:
            import redis

            client = redis.from_url(self.settings.redis_dsn, decode_responses=True)
            client.ping()
            self.redis_error = None
            return client
        except Exception as exc:
            self.redis_error = exc.__class__.__name__
            return None
