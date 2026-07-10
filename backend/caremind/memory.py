import json

from .config import Settings
from .schemas import ChatMessage
from .store import SQLiteStore


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

    def append(self, session_id: str, workspace_id: str, role: str, content: str) -> None:
        self.sqlite_store.append_message(session_id, workspace_id, role, content)
        if self._redis is None:
            return
        history = self.load(session_id, workspace_id, limit=40)
        payload = [message.model_dump(mode="json") for message in history]
        self._redis.setex(
            self._key(session_id, workspace_id),
            self.settings.session_ttl_seconds,
            json.dumps(payload),
        )

    def cache_get(self, workspace_id: str, cache_key: str) -> dict | None:
        if self._redis is None:
            return None
        raw = self._redis.get(self._cache_key(workspace_id, cache_key))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def cache_set(self, workspace_id: str, cache_key: str, value: dict) -> None:
        if self._redis is None:
            return
        self._redis.setex(
            self._cache_key(workspace_id, cache_key),
            self.settings.session_ttl_seconds,
            json.dumps(value),
        )

    def cache_clear_workspace(self, workspace_id: str) -> int:
        if self._redis is None:
            return 0
        keys = list(self._redis.scan_iter(self._cache_key(workspace_id, "*")))
        if not keys:
            return 0
        return int(self._redis.delete(*keys))

    @property
    def redis_enabled(self) -> bool:
        return self._redis is not None

    def _key(self, session_id: str, workspace_id: str) -> str:
        return f"caremind:{workspace_id}:session:{session_id}"

    def _cache_key(self, workspace_id: str, cache_key: str) -> str:
        return f"caremind:{workspace_id}:cache:{cache_key}"

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
