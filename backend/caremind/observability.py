from contextlib import contextmanager
import hashlib
import logging
import sys
from typing import Any

from .config import Settings


logger = logging.getLogger(__name__)


class TraceRun:
    def __init__(self, run: Any | None = None):
        self._run = run

    def end(self, outputs: dict[str, Any] | None = None) -> None:
        if not self._run:
            return
        try:
            self._run.end(outputs=outputs or {})
        except Exception as exc:
            logger.warning("langsmith.trace.end_failed error=%s", exc.__class__.__name__)


@contextmanager
def trace_block(
    settings: Settings,
    name: str,
    run_type: str,
    inputs: dict[str, Any] | None = None,
):
    if not settings.langsmith_enabled:
        yield TraceRun()
        return

    try:
        import langsmith as ls
    except Exception as exc:
        logger.warning("langsmith.trace.unavailable error=%s", exc.__class__.__name__)
        yield TraceRun()
        return

    tracing_cm = None
    run_cm = None
    run = None
    try:
        tracing_cm = ls.tracing_context(enabled=True)
        tracing_cm.__enter__()
        run_cm = ls.trace(
            name,
            run_type,
            project_name=settings.langchain_project,
            inputs=inputs or {},
        )
        run = run_cm.__enter__()
    except Exception as exc:
        logger.warning("langsmith.trace.start_failed name=%s error=%s", name, exc.__class__.__name__)
        _close_contexts(run_cm, tracing_cm, None)
        yield TraceRun()
        return

    exc_info = None
    try:
        yield TraceRun(run)
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _close_contexts(run_cm, tracing_cm, exc_info)


def _close_contexts(run_cm: Any, tracing_cm: Any, exc_info: tuple | None) -> None:
    args = exc_info or (None, None, None)
    for manager in (run_cm, tracing_cm):
        if manager is None:
            continue
        try:
            manager.__exit__(*args)
        except Exception as exc:
            logger.warning("langsmith.trace.close_failed error=%s", exc.__class__.__name__)


def trace_text(settings: Settings, text: str | None) -> str | dict[str, Any]:
    value = text or ""
    return {
        "redacted": True,
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
    }


def chunk_outputs(settings: Settings, chunks: list[Any]) -> list[dict[str, Any]]:
    outputs = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        item = {
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "document_id": getattr(chunk, "document_id", ""),
            "document_name": getattr(chunk, "document_name", ""),
            "page": getattr(chunk, "page", None),
            "score": getattr(chunk, "score", None),
            "text_chars": len(text),
        }
        outputs.append(item)
    return outputs


def chat_request_inputs(settings: Settings, request: Any) -> dict[str, Any]:
    return {
        "message": trace_text(settings, getattr(request, "message", "")),
        "session_id": getattr(request, "session_id", ""),
        "workspace_id": getattr(request, "workspace_id", ""),
        "modality": getattr(request, "modality", ""),
        "top_k": getattr(request, "top_k", None),
    }


def chat_response_outputs(settings: Settings, response: Any) -> dict[str, Any]:
    answer = getattr(response, "answer", "") or ""
    return {
        "route": getattr(response, "route", ""),
        "answer": trace_text(settings, answer),
        "answer_chars": len(answer),
        "citation_count": len(getattr(response, "citations", []) or []),
        "tool_calls": getattr(response, "tool_calls", []) or [],
        "safety_note_count": len(getattr(response, "safety_notes", []) or []),
        "trace": {
            "cache_hit": (getattr(response, "trace", {}) or {}).get("cache_hit"),
            "vector_backend": (getattr(response, "trace", {}) or {}).get("vector_backend"),
            "generation_model": (getattr(response, "trace", {}) or {}).get("generation_model"),
            "embedding_model": (getattr(response, "trace", {}) or {}).get("embedding_model"),
            "total_latency_ms": (getattr(response, "trace", {}) or {}).get("total_latency_ms"),
        },
    }
