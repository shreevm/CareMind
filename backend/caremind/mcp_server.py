"""Minimal MCP-style tool server facade.

The FastAPI backend calls these tools directly, and this module also exposes a
small stdio JSON loop for demos where a separate process is useful.
"""

import json
import sys

from .config import get_settings
from .embeddings import EmbeddingClient
from .store import SQLiteStore
from .tools import DocumentTools
from .vectorstore import VectorStore


def build_tools() -> DocumentTools:
    settings = get_settings()
    store = SQLiteStore(settings.sqlite_path)
    embeddings = EmbeddingClient(settings)
    vectorstore = VectorStore(settings, store)
    return DocumentTools(embeddings, vectorstore, store)


def handle_call(payload: dict) -> dict:
    tools = build_tools()
    name = payload.get("tool")
    args = payload.get("arguments", {})
    if name == "document_search":
        chunks = tools.document_search(
            args.get("query", ""),
            workspace_id=args.get("workspace_id", "default"),
            top_k=int(args.get("top_k", 5)),
        )
        return {"result": [chunk.model_dump() for chunk in chunks]}
    if name == "compare_reports":
        response = tools.compare_reports(
            args.get("document_ids", []),
            workspace_id=args.get("workspace_id", "default"),
        )
        return {"result": response.model_dump()}
    if name == "medical_education_search":
        chunks = tools.medical_education_search(
            args.get("query", ""),
            top_k=int(args.get("top_k", 3)),
        )
        return {"result": [chunk.model_dump() for chunk in chunks]}
    if name == "timeline_extraction":
        return {"result": tools.extract_timeline(args.get("document_id", ""))}
    return {"error": f"Unknown tool: {name}"}


def main() -> None:
    for line in sys.stdin:
        try:
            response = handle_call(json.loads(line))
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
