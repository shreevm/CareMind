"""MCP tool server facade.

The FastAPI backend calls these tools directly, and this module also exposes a
real MCP SDK server plus a legacy stdio JSON loop for demos where a separate
process is useful.
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
    if name == "medlineplus_health_topic_search":
        chunks = tools.medlineplus_health_topic_search(
            args.get("query", ""),
            top_k=int(args.get("top_k", 3)),
        )
        return {"result": [chunk.model_dump() for chunk in chunks]}
    if name == "timeline_extraction":
        return {"result": tools.extract_timeline(args.get("document_id", ""))}
    return {"error": f"Unknown tool: {name}"}


def legacy_stdio_main() -> None:
    for line in sys.stdin:
        try:
            response = handle_call(json.loads(line))
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response), flush=True)


def build_mcp_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("CareMind")

    @server.tool()
    def document_search(query: str, workspace_id: str = "default", top_k: int = 5) -> list[dict]:
        """Search indexed CareMind workspace documents."""
        chunks = build_tools().document_search(query, workspace_id=workspace_id, top_k=top_k)
        return [chunk.model_dump() for chunk in chunks]

    @server.tool()
    def compare_reports(document_ids: list[str], workspace_id: str = "default") -> dict:
        """Compare two indexed CareMind reports."""
        response = build_tools().compare_reports(document_ids, workspace_id=workspace_id)
        return response.model_dump()

    @server.tool()
    def medical_education_search(query: str, top_k: int = 3) -> list[dict]:
        """Search the built-in CareMind medical education corpus."""
        chunks = build_tools().medical_education_search(query, top_k=top_k)
        return [chunk.model_dump() for chunk in chunks]

    @server.tool()
    def medlineplus_health_topic_search(query: str, top_k: int = 3) -> list[dict]:
        """Search MedlinePlus health-topic education summaries."""
        chunks = build_tools().medlineplus_health_topic_search(query, top_k=top_k)
        return [chunk.model_dump() for chunk in chunks]

    @server.tool()
    def timeline_extraction(document_id: str) -> list[str]:
        """Extract simple timeline-like lines from an indexed document."""
        return build_tools().extract_timeline(document_id)

    return server


def main() -> None:
    if "--legacy-stdio" in sys.argv:
        legacy_stdio_main()
        return
    try:
        build_mcp_server().run()
    except ImportError:
        legacy_stdio_main()


if __name__ == "__main__":
    main()
