"""
Debug CareMind multi-turn conversation flow.

Default mode runs observe -> route -> retrieval preparation so you can inspect
conversation rewriting and retrieved chunks without calling the LLM. Use --full
to run the complete agent answer for each turn.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.caremind.agent import CareMindAgent
from backend.caremind.config import Settings
from backend.caremind.embeddings import EmbeddingClient
from backend.caremind.ingestion import DocumentIngestionService
from backend.caremind.llm import LLMClient
from backend.caremind.memory import ConversationMemory
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest
from backend.caremind.store import SQLiteStore
from backend.caremind.tools import DocumentTools
from backend.caremind.vectorstore import VectorStore


DEFAULT_TURNS = [
    "What are the key findings in the latest report?",
    "What does that mean?",
    "Is it serious?",
    "What should the patient do next?",
    "In general, what is pneumonia?",
    "Back to the report: did it mention pneumonia?",
]


class Services:
    def __init__(self) -> None:
        self.settings = Settings()
        self.store = SQLiteStore(self.settings.sqlite_path)
        self.embeddings = EmbeddingClient(self.settings)
        self.vectorstore = VectorStore(self.settings, self.store)
        self.llm = LLMClient(self.settings)
        self.ingestion = DocumentIngestionService(
            self.settings.upload_dir,
            self.store,
            self.embeddings,
            self.vectorstore,
            self.settings.max_upload_bytes,
            self.settings.allowed_upload_types,
            llm=self.llm,
        )
        self.tools = DocumentTools(self.embeddings, self.vectorstore, self.store)
        self.memory = ConversationMemory(self.settings, self.store)
        self.agent = CareMindAgent(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            safety=SafetyLayer(),
        )


def seed_demo_documents(services: Services, workspace_id: str) -> None:
    services.ingestion.ingest_text(
        filename="flow-debug-baseline-report.txt",
        text=(
            "Synthetic medical report. Patient reports fatigue and mild shortness of breath. "
            "Hemoglobin is 11.2 g/dL. White blood cell count is 8.1. "
            "Chest x-ray shows no acute infiltrate. "
            "Plan notes iron studies and routine follow-up."
        ),
        workspace_id=workspace_id,
    )
    services.ingestion.ingest_text(
        filename="flow-debug-followup-report.txt",
        text=(
            "Synthetic follow-up report. Fatigue has improved. Hemoglobin is 12.6 g/dL. "
            "White blood cell count is 7.9. Chest x-ray remains without acute infiltrate. "
            "No pneumonia diagnosis is documented. Plan notes continued monitoring."
        ),
        workspace_id=workspace_id,
    )
    services.memory.cache_clear_workspace(workspace_id)


def inspect_turn(
    services: Services,
    *,
    message: str,
    session_id: str,
    workspace_id: str,
    top_k: int,
    full: bool,
) -> dict[str, Any]:
    original_message = message
    request = ChatRequest(
        message=message,
        session_id=session_id,
        workspace_id=workspace_id,
        top_k=top_k,
        debug=True,
        bypass_cache=True,
    )
    if full:
        response = services.agent.answer(request)
        trace = response.trace or {}
        context_resolution = trace.get("context_resolution", {})
        return {
            "message": message,
            "route": response.route,
            "internal_route": trace.get("internal_route", response.route),
            "used_context": context_resolution.get("used_context", ""),
            "reason": context_resolution.get("reason", ""),
            "rewritten_query": trace.get("rewritten_query", ""),
            "retrieved_count": len(trace.get("retrieved_chunks", [])),
            "retrieved_ids": ids_from_trace(trace.get("retrieved_chunks", [])),
            "citation_count": len(response.citations),
            "answer_preview": response.answer[:180].replace("\n", " "),
        }

    started = time.perf_counter()
    state = services.agent._observe_node({"request": request, "started_at": started})
    route = services.agent._planned_route(state["request"].message, state.get("observations", {}))
    updates: dict[str, Any] = {}
    if route == "retrieve":
        updates = services.agent.document_rag_agent.prepare(state["request"], [], state.get("trace_steps", []))
    elif route == "medical_education":
        updates = services.agent.medical_education_agent.prepare(state["request"], [], state.get("trace_steps", []))
    persist_lightweight_turn(
        services,
        state={
            **state,
            **updates,
            "route": route,
            "original_message": original_message,
        },
        answer="[debug] retrieval-only conversation flow turn",
    )
    trace_step = last_agent_step(updates.get("trace_steps", []))
    context_resolution = state.get("context_resolution", {})
    return {
        "message": message,
        "route": route,
        "internal_route": route,
        "used_context": context_resolution.get("used_context", ""),
        "reason": context_resolution.get("reason", ""),
        "rewritten_query": state["request"].message,
        "retrieved_count": len(updates.get("retrieved_chunks", [])),
        "retrieved_ids": ids_from_trace(updates.get("retrieved_chunks", [])),
        "citation_count": len(updates.get("citations", [])),
        "tool_calls": ";".join(updates.get("tool_calls", [])),
        "kept_count": trace_step.get("kept_count", ""),
        "filtered_count": trace_step.get("filtered_count", ""),
        "answer_preview": "",
    }


def persist_lightweight_turn(services: Services, *, state: dict[str, Any], answer: str) -> None:
    """Mimic the memory/context side effects of finalize without calling the LLM."""
    request = state["request"]
    services.memory.append(
        request.session_id,
        request.workspace_id,
        "user",
        str(state.get("original_message") or request.message),
    )
    services.memory.append(request.session_id, request.workspace_id, "assistant", answer)
    updated_context = services.agent._updated_conversation_context(state, answer)
    if updated_context:
        services.memory.context_set(request.session_id, request.workspace_id, updated_context)


def ids_from_trace(items: list[Any]) -> str:
    ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id") or item.get("document_id")
        if chunk_id:
            ids.append(str(chunk_id))
    return ";".join(ids)


def last_agent_step(steps: list[Any]) -> dict[str, Any]:
    for step in reversed(steps):
        if isinstance(step, dict) and step.get("node") in {"document_rag_agent", "medical_education_agent"}:
            return step
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug multi-turn CareMind conversation routing and retrieval.")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--turn", action="append", dest="turns", help="Conversation turn. Can be passed multiple times.")
    parser.add_argument("--seed-demo", action="store_true", help="Ingest two synthetic reports before running turns.")
    parser.add_argument("--full", action="store_true", help="Run full agent.answer(), including LLM generation.")
    parser.add_argument("--keep-session", action="store_true", help="Do not delete the temporary session at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    services = Services()
    session_id = args.session_id or f"conversation-flow-debug-{uuid.uuid4()}"
    turns = args.turns or DEFAULT_TURNS
    if args.seed_demo:
        seed_demo_documents(services, args.workspace_id)

    rows = []
    try:
        for index, message in enumerate(turns, start=1):
            row = inspect_turn(
                services,
                message=message,
                session_id=session_id,
                workspace_id=args.workspace_id,
                top_k=args.top_k,
                full=args.full,
            )
            row = {"turn": index, **row}
            rows.append(row)
            print(
                f"\nTurn {index}: {message}\n"
                f"  route={row['route']} used_context={row['used_context']} reason={row['reason']}\n"
                f"  rewritten={row['rewritten_query']}\n"
                f"  retrieved={row['retrieved_count']} citations={row['citation_count']} ids={row['retrieved_ids']}"
            )
    finally:
        if not args.keep_session and not args.session_id:
            services.memory.session_delete(session_id, args.workspace_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"conversation_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results written to {filename}")


if __name__ == "__main__":
    main()
