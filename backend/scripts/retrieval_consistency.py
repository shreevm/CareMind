"""
Retrieval consistency test for CareMind.

Measures how much the retrieved document set for a fixed query changes
depending on how many prior turns are in the session. This script exercises
CareMind's observe/context-resolution step and DocumentRAGAgent.prepare(), but
does not call LLM generation.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.caremind.agent import CareMindAgent
from backend.caremind.config import Settings
from backend.caremind.embeddings import EmbeddingClient
from backend.caremind.llm import LLMClient
from backend.caremind.memory import ConversationMemory
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest
from backend.caremind.store import SQLiteStore
from backend.caremind.tools import DocumentTools
from backend.caremind.vectorstore import VectorStore


TEST_QUERIES = [
    "What does my latest cholesterol panel mean?",
    "Are there any drug interactions with my current prescriptions?",
    "What did my last MRI report say about my knee?",
]

FILLER_TURNS = [
    "Can you explain what a blood pressure reading means?",
    "What's a normal resting heart rate?",
    "How often should I get a checkup?",
    "What does BMI actually measure?",
    "Can stress affect blood sugar levels?",
    "What is the difference between a cold and the flu?",
    "How long does it take antibiotics to work?",
    "What foods help lower cholesterol?",
    "Is it normal to feel tired after a vaccine?",
    "What does a CBC test check for?",
]

SESSION_DEPTHS = [0, 5, 10]


@dataclass(frozen=True)
class RetrievalResult:
    ids: set[str]
    route: str
    rewritten_query: str
    tool_calls: list[str]
    retrieved_count: int
    kept_count: int | None
    filtered_count: int | None


class RetrievalConsistencyHarness:
    def __init__(self, *, workspace_id: str, top_k: int, identity: str) -> None:
        self.workspace_id = workspace_id
        self.top_k = top_k
        self.identity = identity
        self.settings = Settings()
        self.store = SQLiteStore(self.settings.sqlite_path)
        self.embeddings = EmbeddingClient(self.settings)
        self.vectorstore = VectorStore(self.settings, self.store)
        self.llm = LLMClient(self.settings)
        self.tools = DocumentTools(self.embeddings, self.vectorstore, self.store)
        self.memory = ConversationMemory(self.settings, self.store)
        self.agent = CareMindAgent(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            safety=SafetyLayer(),
        )

    def run_retrieval(self, query: str, history: list[str]) -> RetrievalResult:
        session_id = f"retrieval-consistency-{uuid.uuid4()}"
        try:
            for turn in history:
                self.memory.append(session_id, self.workspace_id, "user", turn)
                self.memory.append(session_id, self.workspace_id, "assistant", "Recorded.")

            request = ChatRequest(
                message=query,
                session_id=session_id,
                workspace_id=self.workspace_id,
                top_k=self.top_k,
                debug=True,
                bypass_cache=True,
            )
            state = self.agent._observe_node({"request": request, "started_at": time.perf_counter()})
            route = self.agent._planned_route(state["request"].message, state.get("observations", {}))
            updates = self.agent.document_rag_agent.prepare(
                state["request"],
                tool_calls=[],
                trace_steps=[],
            )
            chunks = updates.get("chunks", [])
            retrieved_chunks = updates.get("retrieved_chunks", [])
            identities = {
                self._identity_for_chunk(chunk, fallback)
                for chunk, fallback in zip(chunks, retrieved_chunks, strict=False)
            }
            if not identities:
                identities = {
                    self._identity_for_trace(item)
                    for item in retrieved_chunks
                    if isinstance(item, dict)
                }
            trace = self._last_retrieval_trace(updates)
            return RetrievalResult(
                ids={item for item in identities if item},
                route=route,
                rewritten_query=state["request"].message,
                tool_calls=list(updates.get("tool_calls", [])),
                retrieved_count=len(retrieved_chunks),
                kept_count=trace.get("kept_count"),
                filtered_count=trace.get("filtered_count"),
            )
        finally:
            self.memory.session_delete(session_id, self.workspace_id)

    def _identity_for_chunk(self, chunk: Any, fallback: dict[str, Any]) -> str:
        if self.identity == "document":
            return str(getattr(chunk, "document_id", "") or fallback.get("document_id", ""))
        return str(getattr(chunk, "chunk_id", "") or fallback.get("chunk_id", ""))

    def _identity_for_trace(self, item: dict[str, Any]) -> str:
        key = "document_id" if self.identity == "document" else "chunk_id"
        return str(item.get(key, ""))

    def _last_retrieval_trace(self, updates: dict[str, Any]) -> dict[str, Any]:
        for step in reversed(updates.get("trace_steps", [])):
            if isinstance(step, dict) and step.get("node") == "document_rag_agent":
                return step
        return {}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure retrieval consistency across session depths.")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--identity", choices=["chunk", "document"], default="chunk")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harness = RetrievalConsistencyHarness(
        workspace_id=args.workspace_id,
        top_k=args.top_k,
        identity=args.identity,
    )
    rows: list[dict[str, Any]] = []

    for query in TEST_QUERIES:
        baseline: set[str] | None = None
        for depth in SESSION_DEPTHS:
            history = FILLER_TURNS[:depth]
            result = harness.run_retrieval(query, history)
            if depth == 0:
                baseline = result.ids
                overlap = 1.0
            else:
                overlap = jaccard(baseline or set(), result.ids)

            rows.append(
                {
                    "query": query,
                    "session_depth": depth,
                    "route": result.route,
                    "rewritten_query": result.rewritten_query,
                    "tool_calls": ";".join(result.tool_calls),
                    "retrieved_count": result.retrieved_count,
                    "kept_count": result.kept_count,
                    "filtered_count": result.filtered_count,
                    "retrieved_ids": ";".join(sorted(result.ids)),
                    "jaccard_vs_fresh": round(overlap, 3),
                }
            )
            print(
                f"[depth={depth:>2}] '{query[:40]}...' "
                f"route={result.route} retrieved={len(result.ids)} "
                f"overlap_vs_fresh={overlap:.2f}"
            )

    if not rows:
        print("No rows generated.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"retrieval_consistency_{timestamp}.csv"
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    depth_scores: dict[int, list[float]] = {}
    for row in rows:
        depth_scores.setdefault(int(row["session_depth"]), []).append(float(row["jaccard_vs_fresh"]))

    print("\n--- Summary ---")
    for depth, scores in sorted(depth_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"Session depth {depth}: avg overlap vs fresh = {avg:.2%}")
    print(f"\nFull results written to {filename}")


if __name__ == "__main__":
    main()
