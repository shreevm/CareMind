import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
if os.getenv("CAREMIND_EVAL_USE_CONFIGURED_VECTOR", "").lower() != "true":
    os.environ["CAREMIND_VECTOR_BACKEND"] = "sqlite"

from backend.app import app, get_services


REPORTS_DIR = Path(__file__).resolve().parent / "eval_runs" / "pubmedqa"
K_VALUES = (1, 3, 5)
LABELS = ("yes", "no", "maybe")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CareMind RAG on PubMedQA.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to PubMedQA JSON, for example data/test_set.json or ori_pqal.json.",
    )
    parser.add_argument("--limit", type=int, default=int(os.getenv("PUBMEDQA_LIMIT", "25")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("PUBMEDQA_TOP_K", "5")))
    parser.add_argument("--workspace-id", default="")
    parser.add_argument(
        "--app-route",
        action="store_true",
        help="Use /chat/stream instead of direct retrieval + LLM generation.",
    )
    return parser.parse_args()


def auto_dataset_path() -> Path:
    explicit = os.getenv("PUBMEDQA_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.getenv("PUBMEDQA_DIR", "data/benchmarks/pubmedqa"))
    candidates = [
        root / "test_set.json",
        root / "ori_pqal.json",
        root / "pqal.json",
        root / "data" / "test_set.json",
        root / "data" / "ori_pqal.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find PubMedQA data. Set PUBMEDQA_PATH or pass --dataset. "
        "Example: --dataset E:\\datasets\\pubmedqa\\data\\test_set.json"
    )


def load_pubmedqa(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = [{"pmid": str(pmid), **case} for pmid, case in payload.items()]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported PubMedQA payload in {path}")

    cases = []
    for index, item in enumerate(items):
        pmid = str(item.get("pmid") or item.get("PMID") or item.get("id") or index)
        question = str(item.get("QUESTION") or item.get("question") or "").strip()
        contexts = item.get("CONTEXTS") or item.get("contexts") or item.get("context") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        long_answer = str(item.get("LONG_ANSWER") or item.get("long_answer") or item.get("answer") or "").strip()
        final_decision = str(item.get("final_decision") or item.get("FINAL_DECISION") or item.get("label") or "").lower()
        final_decision = final_decision if final_decision in LABELS else ""
        if not question or not contexts:
            continue
        cases.append(
            {
                "pmid": pmid,
                "question": question,
                "contexts": [str(context).strip() for context in contexts if str(context).strip()],
                "reference": long_answer,
                "label": final_decision,
            }
        )
        if limit and len(cases) >= limit:
            break
    return cases


def index_cases(cases: list[dict[str, Any]], workspace_id: str) -> dict[str, str]:
    services = get_services()
    indexed: dict[str, str] = {}
    for case in cases:
        text = "\n\n".join(
            [
                f"PubMedQA PMID: {case['pmid']}",
                f"Research question: {case['question']}",
                "PubMed abstract context:",
                *case["contexts"],
            ]
        )
        document = services.ingestion.ingest_text(
            filename=f"pubmedqa-{case['pmid']}.txt",
            text=text,
            workspace_id=workspace_id,
        )
        indexed[case["pmid"]] = document.document_id
    services.memory.cache_clear_workspace(workspace_id)
    return indexed


def benchmark_question(case: dict[str, Any]) -> str:
    return (
        "Using only the retrieved PubMed abstract evidence, answer this biomedical research question. "
        "Cite every factual sentence. End with exactly one line in this format: "
        "Final decision: yes/no/maybe.\n\n"
        f"Question: {case['question']}"
    )


def run_direct_case(case: dict[str, Any], workspace_id: str, top_k: int) -> dict[str, Any]:
    services = get_services()
    question = benchmark_question(case)
    started = time.perf_counter()
    chunks = services.tools.document_search(question, workspace_id=workspace_id, top_k=top_k)
    answer = services.agent.llm.answer(question=question, chunks=chunks, history=[])
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "answer": answer,
        "route": "direct_rag",
        "citations": [],
        "trace": {"retrieved_chunks": [chunk_to_dict(chunk) for chunk in chunks]},
        "latency_ms": latency_ms,
        "time_to_first_token_ms": None,
    }


def run_app_case(client: TestClient, case: dict[str, Any], workspace_id: str, top_k: int, session_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    body = chat_case(
        client,
        {
            "message": benchmark_question(case),
            "session_id": session_id,
            "workspace_id": workspace_id,
            "top_k": top_k,
            "bypass_cache": True,
        },
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "answer": body["answer"],
        "route": body["route"],
        "citations": body.get("citations", []),
        "trace": body.get("trace", {}),
        "latency_ms": latency_ms,
        "time_to_first_token_ms": (body.get("trace") or {}).get("time_to_first_token_ms"),
    }


def chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "page": chunk.page,
        "score": chunk.score,
        "text": chunk.text,
    }


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for raw_event in text.replace("\r\n", "\n").split("\n\n"):
        if not raw_event.strip():
            continue
        event = "message"
        data_parts = []
        for line in raw_event.split("\n"):
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_parts.append(line.removeprefix("data:").strip())
        events.append({"event": event, "data": json.loads("".join(data_parts)) if data_parts else {}})
    return events


def chat_case(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    with client.stream("POST", "/chat/stream", json=payload) as response:
        response.raise_for_status()
        text = "".join(response.iter_text())
    events = parse_sse_events(text)
    errors = [event["data"].get("error") for event in events if event["event"] == "error"]
    if errors:
        raise RuntimeError(errors[-1] or "Streaming PubMedQA eval failed")
    finals = [event["data"] for event in events if event["event"] == "final"]
    if not finals:
        raise RuntimeError("Streaming PubMedQA eval ended without a final event")
    return finals[-1]


def predicted_label(answer: str) -> str:
    final_line = re.search(r"final decision\s*:\s*(yes|no|maybe)\b", answer, flags=re.IGNORECASE)
    if final_line:
        return final_line.group(1).lower()
    first_label = re.search(r"\b(yes|no|maybe)\b", answer, flags=re.IGNORECASE)
    return first_label.group(1).lower() if first_label else ""


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def token_f1(expected: str, actual: str) -> float:
    expected_tokens = tokenize(expected)
    actual_tokens = tokenize(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = len(expected_tokens & actual_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def sentence_support_rate(answer: str, contexts: list[str]) -> float:
    context_terms = tokenize(" ".join(contexts))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if len(sentence.strip()) > 24
    ]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        sentence_terms = tokenize(sentence)
        if not sentence_terms:
            continue
        if len(sentence_terms & context_terms) / max(1, len(sentence_terms)) >= 0.35:
            supported += 1
    return supported / len(sentences)


def retrieval_metrics(retrieved: list[dict[str, Any]], gold_pmid: str) -> dict[str, float]:
    ranks = [
        index
        for index, chunk in enumerate(retrieved, start=1)
        if gold_pmid in str(chunk.get("document_name", "")) or gold_pmid in str(chunk.get("text", ""))
    ]
    first_rank = ranks[0] if ranks else None
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        hit = bool(first_rank and first_rank <= k)
        metrics[f"gold_doc_recall@{k}"] = 1.0 if hit else 0.0
        metrics[f"gold_doc_precision@{k}"] = round((1.0 if hit else 0.0) / k, 3)
    metrics["mrr"] = round(1 / first_rank, 3) if first_rank else 0.0
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["gold_label"]]
    label_hits = [row["gold_label"] == row["predicted_label"] for row in labeled]
    latencies = [row["latency_ms"] for row in rows]
    ttft = [row["time_to_first_token_ms"] for row in rows if row["time_to_first_token_ms"] is not None]
    summary = {
        "cases": len(rows),
        "label_accuracy": round(sum(label_hits) / len(label_hits), 3) if label_hits else None,
        "label_macro_f1": macro_f1(labeled),
        "answer_token_f1": average(row["answer_token_f1"] for row in rows),
        "faithfulness_proxy_supported_sentence_rate": average(row["faithfulness_proxy_supported_sentence_rate"] for row in rows),
        "latency_ms_avg": average(latencies),
        "time_to_first_token_ms_avg": average(ttft) if ttft else None,
    }
    for k in K_VALUES:
        summary[f"gold_doc_recall@{k}"] = average(row["retrieval"][f"gold_doc_recall@{k}"] for row in rows)
        summary[f"gold_doc_precision@{k}"] = average(row["retrieval"][f"gold_doc_precision@{k}"] for row in rows)
    summary["mrr"] = average(row["retrieval"]["mrr"] for row in rows)
    return summary


def average(values: Any) -> float:
    value_list = [float(value) for value in values if value is not None]
    return round(sum(value_list) / len(value_list), 3) if value_list else 0.0


def macro_f1(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    scores = []
    for label in LABELS:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 3)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CareMind PubMedQA Benchmark",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Workspace: `{payload['workspace_id']}`",
        f"- Pipeline: `{payload['pipeline']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Cases", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### PMID {row['pmid']}",
                "",
                f"- Question: {row['question']}",
                f"- Gold label: `{row['gold_label']}`",
                f"- Predicted label: `{row['predicted_label']}`",
                f"- Route: `{row['route']}`",
                f"- Latency: `{row['latency_ms']} ms`",
                f"- Gold doc recall@5: `{row['retrieval']['gold_doc_recall@5']}`",
                f"- MRR: `{row['retrieval']['mrr']}`",
                f"- Answer token F1: `{row['answer_token_f1']}`",
                f"- Faithfulness proxy: `{row['faithfulness_proxy_supported_sentence_rate']}`",
                "",
                "Answer:",
                "",
                row["answer"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset or auto_dataset_path()
    cases = load_pubmedqa(dataset, args.limit)
    if not cases:
        raise RuntimeError(f"No valid PubMedQA cases loaded from {dataset}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_id = args.workspace_id or f"pubmedqa-{run_id}"
    indexed_documents = index_cases(cases, workspace_id)
    client = TestClient(app)
    rows = []

    for case in cases:
        if args.app_route:
            result = run_app_case(client, case, workspace_id, args.top_k, run_id)
            retrieved = (result.get("trace") or {}).get("retrieved_chunks", [])
        else:
            result = run_direct_case(case, workspace_id, args.top_k)
            retrieved = result["trace"]["retrieved_chunks"]

        contexts = [chunk.get("text", "") for chunk in retrieved]
        answer = result["answer"]
        rows.append(
            {
                "pmid": case["pmid"],
                "indexed_document_id": indexed_documents.get(case["pmid"]),
                "question": case["question"],
                "reference": case["reference"],
                "gold_label": case["label"],
                "predicted_label": predicted_label(answer),
                "answer": answer,
                "route": result["route"],
                "latency_ms": result["latency_ms"],
                "time_to_first_token_ms": result["time_to_first_token_ms"],
                "retrieved_contexts": contexts,
                "retrieved_context_metadata": retrieved,
                "retrieval": retrieval_metrics(retrieved, case["pmid"]),
                "answer_token_f1": round(token_f1(case["reference"], answer), 3) if case["reference"] else None,
                "faithfulness_proxy_supported_sentence_rate": round(sentence_support_rate(answer, contexts), 3),
            }
        )

    payload = {
        "run_id": run_id,
        "dataset": str(dataset),
        "workspace_id": workspace_id,
        "pipeline": "app_route" if args.app_route else "direct_rag",
        "top_k": args.top_k,
        "summary": summarize(rows),
        "rows": rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{run_id}.json"
    markdown_path = REPORTS_DIR / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(markdown_path, payload)

    print(json.dumps(payload["summary"], indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
