import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
if os.getenv("CAREMIND_EVAL_USE_CONFIGURED_VECTOR", "").lower() != "true":
    os.environ["CAREMIND_VECTOR_BACKEND"] = "sqlite"

from backend.app import app, get_services


REPORTS_DIR = Path(__file__).resolve().parent / "eval_runs" / "ragas"
DEFAULT_CASES_PATH = Path(__file__).with_name("ragas_questions.jsonl")
K_VALUES = (1, 3, 5)
ROUTE_ALIASES = {
    "clinical_document_qa": "retrieve",
    "report_comparison": "compare",
    "medical_knowledge_qa": "medical_education",
}
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
    "include",
    "includes",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def term_recall(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for term in terms if term.lower() in lowered)
    return hits / len(terms)


def term_hit_count(terms: list[str], text: str) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def eval_route(route: str) -> str:
    return ROUTE_ALIASES.get(route, route)


def token_f1(expected: str, actual: str) -> float:
    expected_tokens = tokenize(expected)
    actual_tokens = tokenize(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = len(expected_tokens & actual_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def is_relevant_context(context: str, terms: list[str]) -> bool:
    if not terms:
        return False
    threshold = max(1, min(3, len(terms) // 4))
    return term_hit_count(terms, context) >= threshold


def retrieval_metrics_at_k(contexts: list[str], terms: list[str], k_values: tuple[int, ...] = K_VALUES) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        top_contexts = contexts[:k]
        relevant_count = sum(1 for context in top_contexts if is_relevant_context(context, terms))
        metrics[f"context_recall@{k}"] = round(term_recall(terms, " ".join(top_contexts)), 3)
        metrics[f"precision@{k}"] = round(relevant_count / k, 3)

    first_relevant_rank = next(
        (index for index, context in enumerate(contexts, start=1) if is_relevant_context(context, terms)),
        None,
    )
    metrics["mrr"] = round(1 / first_relevant_rank, 3) if first_relevant_rank else 0.0
    return metrics


def answer_quality_metrics(question: str, reference: str, answer: str, terms: list[str], contexts: list[str]) -> dict[str, float]:
    correctness_recall = term_recall(terms, answer)
    correctness_f1 = token_f1(reference, answer)
    return {
        "answer_relevance_proxy": round(token_f1(question, answer), 3),
        "faithfulness_proxy_supported_sentence_rate": round(sentence_support_rate(answer, contexts), 3),
        "answer_correctness_proxy": round((correctness_recall + correctness_f1) / 2, 3),
        "answer_reference_term_recall": round(correctness_recall, 3),
    }


def average_metric(rows: list[dict[str, Any]], section: str, key: str) -> float:
    values = [row[section][key] for row in rows if key in row.get(section, {})]
    return round(sum(values) / len(values), 3) if values else 0.0


def performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    ttft_values = [
        float(row["time_to_first_token_ms"])
        for row in rows
        if row.get("time_to_first_token_ms") is not None
    ]
    return {
        "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
        "time_to_first_token_ms_avg": round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else None,
        "time_to_first_token_ms_p50": percentile(ttft_values, 0.50) if ttft_values else None,
        "time_to_first_token_ms_p95": percentile(ttft_values, 0.95) if ttft_values else None,
        "time_to_first_token_coverage": round(len(ttft_values) / len(rows), 3) if rows else 0.0,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return round(ordered[index], 2)


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
        overlap = len(sentence_terms & context_terms) / max(1, len(sentence_terms))
        if overlap >= 0.35:
            supported += 1
    return supported / len(sentences)


def retrieve_contexts(question: str, route: str, workspace_id: str, top_k: int = 5) -> list[dict[str, Any]]:
    services = get_services()
    normalized_route = eval_route(route)
    if normalized_route == "medical_education":
        chunks = services.tools.medical_education_search(question, top_k=min(top_k, 3))
    elif normalized_route == "compare":
        documents = services.store.list_documents(workspace_id)
        chunks = []
        for document in documents[:2]:
            chunks.extend(services.store.get_document_chunks(document.document_id)[:2])
    elif normalized_route == "retrieve":
        chunks = services.tools.document_search(question, workspace_id=workspace_id, top_k=top_k)
    else:
        chunks = []
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "score": chunk.score,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


def fallback_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    retrieval_recalls = []
    generation_recalls = []
    faithfulness_proxies = []
    route_hits = []
    citation_hits = []
    context_counts = []
    for row in rows:
        contexts = row["retrieved_contexts"]
        joined_context = " ".join(contexts)
        terms = row.get("reference_terms", [])
        retrieval_recalls.append(term_recall(terms, joined_context))
        generation_recalls.append(term_recall(terms, row["response"]))
        faithfulness_proxies.append(sentence_support_rate(row["response"], contexts))
        route_hits.append(float(row.get("route") == row.get("expected_route")))
        citation_hits.append(float(bool(row.get("citations")) if contexts else True))
        context_counts.append(len(contexts))
    summary = {
        "route_accuracy": round(sum(route_hits) / len(route_hits), 3),
        "citation_pass_rate": round(sum(citation_hits) / len(citation_hits), 3),
        "retrieval_reference_term_recall": round(sum(retrieval_recalls) / len(retrieval_recalls), 3),
        "generation_reference_term_recall": round(sum(generation_recalls) / len(generation_recalls), 3),
        "faithfulness_proxy_supported_sentence_rate": round(
            sum(faithfulness_proxies) / len(faithfulness_proxies),
            3,
        ),
        "average_retrieved_contexts": round(sum(context_counts) / len(context_counts), 2),
    }
    for k in K_VALUES:
        summary[f"context_recall@{k}"] = average_metric(rows, "retrieval", f"context_recall@{k}")
        summary[f"precision@{k}"] = average_metric(rows, "retrieval", f"precision@{k}")
    summary["mrr"] = average_metric(rows, "retrieval", "mrr")
    summary["answer_relevance_proxy"] = average_metric(rows, "generation", "answer_relevance_proxy")
    summary["answer_correctness_proxy"] = average_metric(rows, "generation", "answer_correctness_proxy")
    return summary


def install_ragas_vertexai_compat() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types

    from langchain_community.llms.vertexai import VertexAI

    module = types.ModuleType("langchain_community.chat_models.vertexai")
    module.ChatVertexAI = VertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module


def build_ragas_llm() -> Any | None:
    settings = get_services().settings
    if not settings.nvidia_api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        model=settings.nvidia_chat_model,
        temperature=0,
    )


def try_run_ragas(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    if os.getenv("CAREMIND_RAGAS_FULL_ENABLED", "").lower() != "true":
        return None, "Full RAGAS disabled. Set CAREMIND_RAGAS_FULL_ENABLED=true to run evaluator-LLM metrics."

    try:
        install_ragas_vertexai_compat()
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import AnswerRelevancy, Faithfulness, FactualCorrectness, LLMContextRecall
        from ragas.run_config import RunConfig
    except Exception as exc:
        return None, f"RAGAS import unavailable: {exc}"

    try:
        full_limit = int(os.getenv("CAREMIND_RAGAS_FULL_LIMIT", "2"))
        ragas_rows = rows[:full_limit] if full_limit > 0 else rows
        evaluation_dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["response"],
                    "reference": row["reference"],
                }
                for row in ragas_rows
            ]
        )
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), AnswerRelevancy(), Faithfulness(), FactualCorrectness()],
            llm=build_ragas_llm(),
            run_config=RunConfig(timeout=60, max_retries=1, max_workers=2),
            show_progress=False,
        )
        if hasattr(result, "to_pandas"):
            return {
                "evaluated_cases": len(ragas_rows),
                "rows": result.to_pandas().to_dict(orient="records"),
            }, None
        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        else:
            payload = dict(result)
        payload["evaluated_cases"] = len(ragas_rows)
        return payload, None
    except Exception as exc:
        return None, f"RAGAS execution skipped/failed: {exc}"


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
        data = json.loads("".join(data_parts)) if data_parts else {}
        events.append({"event": event, "data": data})
    return events


def chat_case(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("CAREMIND_EVAL_USE_STREAM", "true").lower() != "true":
        response = client.post("/chat", json=payload)
        response.raise_for_status()
        return response.json()

    with client.stream("POST", "/chat/stream", json=payload) as response:
        response.raise_for_status()
        text = "".join(response.iter_text())
    events = parse_sse_events(text)
    for event in events:
        if event["event"] == "error":
            raise RuntimeError(event["data"].get("error") or "Streaming eval failed")
    final_events = [event for event in events if event["event"] == "final"]
    if not final_events:
        raise RuntimeError("Streaming eval ended without a final event")
    return final_events[-1]["data"]


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CareMind RAGAS Evaluation",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Workspace: `{payload['workspace_id']}`",
        f"- RAGAS status: `{payload['ragas_status']}`",
        "",
        "## Local Retrieval/Generation Diagnostics",
        "",
    ]
    for key, value in payload["fallback_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## System Performance", ""])
    for key, value in payload["performance_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Cases", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### {row['user_input']}",
                "",
                f"- Route: `{row['route']}` expected `{row['expected_route']}`",
                f"- Actual app route: `{row['actual_route']}`",
                f"- Latency: `{row['latency_ms']} ms`",
                f"- Time to first token: `{row['time_to_first_token_ms']} ms`",
                f"- Retrieved contexts: `{len(row['retrieved_contexts'])}`",
                f"- Context recall@1: `{row['retrieval']['context_recall@1']}`",
                f"- Context recall@3: `{row['retrieval']['context_recall@3']}`",
                f"- Context recall@5: `{row['retrieval']['context_recall@5']}`",
                f"- Precision@1: `{row['retrieval']['precision@1']}`",
                f"- Precision@3: `{row['retrieval']['precision@3']}`",
                f"- Precision@5: `{row['retrieval']['precision@5']}`",
                f"- MRR: `{row['retrieval']['mrr']}`",
                f"- Answer relevance proxy: `{row['generation']['answer_relevance_proxy']}`",
                f"- Faithfulness proxy: `{row['generation']['faithfulness_proxy_supported_sentence_rate']}`",
                f"- Answer correctness proxy: `{row['generation']['answer_correctness_proxy']}`",
                "",
                "Response:",
                "",
                row["response"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    client = TestClient(app)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_id = f"ragas-{run_id}"
    client.post("/demo/seed", params={"workspace_id": workspace_id})

    rows = []
    for case in cases:
        started_at = time.perf_counter()
        body = chat_case(
            client,
            {
                "message": case["question"],
                "session_id": run_id,
                "workspace_id": workspace_id,
                "top_k": case.get("top_k", 5),
            },
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        contexts = retrieve_contexts(
            case["question"],
            body["route"],
            workspace_id,
            top_k=case.get("top_k", 5),
        )
        retrieved_contexts = [context["text"] for context in contexts]
        row = {
            "user_input": case["question"],
            "expected_route": case["expected_route"],
            "route": eval_route(body["route"]),
            "actual_route": body["route"],
            "response": body["answer"],
            "reference": case["reference"],
            "reference_terms": case.get("reference_terms", []),
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_metadata": contexts,
            "citations": body.get("citations", []),
            "tool_calls": body.get("tool_calls", []),
            "time_to_first_token_ms": (body.get("trace") or {}).get("time_to_first_token_ms"),
            "latency_ms": round(latency_ms, 2),
        }
        row["fallback"] = {
            "retrieval_reference_term_recall": round(
                term_recall(row["reference_terms"], " ".join(retrieved_contexts)),
                3,
            ),
            "generation_reference_term_recall": round(
                term_recall(row["reference_terms"], row["response"]),
                3,
            ),
            "faithfulness_proxy_supported_sentence_rate": round(
                sentence_support_rate(row["response"], retrieved_contexts),
                3,
            ),
        }
        row["retrieval"] = retrieval_metrics_at_k(retrieved_contexts, row["reference_terms"])
        row["generation"] = answer_quality_metrics(
            row["user_input"],
            row["reference"],
            row["response"],
            row["reference_terms"],
            retrieved_contexts,
        )
        rows.append(row)

    ragas_result, ragas_error = try_run_ragas(rows)
    fallback_summary = fallback_metrics(rows)
    payload = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "rows": rows,
        "fallback_summary": fallback_summary,
        "performance_summary": performance_summary(rows),
        "ragas_status": "completed" if ragas_result is not None else "fallback_only",
        "ragas_result": ragas_result,
        "ragas_error": ragas_error,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{run_id}.json"
    ragas_dataset_path = REPORTS_DIR / f"{run_id}.ragas_dataset.jsonl"
    markdown_path = REPORTS_DIR / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ragas_dataset_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["response"],
                    "reference": row["reference"],
                },
                ensure_ascii=False,
            )
            for row in rows
        ),
        encoding="utf-8",
    )
    write_markdown(markdown_path, payload)

    print(json.dumps(fallback_summary, indent=2))
    if ragas_error:
        print(ragas_error)
    print(f"JSON report: {json_path}")
    print(f"RAGAS dataset: {ragas_dataset_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
