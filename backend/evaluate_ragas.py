import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app import app, get_services


REPORTS_DIR = Path(__file__).resolve().parent / "eval_reports" / "ragas"
DEFAULT_CASES_PATH = Path(__file__).with_name("ragas_questions.jsonl")
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
    if route == "medical_education":
        chunks = services.tools.medical_education_search(question, top_k=min(top_k, 3))
    elif route == "compare":
        documents = services.store.list_documents(workspace_id)
        chunks = []
        for document in documents[:2]:
            chunks.extend(services.store.get_document_chunks(document.document_id)[:2])
    elif route == "retrieve":
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
    return {
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


def try_run_ragas(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall
    except Exception as exc:
        return None, f"RAGAS import unavailable: {exc}"

    try:
        evaluation_dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["response"],
                    "reference": row["reference"],
                }
                for row in rows
            ]
        )
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        )
        if hasattr(result, "to_pandas"):
            return {"rows": result.to_pandas().to_dict(orient="records")}, None
        if hasattr(result, "to_dict"):
            return result.to_dict(), None
        return dict(result), None
    except Exception as exc:
        return None, f"RAGAS execution skipped/failed: {exc}"


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
    lines.extend(["", "## Cases", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### {row['user_input']}",
                "",
                f"- Route: `{row['route']}` expected `{row['expected_route']}`",
                f"- Latency: `{row['latency_ms']} ms`",
                f"- Retrieved contexts: `{len(row['retrieved_contexts'])}`",
                f"- Retrieval term recall: `{row['fallback']['retrieval_reference_term_recall']}`",
                f"- Generation term recall: `{row['fallback']['generation_reference_term_recall']}`",
                f"- Faithfulness proxy: `{row['fallback']['faithfulness_proxy_supported_sentence_rate']}`",
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
        response = client.post(
            "/chat",
            json={
                "message": case["question"],
                "session_id": run_id,
                "workspace_id": workspace_id,
                "top_k": case.get("top_k", 5),
            },
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        body = response.json()
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
            "route": body["route"],
            "response": body["answer"],
            "reference": case["reference"],
            "reference_terms": case.get("reference_terms", []),
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_metadata": contexts,
            "citations": body.get("citations", []),
            "tool_calls": body.get("tool_calls", []),
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
        rows.append(row)

    ragas_result, ragas_error = try_run_ragas(rows)
    fallback_summary = fallback_metrics(rows)
    payload = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "rows": rows,
        "fallback_summary": fallback_summary,
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
