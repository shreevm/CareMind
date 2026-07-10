import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_markdown_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = [
        "# CareMind Evaluation Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Workspace: `{payload['workspace_id']}`",
        f"- Route accuracy: `{summary['route_accuracy']}`",
        f"- Citation pass rate: `{summary['citation_pass_rate']}`",
        f"- Safety disclaimer rate: `{summary['safety_disclaimer_rate']}`",
        f"- Average latency: `{summary['average_latency_ms']} ms`",
        "",
        "## Cases",
        "",
    ]
    for item in payload["results"]:
        lines.extend(
            [
                f"### {item['question']}",
                "",
                f"- Route: `{item['route']}` expected `{item['expected_route']}`",
                f"- Citations: `{item['citation_count']}`",
                f"- Tool calls: `{', '.join(item['tool_calls']) or 'none'}`",
                f"- Latency: `{item['latency_ms']} ms`",
                "",
                "Answer:",
                "",
                item["answer"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    client = TestClient(app)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace_id = f"eval-{run_id}"
    client.post("/demo/seed", params={"workspace_id": workspace_id})
    cases = load_cases(Path(__file__).with_name("eval_questions.jsonl"))
    results = []

    for case in cases:
        started_at = time.perf_counter()
        response = client.post(
            "/chat",
            json={
                "message": case["question"],
                "session_id": run_id,
                "workspace_id": workspace_id,
            },
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        body = response.json()
        route_ok = body["route"] == case["expected_route"]
        citation_ok = bool(body["citations"]) if case["requires_citation"] else True
        disclaimer_ok = "not a diagnosis or treatment plan" in body["answer"].lower()
        results.append(
            {
                "question": case["question"],
                "status_code": response.status_code,
                "route": body["route"],
                "expected_route": case["expected_route"],
                "route_ok": route_ok,
                "answer": body["answer"],
                "citation_count": len(body["citations"]),
                "citation_ok": citation_ok,
                "citations": body["citations"],
                "safety_notes": body["safety_notes"],
                "tool_calls": body["tool_calls"],
                "disclaimer_ok": disclaimer_ok,
                "latency_ms": round(latency_ms, 2),
            }
        )

    route_accuracy = sum(item["route_ok"] for item in results) / len(results)
    citation_pass_rate = sum(item["citation_ok"] for item in results) / len(results)
    safety_disclaimer_rate = sum(item["disclaimer_ok"] for item in results) / len(results)
    average_latency = sum(item["latency_ms"] for item in results) / len(results)
    payload = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "results": results,
        "summary": {
            "route_accuracy": round(route_accuracy, 3),
            "citation_pass_rate": round(citation_pass_rate, 3),
            "safety_disclaimer_rate": round(safety_disclaimer_rate, 3),
            "average_latency_ms": round(average_latency, 2),
        },
    }
    reports_dir = Path(__file__).resolve().parent / "eval_reports"
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / f"{run_id}.json"
    markdown_path = reports_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(markdown_path, payload)

    print(json.dumps(payload["summary"], indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
