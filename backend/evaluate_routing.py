"""Routing-only benchmark with fixed document metadata and isolated sessions.

No retrieval or generation is executed. Timings include preflight and context
resolution but exclude storage/network I/O, which is stubbed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.caremind.agent import CareMindAgent
from backend.caremind.agents.supervisor import SupervisorAgent
from backend.caremind.config import Settings
from backend.caremind.safety import SafetyLayer
from backend.caremind.schemas import ChatRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('backend/eval_questions.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    results = []
    for case in cases:
        agent = CareMindAgent.__new__(CareMindAgent)
        agent.memory = Mock()
        agent.memory.load.return_value = []
        agent.memory.context_get.return_value = {}
        agent.tools = Mock()
        agent.tools.store.list_documents.return_value = [
            SimpleNamespace(document_id=f'doc-{i}', filename=f'synthetic-lab-report-{name}.txt')
            for i, name in enumerate(['baseline', 'followup'])
        ]
        agent.tools.store.list_images.return_value = []
        agent.tools.external_literature_available.return_value = False
        agent.tools.medlineplus_available.return_value = False
        agent.supervisor_agent = SupervisorAgent()
        agent.safety = SafetyLayer()
        agent.llm = SimpleNamespace(settings=Settings())
        request = ChatRequest(**{'message': case['question'], **case.get('request', {})})
        start = time.perf_counter()
        preflight = agent._preflight_response(request)
        if preflight:
            route = preflight.route
        else:
            state = agent._observe_node({'request': request, 'started_at': start})
            route = agent._planned_route(state['request'].message, state['observations'])
            route = agent._public_route(route, state['request'])
        results.append(dict(question=case['question'], expected=case['expected_route'],
                            actual=route, correct=route == case['expected_route'],
                            latency_ms=(time.perf_counter() - start) * 1000))
    timings = sorted(item['latency_ms'] for item in results)
    payload = dict(scope='Routing only; isolated sessions; two synthetic document metadata records; stubbed storage; no retrieval/generation',
                   cases_sha256=hashlib.sha256(raw).hexdigest(), results=results,
                   summary=dict(total=len(results), correct=sum(r['correct'] for r in results),
                                accuracy=sum(r['correct'] for r in results) / len(results),
                                p50_ms=statistics.median(timings),
                                p95_ms=timings[math.ceil(.95 * len(timings)) - 1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
