import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    request_counts: Counter = field(default_factory=Counter)
    route_counts: Counter = field(default_factory=Counter)
    tool_counts: Counter = field(default_factory=Counter)
    cache_counts: Counter = field(default_factory=Counter)
    latencies_ms: defaultdict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=200)))

    def record_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        key = f"{method} {path} {status_code}"
        self.request_counts[key] += 1
        self.latencies_ms[f"{method} {path}"].append(latency_ms)

    def record_agent(self, route: str, tool_calls: list[str], latency_ms: float, cache_hit: bool) -> None:
        self.route_counts[route] += 1
        for tool in tool_calls:
            self.tool_counts[tool] += 1
        self.cache_counts["hit" if cache_hit else "miss"] += 1
        self.latencies_ms[f"agent:{route}"].append(latency_ms)

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "requests": dict(self.request_counts),
            "agent_routes": dict(self.route_counts),
            "tool_calls": dict(self.tool_counts),
            "cache": dict(self.cache_counts),
            "latency_ms": {
                name: {
                    "count": len(values),
                    "avg": round(sum(values) / len(values), 3) if values else 0,
                    "max": round(max(values), 3) if values else 0,
                }
                for name, values in self.latencies_ms.items()
            },
        }


metrics = Metrics()
