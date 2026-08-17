import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


def _labels(**items: str | int | bool) -> str:
    if not items:
        return ""
    parts = []
    for key, value in items.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')
    return "{" + ",".join(parts) + "}"


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    request_counts: Counter = field(default_factory=Counter)
    route_counts: Counter = field(default_factory=Counter)
    tool_counts: Counter = field(default_factory=Counter)
    cache_counts: Counter = field(default_factory=Counter)
    cache_lookup_counts: Counter = field(default_factory=Counter)
    retrieval_reuse_counts: Counter = field(default_factory=Counter)
    fallback_counts: Counter = field(default_factory=Counter)
    guardrail_counts: Counter = field(default_factory=Counter)
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

    def record_fallback(self, component: str, provider: str, error_type: str) -> None:
        self.fallback_counts[f"{component}|{provider}|{error_type}"] += 1

    def record_guardrail(self, guardrail: str, action: str) -> None:
        self.guardrail_counts[f"{guardrail}|{action}"] += 1

    def record_cache_lookup(self, source: str, result: str, reason: str = "") -> None:
        self.cache_lookup_counts[f"{source}|{result}|{reason}"] += 1

    def record_retrieval_reuse(self, result: str, reason: str = "") -> None:
        self.retrieval_reuse_counts[f"{result}|{reason}"] += 1

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "requests": dict(self.request_counts),
            "agent_routes": dict(self.route_counts),
            "tool_calls": dict(self.tool_counts),
            "cache": dict(self.cache_counts),
            "cache_lookups": dict(self.cache_lookup_counts),
            "retrieval_reuse": dict(self.retrieval_reuse_counts),
            "fallbacks": dict(self.fallback_counts),
            "guardrails": dict(self.guardrail_counts),
            "latency_ms": {
                name: {
                    "count": len(values),
                    "avg": round(sum(values) / len(values), 3) if values else 0,
                    "max": round(max(values), 3) if values else 0,
                }
                for name, values in self.latencies_ms.items()
            },
        }

    def to_prometheus(self) -> str:
        lines: list[str] = []
        lines.extend(
            [
                "# HELP caremind_uptime_seconds Seconds since process metrics started.",
                "# TYPE caremind_uptime_seconds gauge",
                f"caremind_uptime_seconds {time.time() - self.started_at:.3f}",
                "# HELP caremind_http_requests_total HTTP requests handled by route, method, and status.",
                "# TYPE caremind_http_requests_total counter",
            ]
        )
        for key, count in self.request_counts.items():
            method, path, status_code = key.split(" ", 2)
            lines.append(f"caremind_http_requests_total{_labels(method=method, path=path, status_code=status_code)} {count}")

        lines.extend(
            [
                "# HELP caremind_agent_routes_total Agent responses by route.",
                "# TYPE caremind_agent_routes_total counter",
            ]
        )
        for route, count in self.route_counts.items():
            lines.append(f"caremind_agent_routes_total{_labels(route=route)} {count}")

        lines.extend(
            [
                "# HELP caremind_tool_calls_total Tool calls by tool name.",
                "# TYPE caremind_tool_calls_total counter",
            ]
        )
        for tool, count in self.tool_counts.items():
            lines.append(f"caremind_tool_calls_total{_labels(tool=tool)} {count}")

        lines.extend(
            [
                "# HELP caremind_cache_events_total Cache hits and misses.",
                "# TYPE caremind_cache_events_total counter",
            ]
        )
        for result, count in self.cache_counts.items():
            lines.append(f"caremind_cache_events_total{_labels(result=result)} {count}")

        lines.extend(
            [
                "# HELP caremind_cache_lookup_total Cache lookup outcomes by source and reason.",
                "# TYPE caremind_cache_lookup_total counter",
            ]
        )
        for key, count in self.cache_lookup_counts.items():
            source, result, reason = key.split("|", 2)
            lines.append(f"caremind_cache_lookup_total{_labels(source=source, result=result, reason=reason)} {count}")

        lines.extend(
            [
                "# HELP caremind_retrieval_reuse_total Retrieval reuse decisions by outcome and reason.",
                "# TYPE caremind_retrieval_reuse_total counter",
            ]
        )
        for key, count in self.retrieval_reuse_counts.items():
            result, reason = key.split("|", 1)
            lines.append(f"caremind_retrieval_reuse_total{_labels(result=result, reason=reason)} {count}")

        lines.extend(
            [
                "# HELP caremind_fallbacks_total LLM and embedding provider fallbacks.",
                "# TYPE caremind_fallbacks_total counter",
            ]
        )
        for key, count in self.fallback_counts.items():
            component, provider, error_type = key.split("|", 2)
            lines.append(
                f"caremind_fallbacks_total{_labels(component=component, provider=provider, error_type=error_type)} {count}"
            )

        lines.extend(
            [
                "# HELP caremind_guardrail_events_total Guardrail actions by type.",
                "# TYPE caremind_guardrail_events_total counter",
            ]
        )
        for key, count in self.guardrail_counts.items():
            guardrail, action = key.split("|", 1)
            lines.append(f"caremind_guardrail_events_total{_labels(guardrail=guardrail, action=action)} {count}")

        lines.extend(
            [
                "# HELP caremind_latency_ms Request and agent latency gauges over the in-memory window.",
                "# TYPE caremind_latency_ms gauge",
            ]
        )
        for name, values in self.latencies_ms.items():
            if not values:
                continue
            labels = _latency_labels(name)
            lines.append(f"caremind_latency_ms{_labels(**labels, statistic='avg')} {sum(values) / len(values):.3f}")
            lines.append(f"caremind_latency_ms{_labels(**labels, statistic='max')} {max(values):.3f}")
            lines.append(f"caremind_latency_ms{_labels(**labels, statistic='count')} {len(values)}")
        return "\n".join(lines) + "\n"


def _latency_labels(name: str) -> dict[str, str]:
    if name.startswith("agent:"):
        return {"kind": "agent", "route": name.removeprefix("agent:")}
    try:
        method, path = name.split(" ", 1)
    except ValueError:
        return {"kind": "other", "name": name}
    return {"kind": "http", "method": method, "path": path}


metrics = Metrics()
