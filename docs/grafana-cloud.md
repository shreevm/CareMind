# Grafana Cloud monitoring

CareMind exposes Prometheus metrics at `/metrics`. The standalone Alloy collector
scrapes them every 15 seconds and sends them to Grafana Cloud. Redis Cloud remains
the independent application cache; these monitoring settings do not change it.

## Account configuration

Create or open your Grafana Cloud stack. In its Prometheus/Cloud Metrics connection
instructions, find the remote write endpoint (ending in `/api/prom/push`), metrics
username, and an access policy token with `metrics:write` permission for that stack.
The metrics token is separate from a Grafana service account token or Redis password.

See [Grafana's remote write guide](https://grafana.com/docs/learning-paths/prom-remote-write/)
and [Alloy remote write documentation](https://grafana.com/docs/alloy/latest/reference/components/prometheus/prometheus.remote_write/).

From the repository root, copy `.env.grafana.example` to `.env.grafana` if it does not
already exist, then enter the three Grafana values in that file. Do not paste tokens
into chat or commit them. Compose reads this file explicitly via `--env-file`.

## Run with the local app

Docker Desktop must be running. Keep your frontend and backend running as usual.
The collector reaches the backend at `host.docker.internal:8002`. If Docker cannot
reach a backend bound to loopback, restart Uvicorn with `--host 0.0.0.0 --port 8002`.
That binding also allows LAN access, subject to your firewall.

If API basic authentication is enabled, set `CAREMIND_METRICS_USERNAME` and
`CAREMIND_METRICS_PASSWORD` to match the backend credentials.

```powershell
docker compose --env-file .env.grafana -f compose.grafana-cloud.yml up -d
docker compose --env-file .env.grafana -f compose.grafana-cloud.yml logs --tail 50 alloy
```

The collector starts independently; it does not start or stop your API, Redis,
or frontend. It needs to stay running to send new metrics. After editing its
environment file, run the `up -d` command again to apply changes.

## Dashboard and verification

In Grafana Cloud, use Dashboards > New > Import and upload
`ops/grafana/cloud/caremind-overview.json`. Select your Cloud Metrics Prometheus
data source when prompted.

In Explore, select the same data source and query:

```promql
up{job="caremind-api"}
```

A value of `1` confirms the backend scrape succeeds. Then query
`caremind_uptime_seconds` and make a request in CareMind. Chat, cache, and guardrail
panels populate as those events occur; empty panels before activity are expected.
Allow a minute for the first samples. A running container alone does not prove
successful cloud delivery: check Explore and the collector logs.

## Deployed app

Set `CAREMIND_METRICS_TARGET` to the API hostname, without a scheme or `/metrics`,
and `CAREMIND_METRICS_SCHEME=https`. Run the collector on an always-on host for
continuous collection; a laptop collector stops sending when the laptop sleeps.
The same Compose file can run on that host with its own `.env.grafana`.

To stop only the collector:

```powershell
docker compose --env-file .env.grafana -f compose.grafana-cloud.yml down
```
