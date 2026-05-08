# Monitoring stack (local)

A docker-compose stack that runs the Heart Disease API alongside
Prometheus + Grafana for end-to-end observability.

```
api  ──/metrics──▶ prometheus  ──datasource──▶ grafana
 │
 └── stdout JSON logs (collect with Loki / fluentd in prod)
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | API + Prometheus + Grafana services |
| `prometheus.yml` | Scrape config (15s interval, scrapes `api:8000/metrics`) |
| `grafana/provisioning/datasources/prometheus.yml` | Auto-wires Prometheus as default datasource |
| `grafana/provisioning/dashboards/dashboards.yml` | Auto-loads dashboards from `/var/lib/grafana/dashboards` |
| `grafana/dashboards/heart-disease-api.json` | Pre-built dashboard (10 panels) |

## Run

```bash
# From repo root
docker compose -f monitoring/docker-compose.yml up --build
```

Then:

| Service    | URL                       | Credentials   |
|------------|---------------------------|---------------|
| API docs   | http://localhost:8000/docs | —             |
| Prometheus | http://localhost:9090      | —             |
| Grafana    | http://localhost:3000      | admin / admin |

The dashboard appears under **Dashboards → ML → Heart Disease API**.

## Generate traffic for the dashboard

```bash
# Tiny load generator
for i in {1..200}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
         "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,
         "slope":3,"ca":0,"thal":6}' > /dev/null
  sleep 0.1
done
```

## Custom metrics exposed by the API

| Metric | Type | Labels | What it tells you |
|--------|------|--------|-------------------|
| `heart_predictions_total` | counter | `label`, `model_version` | Predicted-class throughput; class-balance drift |
| `heart_prediction_probability` | histogram | `model_version` | Score distribution drift |
| `heart_prediction_latency_seconds` | histogram | `model_version` | Model inference latency (SLO source) |
| `heart_prediction_errors_total` | counter | `model_version`, `error_type` | Inference error rate by exception class |

Plus all standard HTTP metrics from `prometheus-fastapi-instrumentator`
(`http_requests_total`, `http_request_duration_seconds`, etc.).

## Structured logs

Every request emits a one-line JSON record on stdout, e.g.

```json
{"ts":"2026-05-05T14:31:02.418+00:00","level":"INFO","logger":"heart-api",
 "service":"heart-disease-api","version":"v1.0.0","message":"prediction",
 "event":"prediction","request_id":"a4f1c2d3","prediction":1,
 "label":"Disease","probability":0.8124,"confidence":0.8124,
 "latency_ms":3.41,"model_version":"v1.0.0"}
```

Set `LOG_FORMAT=plain` to switch to human-readable formatting for local
debugging.

## Tear down

```bash
docker compose -f monitoring/docker-compose.yml down -v
```
