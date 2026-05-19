# LoanBook-Sentinel

> Real-time loan portfolio health & anomaly detection for GCC SME lending.
> Built to demonstrate the production stack Orbii uses: FastAPI · ONNX · Prometheus · Seldon-ready · K8s.

`loanbook-sentinel` ingests bank-feed events for active SME loans, computes
vintage-decay curves and sector-concentration risk, and runs an Isolation Forest
(exported to ONNX) over behavioral features to flag anomalies *before* they
turn into defaults.

---

## What's in this repo

| Layer            | Stack                                              | Where                          |
| ---------------- | -------------------------------------------------- | ------------------------------ |
| API              | FastAPI · Pydantic v2 · async SQLAlchemy 2.0       | `src/sentinel/api/`            |
| ML serving       | scikit-learn Isolation Forest → ONNX → onnxruntime | `src/sentinel/ml/`             |
| Domain analytics | Vintage curves · HHI concentration · DPD scanner   | `src/sentinel/ml/`, `services/`|
| Persistence      | SQLAlchemy + SQLite (dev) / Postgres (prod)        | `src/sentinel/db/`             |
| Observability    | Prometheus exporter on `/metrics` · structlog JSON | `src/sentinel/core/`           |
| Packaging        | Multi-stage Dockerfile · docker-compose dev stack  | `Dockerfile`, `docker-compose.yml` |
| Orchestration    | K8s manifests + HPA + ServiceMonitor               | `k8s/`                         |
| CI               | GitHub Actions: lint · pytest · docker build · k8s validate | `.github/workflows/ci.yml` |
| Tests            | pytest + httpx async client                        | `tests/`                       |

---

## Quickstart — 60 seconds

```bash
# 1. Install
make install

# 2. Generate synthetic GCC SME portfolio + train anomaly model (sklearn → ONNX)
make bootstrap

# 3. Run the API
make run
# → http://localhost:8000/docs
```

Or the full Prometheus + Grafana + API stack:

```bash
docker compose up -d
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

---

## Key endpoints

```
POST   /v1/loans/events       Ingest a bank-feed event (repayment, NSF, overdraft…)
GET    /v1/portfolio/health   KPIs: AUM, active count, 30D default rate, PD drift
GET    /v1/portfolio/vintage  Cohort decay curve (expected vs actual PD per cohort)
GET    /v1/portfolio/sectors  HHI sector concentration + risk × exposure heatmap
GET    /v1/anomalies          Active anomaly alerts (paginated)
POST   /v1/anomalies/scan     Trigger a synchronous behavioral scan
GET    /health                Liveness
GET    /ready                 Readiness (DB + model)
GET    /metrics               Prometheus exposition format
```

OpenAPI / Swagger UI is auto-generated at `/docs`.

---

## Architecture

```
                 ┌───────────────────────────────────────────────────────────┐
                 │                       FastAPI app                          │
                 │                                                            │
   bank-feed ──▶ │  /events ─▶ EventService ─▶ FeatureBuilder ─▶ Detector ─▶ │ ─▶ Postgres
   webhooks      │                                  │                         │
                 │                                  └─▶ AnomalyService ──────▶│ ─▶ Slack / PagerDuty
                 │                                                            │
   cron (5min)──▶│  scanner.run() ─▶ batch scan all active loans              │
                 │                                                            │
                 │  /metrics ◀────── prometheus_client instrumentation        │
                 └───────────────────────────────────────────────────────────┘
                                            │
                       ┌────────────────────┼────────────────────┐
                       ▼                    ▼                    ▼
                 Prometheus ────────▶  Grafana            Seldon Core
                                                          (drop-in for ONNX model)
```

The anomaly detector is exported as **ONNX** so it can be served identically
through `onnxruntime` (current) or **Seldon Core / TorchServe** behind a
K8s `InferenceService`.

---

## Repository layout

```
src/sentinel/
├── main.py                FastAPI app factory, lifespan, middleware
├── config.py              pydantic-settings, 12-factor env config
├── api/
│   ├── deps.py            Dependency injection (db, detector)
│   └── routes/            health, loans, portfolio, anomalies
├── core/
│   ├── logging.py         structlog JSON logging
│   └── metrics.py         Prometheus counters / histograms
├── db/
│   ├── base.py            Declarative base
│   ├── models.py          Loan, LoanEvent, Anomaly (SQLAlchemy 2.0)
│   └── session.py         Async engine + session factory
├── schemas/               Pydantic v2 request/response models
├── ml/
│   ├── features.py        42 cashflow features from a transaction window
│   ├── anomaly_detector.py  ONNX Runtime wrapper around Isolation Forest
│   ├── vintage_analyzer.py  Expected vs actual cohort PD curves
│   └── concentration.py     HHI + risk-weighted sector exposure
└── services/
    ├── portfolio_service.py
    └── scanner.py         Background anomaly scanner (anyio task group)

scripts/
├── generate_synthetic_data.py   GCC SME portfolio generator (sectors, emirates, POS feeds)
├── train_anomaly_model.py       sklearn IsolationForest → skl2onnx → models/anomaly.onnx
└── seed_db.py                   Load synthetic data into the DB

tests/                    pytest + httpx.AsyncClient
.github/workflows/ci.yml  lint → mypy → pytest → docker → k8s manifests validate
k8s/                      Deployment + Service + ConfigMap + HPA + ServiceMonitor
```

---

## Tech choices, justified

- **ONNX over a raw pickle.** Same artifact runs under `onnxruntime`, Seldon Core,
  or TorchServe. Decouples training framework from serving runtime — matches
  Orbii's stated stack.
- **Async SQLAlchemy 2.0.** FastAPI is async-native; sync DB calls block the
  event loop under load. Async ORM keeps the request budget for ONNX inference.
- **pydantic-settings.** Single source of truth for env vars, types validated
  at startup. Plays nicely with K8s ConfigMaps + Secrets.
- **structlog JSON logs.** Loki / Datadog / Cloud Logging can parse without
  regex.
- **prometheus_client.** Native histogram for inference latency + counter for
  events ingested by sector. ServiceMonitor in `k8s/` exposes this to a
  Prometheus Operator.
- **Isolation Forest over deep-learning.** Tabular cashflow features, sparse
  anomalies, no time for fine-tuning a transformer. Iso Forest is the right
  baseline + interpretable + fast (<2 ms p95 over ONNX).

---

## Tests

```bash
make test       # pytest -v
make lint       # ruff + mypy
make coverage   # pytest --cov=src/sentinel
```

---


