"""Prometheus metrics registry.

We expose counters for ingest volume and gauges for portfolio KPIs, plus a
histogram around model inference. The metrics are scraped by Prometheus via the
ServiceMonitor in `k8s/servicemonitor.yaml`.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

# ── ingest ──────────────────────────────────────────────────────────────────
events_ingested_total = Counter(
    "sentinel_events_ingested_total",
    "Total bank-feed events ingested by sector and event type.",
    labelnames=("sector", "event_type"),
    registry=registry,
)

# ── inference ───────────────────────────────────────────────────────────────
inference_latency_seconds = Histogram(
    "sentinel_inference_latency_seconds",
    "Latency of the ONNX anomaly detector per call.",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    registry=registry,
)

inference_total = Counter(
    "sentinel_inference_total",
    "Total inference calls labelled by outcome.",
    labelnames=("outcome",),  # 'anomaly' | 'normal' | 'error'
    registry=registry,
)

# ── portfolio KPIs ──────────────────────────────────────────────────────────
portfolio_aum_aed = Gauge(
    "sentinel_portfolio_aum_aed",
    "Total assets-under-management in AED.",
    registry=registry,
)

portfolio_active_loans = Gauge(
    "sentinel_portfolio_active_loans",
    "Count of loans in ACTIVE status.",
    registry=registry,
)

portfolio_default_rate = Gauge(
    "sentinel_portfolio_default_rate_30d",
    "Trailing-30-day default rate (0..1).",
    registry=registry,
)

anomalies_active = Gauge(
    "sentinel_anomalies_active",
    "Open anomaly alerts by severity.",
    labelnames=("severity",),
    registry=registry,
)

# ── scanner ─────────────────────────────────────────────────────────────────
scanner_runs_total = Counter(
    "sentinel_scanner_runs_total",
    "Number of background scanner runs completed.",
    registry=registry,
)

scanner_duration_seconds = Histogram(
    "sentinel_scanner_duration_seconds",
    "End-to-end duration of a single scan over the active book.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=registry,
)
