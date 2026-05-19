"""End-to-end API tests via httpx ASGI client."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from httpx import AsyncClient

from sentinel.db.models import Loan, LoanStatus


@pytest.mark.asyncio
async def test_health_and_ready(client: AsyncClient) -> None:
    r1 = await client.get("/health")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok"}

    r2 = await client.get("/ready")
    assert r2.status_code == 200
    body = r2.json()
    assert body["ready"] is True
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["model"] == "ok"


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus(client: AsyncClient) -> None:
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "sentinel_events_ingested_total" in body
    assert "sentinel_inference_latency_seconds" in body


@pytest.mark.asyncio
async def test_ingest_event_404_for_unknown_loan(client: AsyncClient) -> None:
    payload = {
        "loan_id": "DOES_NOT_EXIST",
        "event_type": "POS_INFLOW",
        "amount_aed": "100.00",
        "ts": datetime.now(UTC).isoformat(),
    }
    r = await client.post("/v1/loans/events", json=payload)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_ingest_event_happy_path(client: AsyncClient) -> None:
    # Seed a single loan directly into the DB the app is using.
    from sentinel.db.session import SessionLocal

    async with SessionLocal() as s:
        s.add(
            Loan(
                id="L0001",
                merchant_id="M0001",
                merchant_name="Test SME",
                sector="Retail",
                emirate="Dubai",
                country="AE",
                principal_aed=Decimal("50000"),
                outstanding_aed=Decimal("48000"),
                apr=0.18,
                tenor_months=12,
                origination_date=date(2026, 1, 1),
                cohort="2026-01",
                status=LoanStatus.ACTIVE,
                pd_predicted=0.03,
                pd_current=0.04,
                dpd=0,
                case_score=720,
            )
        )
        await s.commit()

    payload = {
        "loan_id": "L0001",
        "event_type": "POS_INFLOW",
        "amount_aed": "1250.50",
        "ts": datetime.now(UTC).isoformat(),
    }
    r = await client.post("/v1/loans/events", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["accepted"] is True
    assert "event_id" in body
    assert isinstance(body["anomaly_score"], float)


@pytest.mark.asyncio
async def test_portfolio_endpoints_work_on_empty_book(client: AsyncClient) -> None:
    r = await client.get("/v1/portfolio/health")
    assert r.status_code == 200
    body = r.json()
    assert body["active_loans"] == 0
    assert body["defaulted_loans"] == 0

    r = await client.get("/v1/portfolio/sectors")
    assert r.status_code == 200
    assert r.json()["hhi"] == 0.0


@pytest.mark.asyncio
async def test_anomalies_list_empty(client: AsyncClient) -> None:
    r = await client.get("/v1/anomalies")
    assert r.status_code == 200
    assert r.json() == []
