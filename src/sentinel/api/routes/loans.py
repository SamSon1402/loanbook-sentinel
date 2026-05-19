"""Loan event ingestion.

POST /v1/loans/events is the primary write path: bank-feed connectors and the
WhatsApp/Crisp bot post here. Each event is persisted and (when relevant)
scored inline by the ONNX detector — sub-200ms p95.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from ulid import ULID

from sentinel.api.deps import DetectorDep, SessionDep
from sentinel.core.logging import get_logger
from sentinel.core.metrics import events_ingested_total
from sentinel.db.models import (
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    Loan,
    LoanEvent,
)
from sentinel.ml.features import build_features
from sentinel.schemas.event import EventIngest, IngestResult

router = APIRouter(prefix="/v1/loans", tags=["loans"])
log = get_logger(__name__)


@router.post(
    "/events",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a bank-feed event",
)
async def ingest_event(
    payload: EventIngest,
    session: SessionDep,
    detector: DetectorDep,
) -> IngestResult:
    """Persist an event and inline-score the loan for anomalies."""
    # Load the loan with eager events for the feature window.
    result = await session.execute(
        select(Loan)
        .where(Loan.id == payload.loan_id)
        .options(selectinload(Loan.events))
    )
    loan = result.scalar_one_or_none()
    if loan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"loan_id {payload.loan_id!r} not found",
        )

    # Persist the event.
    event = LoanEvent(
        id=str(ULID()),
        loan_id=loan.id,
        event_type=payload.event_type,
        amount_aed=payload.amount_aed,
        ts=payload.ts,
        raw_payload=payload.raw_payload,
    )
    session.add(event)
    await session.flush()
    loan.events.append(event)  # in-memory, for feature building below

    events_ingested_total.labels(
        sector=loan.sector, event_type=payload.event_type.value
    ).inc()

    # Inline ONNX scoring.
    features = build_features(
        list(loan.events),
        as_of=datetime.now(UTC),
        principal=float(loan.principal_aed),
    )
    score, is_anomaly = detector.score(features)

    new_anomaly_msg: str | None = None
    if is_anomaly:
        anomaly = Anomaly(
            id=str(ULID()),
            loan_id=loan.id,
            severity=AnomalySeverity.WARNING,
            type=AnomalyType.BEHAVIORAL,
            score=score,
            message=f"Behavioral anomaly on event ingest ({loan.merchant_name})",
            detected_at=datetime.now(UTC),
        )
        session.add(anomaly)
        new_anomaly_msg = anomaly.message
        log.info(
            "anomaly.flagged",
            loan_id=loan.id,
            score=score,
            event_type=payload.event_type.value,
        )

    await session.commit()

    return IngestResult(
        event_id=event.id,
        accepted=True,
        anomaly_detected=is_anomaly,
        anomaly_score=score,
        message=new_anomaly_msg,
    )
