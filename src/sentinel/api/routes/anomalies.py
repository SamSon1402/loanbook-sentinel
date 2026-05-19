"""Anomaly endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from sentinel.api.deps import ScannerDep, SessionDep
from sentinel.db.models import Anomaly, AnomalySeverity
from sentinel.schemas.anomaly import AnomalyRead, ScanResult

router = APIRouter(prefix="/v1/anomalies", tags=["anomalies"])


@router.get("", response_model=list[AnomalyRead], summary="List open anomalies")
async def list_anomalies(
    session: SessionDep,
    severity: AnomalySeverity | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_resolved: bool = Query(default=False),
) -> list[AnomalyRead]:
    """Return open anomalies, newest first. Optionally filter by severity."""
    stmt = select(Anomaly).order_by(Anomaly.detected_at.desc()).limit(limit).offset(offset)
    if not include_resolved:
        stmt = stmt.where(Anomaly.resolved_at.is_(None))
    if severity is not None:
        stmt = stmt.where(Anomaly.severity == severity)

    rows = (await session.execute(stmt)).scalars().all()
    return [AnomalyRead.model_validate(row) for row in rows]


@router.post("/scan", response_model=ScanResult, summary="Trigger a manual scan")
async def trigger_scan(session: SessionDep, scanner: ScannerDep) -> ScanResult:
    """Synchronously scan the active book. Useful for ad-hoc checks and
    integration tests; production runs this periodically via the background
    task started in `main.py`.
    """
    summary = await scanner.run_once(session)
    return ScanResult(
        loans_scanned=int(summary["loans_scanned"]),  # type: ignore[arg-type]
        anomalies_found=int(summary["anomalies_found"]),  # type: ignore[arg-type]
        duration_seconds=float(summary["duration_seconds"]),  # type: ignore[arg-type]
        new_anomaly_ids=list(summary["new_anomaly_ids"]),  # type: ignore[arg-type]
    )
