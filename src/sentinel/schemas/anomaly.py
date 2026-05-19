"""Pydantic schemas for anomalies."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from sentinel.db.models import AnomalySeverity, AnomalyType


class AnomalyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    loan_id: str
    severity: AnomalySeverity
    type: AnomalyType
    score: float
    message: str
    detected_at: datetime
    resolved_at: datetime | None


class ScanResult(BaseModel):
    """Response from POST /v1/anomalies/scan."""

    loans_scanned: int
    anomalies_found: int
    duration_seconds: float
    new_anomaly_ids: list[str]
