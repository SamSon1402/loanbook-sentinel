"""Pydantic schemas for loan events."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sentinel.db.models import EventType


class EventIngest(BaseModel):
    """Bank-feed event payload — what an external connector POSTs to us."""

    loan_id: str = Field(min_length=1, max_length=32)
    event_type: EventType
    amount_aed: Decimal = Field(description="Positive for inflows, negative for outflows.")
    ts: datetime
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    loan_id: str
    event_type: EventType
    amount_aed: Decimal
    ts: datetime


class IngestResult(BaseModel):
    """Synchronous response from POST /v1/loans/events."""

    event_id: str
    accepted: bool
    anomaly_detected: bool
    anomaly_score: float | None = None
    message: str | None = None
