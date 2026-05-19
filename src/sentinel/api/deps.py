"""FastAPI dependencies — DB session and the singleton AnomalyDetector."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.db.session import get_session
from sentinel.ml.anomaly_detector import AnomalyDetector
from sentinel.services.scanner import Scanner


def get_detector(request: Request) -> AnomalyDetector:
    """Resolve the AnomalyDetector singleton built at app startup."""
    return request.app.state.detector  # type: ignore[no-any-return]


def get_scanner(request: Request) -> Scanner:
    """Resolve the Scanner singleton built at app startup."""
    return request.app.state.scanner  # type: ignore[no-any-return]


SessionDep = Annotated[AsyncSession, Depends(get_session)]
DetectorDep = Annotated[AnomalyDetector, Depends(get_detector)]
ScannerDep = Annotated[Scanner, Depends(get_scanner)]
