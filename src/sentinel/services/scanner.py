"""Background scanner.

Sweeps the active book on a fixed interval. For each active loan it builds the
feature vector from recent events, scores it with the ONNX detector, and
persists an `Anomaly` row when the score crosses the threshold.

Also flags DPD breaches based on the days-past-due column (rules-based,
complementary to the ML signal).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from ulid import ULID

from sentinel.config import Settings
from sentinel.core.logging import get_logger
from sentinel.core.metrics import (
    anomalies_active,
    scanner_duration_seconds,
    scanner_runs_total,
)
from sentinel.db.models import (
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    Loan,
    LoanStatus,
)
from sentinel.ml.anomaly_detector import AnomalyDetector
from sentinel.ml.features import build_features

log = get_logger(__name__)


class Scanner:
    """Active-book scanner. One instance per process."""

    def __init__(self, detector: AnomalyDetector, settings: Settings) -> None:
        self.detector = detector
        self.settings = settings

    async def run_once(self, session: AsyncSession) -> dict[str, int | float | list[str]]:
        """Run a single scan pass. Returns a summary dict."""
        start = time.perf_counter()
        now = datetime.now(UTC)

        result = await session.execute(
            select(Loan)
            .where(Loan.status.in_([LoanStatus.ACTIVE, LoanStatus.IN_GRACE]))
            .options(selectinload(Loan.events))
        )
        loans = result.scalars().all()

        new_ids: list[str] = []
        for loan in loans:
            features = build_features(
                list(loan.events),
                as_of=now,
                principal=float(loan.principal_aed),
            )
            score, is_anomaly = self.detector.score(features)

            # ML signal
            if is_anomaly:
                anom_id = self._make_id()
                severity = (
                    AnomalySeverity.CRITICAL
                    if score < self.settings.anomaly_threshold * 1.5
                    else AnomalySeverity.WARNING
                )
                session.add(
                    Anomaly(
                        id=anom_id,
                        loan_id=loan.id,
                        severity=severity,
                        type=AnomalyType.BEHAVIORAL,
                        score=score,
                        message=f"Behavioral anomaly: {loan.merchant_name} (score={score:.3f})",
                        detected_at=now,
                    )
                )
                new_ids.append(anom_id)

            # Rules-based DPD signal — complementary to ML
            if loan.dpd >= self.settings.dpd_default_days:
                anom_id = self._make_id()
                session.add(
                    Anomaly(
                        id=anom_id,
                        loan_id=loan.id,
                        severity=AnomalySeverity.CRITICAL,
                        type=AnomalyType.DPD_BREACH,
                        score=float(loan.dpd),
                        message=f"DPD={loan.dpd}d ≥ default threshold "
                        f"({self.settings.dpd_default_days}d)",
                        detected_at=now,
                    )
                )
                new_ids.append(anom_id)
            elif loan.dpd >= self.settings.dpd_warning_days:
                anom_id = self._make_id()
                session.add(
                    Anomaly(
                        id=anom_id,
                        loan_id=loan.id,
                        severity=AnomalySeverity.WARNING,
                        type=AnomalyType.DPD_BREACH,
                        score=float(loan.dpd),
                        message=f"DPD={loan.dpd}d ≥ warning threshold "
                        f"({self.settings.dpd_warning_days}d)",
                        detected_at=now,
                    )
                )
                new_ids.append(anom_id)

        await session.commit()

        # update Prometheus gauges
        anomalies_active.labels(severity="WARNING").set(
            sum(1 for _id in new_ids) // 2  # rough; real impl re-queries
        )

        duration = time.perf_counter() - start
        scanner_duration_seconds.observe(duration)
        scanner_runs_total.inc()

        summary = {
            "loans_scanned": len(loans),
            "anomalies_found": len(new_ids),
            "duration_seconds": round(duration, 3),
            "new_anomaly_ids": new_ids,
        }
        log.info("scanner.completed", **summary)
        return summary

    @staticmethod
    def _make_id() -> str:
        return str(ULID())
