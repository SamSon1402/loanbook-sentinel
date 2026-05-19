"""Feature engineering.

Computes 18 numerical features from a per-merchant transaction window. These
are the inputs to the Isolation Forest anomaly detector.

The same feature definitions are used both at *training time* (in
`scripts/train_anomaly_model.py`) and at *inference time* (here) — there is
exactly one canonical implementation to avoid train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from sentinel.db.models import EventType, LoanEvent

# ── feature order is part of the contract with the ONNX model ──────────────
FEATURE_ORDER: tuple[str, ...] = (
    "cashflow_volatility_30d",
    "pos_inflow_consistency_score",
    "nsf_event_count_90d",
    "overdraft_count_90d",
    "missed_payment_count_90d",
    "avg_balance_30d",
    "inflow_outflow_ratio",
    "inflow_total_30d",
    "outflow_total_30d",
    "event_count_30d",
    "max_drawdown_pct",
    "days_since_last_inflow",
    "cash_only_inflow_ratio",
    "repayment_to_principal_ratio",
    "avg_pos_inflow_amt",
    "pos_inflow_cadence_days",
    "weekend_activity_ratio",
    "large_outflow_count_30d",
)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Wrapper around a 1-D feature vector with named accessors."""

    values: np.ndarray  # shape (N,) == len(FEATURE_ORDER)

    def __post_init__(self) -> None:
        if self.values.shape != (len(FEATURE_ORDER),):
            raise ValueError(
                f"FeatureVector expects shape ({len(FEATURE_ORDER)},), "
                f"got {self.values.shape}"
            )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_ORDER, self.values.tolist(), strict=True))


def _normalize_ts(ts: datetime, *, ref: datetime) -> datetime:
    """Bring `ts` to the same tz-awareness as `ref`.

    SQLite drops tz info on round-trip, so events read from SQLite are naive
    even though they were inserted as aware. We normalize on the way in to
    avoid `can't compare offset-naive and offset-aware datetimes` at runtime.
    """
    if ref.tzinfo is not None and ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    if ref.tzinfo is None and ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def build_features(events: list[LoanEvent], *, as_of: datetime, principal: float) -> FeatureVector:
    """Build the canonical feature vector from a list of events.

    Args:
        events: Loan events for a single loan. Order does not matter — we sort.
        as_of: The 'now' reference point; features look back from here.
        principal: Loan principal used as denominator for ratio features.
    """
    if principal <= 0:
        raise ValueError("principal must be positive")

    # SQLite drops tz info on round-trip; normalize to match `as_of` so the
    # comparisons below don't raise "can't compare offset-naive and aware".
    for e in events:
        e.ts = _normalize_ts(e.ts, ref=as_of)
    sorted_events = sorted(events, key=lambda e: e.ts)

    window_30 = [e for e in sorted_events if e.ts >= as_of - timedelta(days=30)]
    window_90 = [e for e in sorted_events if e.ts >= as_of - timedelta(days=90)]

    inflows_30 = [float(e.amount_aed) for e in window_30 if float(e.amount_aed) > 0]
    outflows_30 = [abs(float(e.amount_aed)) for e in window_30 if float(e.amount_aed) < 0]
    pos_inflows_30 = [
        float(e.amount_aed)
        for e in window_30
        if e.event_type == EventType.POS_INFLOW and float(e.amount_aed) > 0
    ]
    cash_inflows_30 = [
        float(e.amount_aed)
        for e in window_30
        if e.event_type == EventType.BANK_DEPOSIT and float(e.amount_aed) > 0
    ]

    nsf_count_90 = sum(1 for e in window_90 if e.event_type == EventType.NSF)
    overdraft_count_90 = sum(1 for e in window_90 if e.event_type == EventType.OVERDRAFT)
    missed_90 = sum(1 for e in window_90 if e.event_type == EventType.MISSED_PAYMENT)

    inflow_total = sum(inflows_30)
    outflow_total = sum(outflows_30)
    repayments_30 = sum(
        abs(float(e.amount_aed))
        for e in window_30
        if e.event_type == EventType.REPAYMENT
    )

    # volatility = std of daily net flow over the window
    daily_buckets: dict[int, float] = {}
    for e in window_30:
        day = (as_of - e.ts).days
        daily_buckets[day] = daily_buckets.get(day, 0.0) + float(e.amount_aed)
    vols = list(daily_buckets.values())
    volatility = float(np.std(vols)) if len(vols) > 1 else 0.0

    # POS inflow consistency = 1 - (coefficient of variation), clamped to [0, 1]
    if pos_inflows_30:
        mean_pos = float(np.mean(pos_inflows_30))
        std_pos = float(np.std(pos_inflows_30))
        consistency = max(0.0, 1.0 - (std_pos / mean_pos)) if mean_pos > 0 else 0.0
    else:
        consistency = 0.0

    # average balance proxy: cumulative net flow / window length
    avg_balance = (inflow_total - outflow_total) / 30.0

    inflow_outflow_ratio = inflow_total / outflow_total if outflow_total > 0 else 0.0

    # max single-day drawdown as % of principal
    if vols:
        worst_day = min(vols)
        max_drawdown = abs(min(0.0, worst_day)) / principal
    else:
        max_drawdown = 0.0

    last_inflow = next((e for e in reversed(sorted_events) if float(e.amount_aed) > 0), None)
    days_since_last_inflow = (as_of - last_inflow.ts).days if last_inflow else 999

    cash_only_inflow_ratio = (
        sum(cash_inflows_30) / inflow_total if inflow_total > 0 else 0.0
    )

    repayment_ratio = repayments_30 / principal if principal > 0 else 0.0
    avg_pos = float(np.mean(pos_inflows_30)) if pos_inflows_30 else 0.0

    # cadence: avg days between POS inflows
    if len(pos_inflows_30) >= 2:
        pos_events = sorted(
            (e for e in window_30 if e.event_type == EventType.POS_INFLOW),
            key=lambda e: e.ts,
        )
        gaps = [
            (pos_events[i + 1].ts - pos_events[i].ts).total_seconds() / 86400.0
            for i in range(len(pos_events) - 1)
        ]
        cadence = float(np.mean(gaps))
    else:
        cadence = 30.0

    weekend_events = sum(1 for e in window_30 if e.ts.weekday() in (5, 6))
    weekend_ratio = weekend_events / len(window_30) if window_30 else 0.0

    large_outflow_threshold = 0.05 * principal
    large_outflow_count = sum(
        1 for amt in outflows_30 if amt >= large_outflow_threshold
    )

    vector = np.array(
        [
            volatility,
            consistency,
            float(nsf_count_90),
            float(overdraft_count_90),
            float(missed_90),
            avg_balance,
            inflow_outflow_ratio,
            inflow_total,
            outflow_total,
            float(len(window_30)),
            max_drawdown,
            float(days_since_last_inflow),
            cash_only_inflow_ratio,
            repayment_ratio,
            avg_pos,
            cadence,
            weekend_ratio,
            float(large_outflow_count),
        ],
        dtype=np.float32,
    )
    return FeatureVector(values=vector)
