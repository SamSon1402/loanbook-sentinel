"""Portfolio service — orchestrates ML modules with DB access."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.db.models import Anomaly, Loan, LoanStatus
from sentinel.ml.concentration import compute_concentration
from sentinel.ml.vintage_analyzer import build_vintage_curve, compute_pd_drift
from sentinel.schemas.portfolio import (
    PortfolioHealth,
    SectorConcentration,
    VintageCurve,
)


class PortfolioService:
    """Read-side facade over the loan book.

    Each method runs one or two queries and delegates analytics to a pure
    function in `sentinel.ml.*`. Keeps the routes thin and the analytics
    unit-testable in isolation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def health(self) -> PortfolioHealth:
        result = await self.session.execute(
            select(
                func.coalesce(func.sum(Loan.outstanding_aed), 0),
                func.count(Loan.id).filter(Loan.status == LoanStatus.ACTIVE),
                func.count(Loan.id).filter(Loan.status == LoanStatus.DEFAULTED),
                func.coalesce(func.avg(Loan.case_score), 0.0),
            )
        )
        aum, active, defaulted, avg_case = result.one()

        thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
        recent_defaults_q = await self.session.execute(
            select(func.count(Loan.id)).where(
                Loan.status == LoanStatus.DEFAULTED,
                Loan.updated_at >= thirty_days_ago,
            )
        )
        recent_defaults = recent_defaults_q.scalar_one() or 0
        denom = max(1, (active or 0) + recent_defaults)
        default_rate = recent_defaults / denom

        # PD drift — fetch active loans cheaply
        loans = (
            (await self.session.execute(
                select(Loan).where(Loan.status.in_([LoanStatus.ACTIVE, LoanStatus.IN_GRACE]))
            ))
            .scalars()
            .all()
        )
        drift = compute_pd_drift(list(loans))

        return PortfolioHealth(
            aum_aed=Decimal(aum),
            active_loans=int(active or 0),
            defaulted_loans=int(defaulted or 0),
            default_rate_30d=round(default_rate, 4),
            avg_case_score=round(float(avg_case), 2),
            pd_drift=round(drift, 4),
        )

    async def vintage(self, cohort: str, as_of: date | None = None) -> VintageCurve:
        as_of = as_of or date.today()
        loans = (
            (await self.session.execute(select(Loan).where(Loan.cohort == cohort)))
            .scalars()
            .all()
        )
        return build_vintage_curve(list(loans), cohort=cohort, as_of=as_of)

    async def concentration(self) -> SectorConcentration:
        loans = (await self.session.execute(select(Loan))).scalars().all()
        return compute_concentration(list(loans))

    async def open_anomaly_count_by_severity(self) -> dict[str, int]:
        rows = await self.session.execute(
            select(Anomaly.severity, func.count(Anomaly.id))
            .where(Anomaly.resolved_at.is_(None))
            .group_by(Anomaly.severity)
        )
        return {sev.value: count for sev, count in rows.all()}
