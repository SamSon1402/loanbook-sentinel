"""Portfolio analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from sentinel.api.deps import SessionDep
from sentinel.core.metrics import (
    portfolio_active_loans,
    portfolio_aum_aed,
    portfolio_default_rate,
)
from sentinel.schemas.portfolio import (
    PortfolioHealth,
    SectorConcentration,
    VintageCurve,
)
from sentinel.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


@router.get("/health", response_model=PortfolioHealth, summary="Top-line KPIs")
async def get_health(session: SessionDep) -> PortfolioHealth:
    """Return the top-line portfolio KPIs.

    Side-effect: refreshes Prometheus gauges so dashboards stay in sync.
    """
    service = PortfolioService(session)
    result = await service.health()

    portfolio_aum_aed.set(float(result.aum_aed))
    portfolio_active_loans.set(result.active_loans)
    portfolio_default_rate.set(result.default_rate_30d)

    return result


@router.get(
    "/vintage",
    response_model=VintageCurve,
    summary="Vintage decay curve for a cohort",
)
async def get_vintage(
    session: SessionDep,
    cohort: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM"),
) -> VintageCurve:
    service = PortfolioService(session)
    return await service.vintage(cohort)


@router.get(
    "/sectors",
    response_model=SectorConcentration,
    summary="HHI + sector × emirate exposure",
)
async def get_sectors(session: SessionDep) -> SectorConcentration:
    service = PortfolioService(session)
    return await service.concentration()
