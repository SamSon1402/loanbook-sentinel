"""Pydantic schemas for portfolio analytics."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class PortfolioHealth(BaseModel):
    aum_aed: Decimal
    active_loans: int
    defaulted_loans: int
    default_rate_30d: float
    avg_case_score: float
    pd_drift: float


class VintagePoint(BaseModel):
    months_on_book: int
    expected_pd: float
    actual_pd: float
    cohort_size: int


class VintageCurve(BaseModel):
    cohort: str
    points: list[VintagePoint]


class SectorExposure(BaseModel):
    sector: str
    emirate: str
    outstanding_aed: Decimal
    loan_count: int
    weighted_pd: float


class SectorConcentration(BaseModel):
    hhi: float
    exposures: list[SectorExposure]
