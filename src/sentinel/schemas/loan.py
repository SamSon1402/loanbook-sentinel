"""Pydantic schemas for loans."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.db.models import LoanStatus


class LoanRead(BaseModel):
    """Public loan representation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    merchant_id: str
    merchant_name: str
    sector: str
    emirate: str
    country: str
    principal_aed: Decimal
    outstanding_aed: Decimal
    apr: float
    tenor_months: int
    origination_date: date
    cohort: str
    status: LoanStatus
    pd_predicted: float
    pd_current: float
    dpd: int
    case_score: int


class LoanCreate(BaseModel):
    """Body for POST /v1/loans (used by seed script + onboarding service)."""

    merchant_id: str = Field(min_length=1, max_length=32)
    merchant_name: str = Field(min_length=1, max_length=128)
    sector: str
    emirate: str
    country: str = Field(min_length=2, max_length=2)
    principal_aed: Decimal = Field(gt=0)
    apr: float = Field(gt=0, lt=1)
    tenor_months: int = Field(gt=0, le=120)
    origination_date: date
