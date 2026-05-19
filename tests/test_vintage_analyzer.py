"""Unit tests for vintage analyzer + concentration."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sentinel.db.models import Loan, LoanStatus
from sentinel.ml.concentration import compute_concentration
from sentinel.ml.vintage_analyzer import build_vintage_curve, compute_pd_drift


def _loan(
    *,
    cohort: str = "2026-01",
    status: LoanStatus = LoanStatus.ACTIVE,
    outstanding: float = 10_000,
    sector: str = "Retail",
    emirate: str = "Dubai",
    pd_predicted: float = 0.05,
    pd_current: float = 0.05,
    origination: date = date(2026, 1, 15),
) -> Loan:
    return Loan(
        id=f"L_{id(cohort)}_{origination.isoformat()}_{outstanding}",
        merchant_id="M",
        merchant_name="X",
        sector=sector,
        emirate=emirate,
        country="AE",
        principal_aed=Decimal(str(outstanding)),
        outstanding_aed=Decimal(str(outstanding)),
        apr=0.18,
        tenor_months=12,
        origination_date=origination,
        cohort=cohort,
        status=status,
        pd_predicted=pd_predicted,
        pd_current=pd_current,
        dpd=0,
        case_score=720,
    )


def test_vintage_empty_cohort_returns_zero_points() -> None:
    curve = build_vintage_curve([], cohort="2026-01", as_of=date(2026, 5, 1))
    assert curve.cohort == "2026-01"
    assert curve.points == []


def test_vintage_curve_increasing_with_defaults() -> None:
    loans = [_loan() for _ in range(10)]
    loans[0].status = LoanStatus.DEFAULTED
    loans[1].status = LoanStatus.DEFAULTED

    curve = build_vintage_curve(loans, cohort="2026-01", as_of=date(2026, 5, 1))
    assert curve.cohort == "2026-01"
    assert curve.points
    # cumulative default rate should be 2/10 = 0.2 at the latest MoB
    assert curve.points[-1].actual_pd == 0.2


def test_pd_drift_zero_for_perfect_book() -> None:
    loans = [_loan(pd_predicted=0.05, pd_current=0.05) for _ in range(5)]
    assert compute_pd_drift(loans) == 0.0


def test_pd_drift_positive_when_book_worse_than_predicted() -> None:
    loans = [_loan(pd_predicted=0.05, pd_current=0.10) for _ in range(5)]
    assert compute_pd_drift(loans) == 1.0  # 2x prediction → drift = +1.0


def test_concentration_empty_returns_zero_hhi() -> None:
    c = compute_concentration([])
    assert c.hhi == 0.0
    assert c.exposures == []


def test_concentration_single_sector_yields_hhi_one() -> None:
    loans = [_loan(sector="Retail", outstanding=10_000) for _ in range(5)]
    c = compute_concentration(loans)
    assert c.hhi == 1.0  # 100% in one sector


def test_concentration_even_split_lower_hhi() -> None:
    sectors = ["Retail", "F&B", "Logistics", "Tech"]
    loans = [_loan(sector=s, outstanding=10_000) for s in sectors]
    c = compute_concentration(loans)
    # 4 equal sectors → HHI = 4 * (0.25^2) = 0.25
    assert abs(c.hhi - 0.25) < 0.001
