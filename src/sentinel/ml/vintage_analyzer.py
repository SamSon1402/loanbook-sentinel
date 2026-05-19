"""Vintage decay analysis.

Builds expected-vs-actual probability-of-default curves per origination cohort.
The 'expected' curve is the average PD across the entire training book; the
'actual' curve is realized DPD≥90 events at each month-on-book.

A positive delta (actual > expected) on early MoBs is a leading indicator that
something is wrong with the underwriting model and warrants a CASE retrain.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from sentinel.db.models import Loan, LoanStatus
from sentinel.schemas.portfolio import VintageCurve, VintagePoint


def _months_between(a: date, b: date) -> int:
    """Calendar months between two dates, floored to 0."""
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def build_vintage_curve(loans: list[Loan], cohort: str, *, as_of: date) -> VintageCurve:
    """Build a vintage curve for one origination cohort (YYYY-MM)."""
    cohort_loans = [loan for loan in loans if loan.cohort == cohort]
    if not cohort_loans:
        return VintageCurve(cohort=cohort, points=[])

    # group by months-on-book
    by_mob: dict[int, list[Loan]] = defaultdict(list)
    for loan in cohort_loans:
        mob = _months_between(loan.origination_date, as_of)
        by_mob[mob].append(loan)

    points: list[VintagePoint] = []
    cumulative_defaults = 0
    cohort_size = len(cohort_loans)

    for mob in sorted(by_mob.keys()):
        loans_at_mob = by_mob[mob]
        defaults_this_mob = sum(
            1 for loan in loans_at_mob if loan.status == LoanStatus.DEFAULTED
        )
        cumulative_defaults += defaults_this_mob
        actual_pd = cumulative_defaults / cohort_size

        # expected = average pd_predicted across the cohort, blended by MoB
        # (a real shop would model this from history; we use a Weibull-like proxy)
        avg_pred = float(np.mean([loan.pd_predicted for loan in cohort_loans]))
        expected_pd = float(min(1.0, avg_pred * (1.0 - np.exp(-mob / 6.0))))

        points.append(
            VintagePoint(
                months_on_book=mob,
                expected_pd=round(expected_pd, 4),
                actual_pd=round(actual_pd, 4),
                cohort_size=cohort_size,
            )
        )

    return VintageCurve(cohort=cohort, points=points)


def compute_pd_drift(loans: list[Loan]) -> float:
    """Aggregate PD drift across the active book.

    Returns the mean ratio of `pd_current / pd_predicted` across active loans,
    minus 1.0 — i.e. how much worse (positive) or better (negative) the book is
    behaving than the underwriting model said it would.
    """
    active = [
        loan for loan in loans
        if loan.status in (LoanStatus.ACTIVE, LoanStatus.IN_GRACE)
        and loan.pd_predicted > 0
    ]
    if not active:
        return 0.0
    ratios = [loan.pd_current / loan.pd_predicted for loan in active]
    return float(np.mean(ratios) - 1.0)
