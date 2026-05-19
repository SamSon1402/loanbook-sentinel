"""Sector concentration analysis.

Computes Herfindahl-Hirschman Index (HHI) on outstanding exposure by sector
and breaks down risk-weighted exposure per sector × emirate cell.

HHI scale (industry convention):
    < 0.15  : unconcentrated
    0.15 - 0.25 : moderate concentration
    > 0.25  : high concentration (escalate to risk committee)
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from sentinel.db.models import Loan, LoanStatus
from sentinel.schemas.portfolio import SectorConcentration, SectorExposure


def compute_concentration(loans: list[Loan]) -> SectorConcentration:
    """Compute HHI and per-(sector, emirate) exposure table."""
    active = [
        loan for loan in loans
        if loan.status in (LoanStatus.ACTIVE, LoanStatus.IN_GRACE)
    ]
    if not active:
        return SectorConcentration(hhi=0.0, exposures=[])

    total = sum(loan.outstanding_aed for loan in active)
    if total <= 0:
        return SectorConcentration(hhi=0.0, exposures=[])

    # HHI on sector level
    by_sector: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for loan in active:
        by_sector[loan.sector] += loan.outstanding_aed

    hhi = sum((float(amt / total)) ** 2 for amt in by_sector.values())

    # detailed grid by (sector, emirate)
    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"outstanding": Decimal("0"), "count": 0, "pd_sum": 0.0, "pd_w": 0.0}
    )
    for loan in active:
        cell = cells[(loan.sector, loan.emirate)]
        cell["outstanding"] += loan.outstanding_aed
        cell["count"] += 1
        cell["pd_sum"] += loan.pd_current * float(loan.outstanding_aed)
        cell["pd_w"] += float(loan.outstanding_aed)

    exposures: list[SectorExposure] = [
        SectorExposure(
            sector=sector,
            emirate=emirate,
            outstanding_aed=cell["outstanding"],
            loan_count=cell["count"],
            weighted_pd=round(cell["pd_sum"] / cell["pd_w"], 4) if cell["pd_w"] > 0 else 0.0,
        )
        for (sector, emirate), cell in sorted(
            cells.items(),
            key=lambda kv: kv[1]["outstanding"],
            reverse=True,
        )
    ]
    return SectorConcentration(hhi=round(hhi, 4), exposures=exposures)
