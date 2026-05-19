"""Seed the database from `data/synthetic/*.parquet`.

Idempotent-ish: drops and recreates all tables, then bulk-inserts. Intended
for dev / CI bootstrap only — production uses Alembic migrations + your real
data pipeline.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy.ext.asyncio import async_sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.db.base import Base  # noqa: E402
from sentinel.db.models import EventType, Loan, LoanEvent, LoanStatus  # noqa: E402
from sentinel.db.session import engine  # noqa: E402


async def reset_schema() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def seed(data_dir: Path = Path("data/synthetic")) -> None:
    loans_df = pd.read_parquet(data_dir / "loans.parquet")
    events_df = pd.read_parquet(data_dir / "events.parquet")
    print(f"[seed] {len(loans_df):,} loans · {len(events_df):,} events")

    await reset_schema()

    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session() as session:
        chunk = 1_000
        for start in range(0, len(loans_df), chunk):
            batch = loans_df.iloc[start : start + chunk]
            session.add_all(
                Loan(
                    id=row.id,
                    merchant_id=row.merchant_id,
                    merchant_name=row.merchant_name,
                    sector=row.sector,
                    emirate=row.emirate,
                    country=row.country,
                    principal_aed=Decimal(str(row.principal_aed)),
                    outstanding_aed=Decimal(str(row.outstanding_aed)),
                    apr=float(row.apr),
                    tenor_months=int(row.tenor_months),
                    origination_date=date.fromisoformat(row.origination_date),
                    cohort=row.cohort,
                    status=LoanStatus(row.status),
                    pd_predicted=float(row.pd_predicted),
                    pd_current=float(row.pd_current),
                    dpd=int(row.dpd),
                    case_score=int(row.case_score),
                )
                for row in batch.itertuples(index=False)
            )
            await session.commit()
            print(f"[seed] loans: committed {min(start + chunk, len(loans_df)):,}")

        for start in range(0, len(events_df), chunk):
            batch = events_df.iloc[start : start + chunk]
            session.add_all(
                LoanEvent(
                    id=row.id,
                    loan_id=row.loan_id,
                    event_type=EventType(row.event_type),
                    amount_aed=Decimal(str(row.amount_aed)),
                    ts=datetime.fromisoformat(row.ts) if isinstance(row.ts, str) else row.ts,
                    raw_payload={},
                )
                for row in batch.itertuples(index=False)
            )
            await session.commit()
            if (start // chunk) % 10 == 0:
                print(f"[seed] events: committed {min(start + chunk, len(events_df)):,}")

    await engine.dispose()
    print("[seed] ✅ done")


if __name__ == "__main__":
    asyncio.run(seed())
