"""Synthetic GCC SME portfolio generator.

Produces a realistic-shape portfolio for development and CI: ~5,000 active SME
loans across sectors and emirates, with per-loan event streams (POS inflows,
NSFs, repayments, etc.). Writes pickled DataFrames to data/synthetic/ so the
training script and seed script can consume them independently.

Reproducible via the seed passed in via SENTINEL_SEED (defaults to 42).
"""

from __future__ import annotations

import os
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ulid import ULID

SECTORS = ["Retail", "F&B", "Logistics", "Construction", "Healthcare", "Tech", "Manufacturing", "Trading"]
EMIRATES_BY_COUNTRY = {
    "AE": ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Ras Al Khaimah"],
    "SA": ["Riyadh", "Jeddah", "Dammam", "Mecca"],
    "QA": ["Doha"],
    "BH": ["Manama"],
    "OM": ["Muscat"],
}
COUNTRY_WEIGHTS = [0.55, 0.30, 0.06, 0.05, 0.04]
EVENT_TYPES_POS = ["REPAYMENT", "POS_INFLOW", "BANK_DEPOSIT"]
EVENT_TYPES_NEG = ["MISSED_PAYMENT", "NSF", "OVERDRAFT"]


def make_merchant_name(sector: str, idx: int) -> str:
    prefixes = ["Al", "Emaar", "Gulf", "Dar", "Pearl", "Salam", "Royal", "Star", "Bright", "Prime"]
    suffix_by_sector = {
        "Retail": ["Trading", "Stores", "Mart"],
        "F&B": ["Catering", "Restaurants", "Foods"],
        "Logistics": ["Logistics", "Transport", "Cargo"],
        "Construction": ["Construction", "Contracting", "Builders"],
        "Healthcare": ["Medical", "Pharmacy", "Clinic"],
        "Tech": ["Tech", "Digital", "Systems"],
        "Manufacturing": ["Industries", "Factory", "Mfg"],
        "Trading": ["Trading", "General Trading", "Commerce"],
    }
    return f"{random.choice(prefixes)} {random.choice(suffix_by_sector[sector])} {idx:04d}"


def synthesize_loan(idx: int, as_of: date) -> dict:
    country = random.choices(list(EMIRATES_BY_COUNTRY.keys()), weights=COUNTRY_WEIGHTS)[0]
    emirate = random.choice(EMIRATES_BY_COUNTRY[country])
    sector = random.choice(SECTORS)
    tenor = random.choice([6, 12, 18, 24, 36])

    months_ago = random.randint(0, min(tenor, 24))
    origination = as_of - timedelta(days=months_ago * 30)

    principal = round(np.random.lognormal(mean=11.2, sigma=0.6), 2)  # roughly 10k-300k AED
    principal = min(max(principal, 5_000), 500_000)

    pd_predicted = float(np.clip(np.random.beta(2.0, 25.0), 0.005, 0.4))

    # Realised behavior — most loans behave; some deviate
    behavior = np.random.choice(
        ["good", "bumpy", "bad"], p=[0.78, 0.17, 0.05]
    )
    if behavior == "good":
        pd_current = pd_predicted * float(np.random.uniform(0.7, 1.2))
        dpd = 0 if random.random() > 0.05 else random.randint(1, 14)
        status = "ACTIVE"
    elif behavior == "bumpy":
        pd_current = pd_predicted * float(np.random.uniform(1.5, 2.8))
        dpd = random.randint(15, 60)
        status = "IN_GRACE" if dpd >= 30 else "ACTIVE"
    else:  # bad
        pd_current = float(np.clip(pd_predicted * np.random.uniform(3.0, 6.0), 0, 1))
        dpd = random.randint(60, 120)
        status = "DEFAULTED" if dpd >= 90 else "IN_GRACE"

    outstanding = principal * (1.0 - months_ago / max(tenor, 1))
    outstanding = max(0, round(outstanding, 2))
    if status == "DEFAULTED":
        outstanding = round(principal * 0.6, 2)
    case_score = int(800 - pd_predicted * 1000 + np.random.normal(0, 30))
    case_score = max(300, min(850, case_score))

    return {
        "id": str(ULID()),
        "merchant_id": f"M{idx:06d}",
        "merchant_name": make_merchant_name(sector, idx),
        "sector": sector,
        "emirate": emirate,
        "country": country,
        "principal_aed": principal,
        "outstanding_aed": outstanding,
        "apr": round(float(np.random.uniform(0.12, 0.28)), 4),
        "tenor_months": tenor,
        "origination_date": origination.isoformat(),
        "cohort": origination.strftime("%Y-%m"),
        "status": status,
        "pd_predicted": round(pd_predicted, 4),
        "pd_current": round(pd_current, 4),
        "dpd": dpd,
        "case_score": case_score,
        "_behavior": behavior,
    }


def synthesize_events(loan: dict, as_of: datetime) -> list[dict]:
    events: list[dict] = []
    behavior = loan["_behavior"]
    months_ago = (
        (as_of.date() - date.fromisoformat(loan["origination_date"])).days // 30
    )
    weeks_active = max(1, months_ago * 4)

    base_pos_amt = float(loan["principal_aed"]) / max(loan["tenor_months"], 1) * 1.3
    base_pos_amt = max(500, base_pos_amt)
    repayment_amt = float(loan["principal_aed"]) / max(loan["tenor_months"], 1)

    if behavior == "good":
        pos_per_week = (3, 5)
        nsf_prob, overdraft_prob, missed_prob = 0.005, 0.005, 0.01
        repay_prob = 0.95
    elif behavior == "bumpy":
        pos_per_week = (2, 4)
        nsf_prob, overdraft_prob, missed_prob = 0.04, 0.03, 0.08
        repay_prob = 0.80
    else:
        pos_per_week = (0, 2)
        nsf_prob, overdraft_prob, missed_prob = 0.12, 0.10, 0.30
        repay_prob = 0.40

    for w in range(weeks_active):
        week_ts = as_of - timedelta(weeks=weeks_active - w)

        # POS inflows for the week
        n_pos = random.randint(*pos_per_week)
        for _ in range(n_pos):
            day_offset = random.randint(0, 6)
            ts = week_ts + timedelta(days=day_offset, hours=random.randint(8, 22))
            amt = round(base_pos_amt * float(np.random.uniform(0.6, 1.4)), 2)
            events.append({
                "id": str(ULID()),
                "loan_id": loan["id"],
                "event_type": "POS_INFLOW",
                "amount_aed": amt,
                "ts": ts.isoformat(),
            })

        # Repayment (monthly cadence ≈ every 4 weeks)
        if w % 4 == 3:
            if random.random() < repay_prob:
                events.append({
                    "id": str(ULID()),
                    "loan_id": loan["id"],
                    "event_type": "REPAYMENT",
                    "amount_aed": -round(repayment_amt, 2),
                    "ts": (week_ts + timedelta(days=2)).isoformat(),
                })
            else:
                events.append({
                    "id": str(ULID()),
                    "loan_id": loan["id"],
                    "event_type": "MISSED_PAYMENT",
                    "amount_aed": 0,
                    "ts": (week_ts + timedelta(days=2)).isoformat(),
                })

        if random.random() < nsf_prob:
            events.append({
                "id": str(ULID()),
                "loan_id": loan["id"],
                "event_type": "NSF",
                "amount_aed": -round(random.uniform(50, 250), 2),
                "ts": (week_ts + timedelta(days=random.randint(0, 6))).isoformat(),
            })
        if random.random() < overdraft_prob:
            events.append({
                "id": str(ULID()),
                "loan_id": loan["id"],
                "event_type": "OVERDRAFT",
                "amount_aed": -round(random.uniform(100, 500), 2),
                "ts": (week_ts + timedelta(days=random.randint(0, 6))).isoformat(),
            })
        if random.random() < missed_prob and w % 4 != 3:
            events.append({
                "id": str(ULID()),
                "loan_id": loan["id"],
                "event_type": "MISSED_PAYMENT",
                "amount_aed": 0,
                "ts": (week_ts + timedelta(days=random.randint(0, 6))).isoformat(),
            })

    return events


def main(n_loans: int = 5_000, out_dir: Path = Path("data/synthetic")) -> None:
    seed = int(os.environ.get("SENTINEL_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    as_of_date = date.today()
    as_of_dt = datetime.now(timezone.utc)

    print(f"[generate] synthesizing {n_loans} loans (seed={seed})…")
    loans: list[dict] = [synthesize_loan(i, as_of_date) for i in range(n_loans)]

    all_events: list[dict] = []
    for i, loan in enumerate(loans):
        all_events.extend(synthesize_events(loan, as_of_dt))
        if (i + 1) % 1000 == 0:
            print(f"[generate] events for {i + 1} loans…")

    for loan in loans:
        loan.pop("_behavior", None)

    loans_df = pd.DataFrame(loans)
    events_df = pd.DataFrame(all_events)

    loans_df.to_parquet(out_dir / "loans.parquet", index=False)
    events_df.to_parquet(out_dir / "events.parquet", index=False)

    print(
        f"[generate] wrote {len(loans_df):,} loans + {len(events_df):,} events → "
        f"{out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
