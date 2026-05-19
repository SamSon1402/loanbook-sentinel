"""ORM models — Loan, LoanEvent, Anomaly.

Uses SQLAlchemy 2.0 typed `Mapped[...]` syntax. Indexes match the read paths
in `services/portfolio_service.py` and `services/scanner.py`.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.db.base import Base


# ── enums ───────────────────────────────────────────────────────────────────
class LoanStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    IN_GRACE = "IN_GRACE"
    DEFAULTED = "DEFAULTED"
    REPAID = "REPAID"
    WRITTEN_OFF = "WRITTEN_OFF"


class EventType(str, enum.Enum):
    REPAYMENT = "REPAYMENT"
    MISSED_PAYMENT = "MISSED_PAYMENT"
    POS_INFLOW = "POS_INFLOW"
    BANK_DEPOSIT = "BANK_DEPOSIT"
    NSF = "NSF"
    OVERDRAFT = "OVERDRAFT"
    DISBURSEMENT = "DISBURSEMENT"


class AnomalySeverity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AnomalyType(str, enum.Enum):
    DPD_BREACH = "DPD_BREACH"
    BEHAVIORAL = "BEHAVIORAL"
    SECTOR_COMOVEMENT = "SECTOR_COMOVEMENT"
    CASHFLOW_DROP = "CASHFLOW_DROP"
    NSF_BURST = "NSF_BURST"


# ── tables ──────────────────────────────────────────────────────────────────
class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(32), index=True)
    merchant_name: Mapped[str] = mapped_column(String(128))
    sector: Mapped[str] = mapped_column(String(32), index=True)
    emirate: Mapped[str] = mapped_column(String(32), index=True)
    country: Mapped[str] = mapped_column(String(2))

    principal_aed: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    outstanding_aed: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    apr: Mapped[float] = mapped_column()
    tenor_months: Mapped[int] = mapped_column(Integer)
    origination_date: Mapped[date] = mapped_column(Date)
    cohort: Mapped[str] = mapped_column(String(7), index=True)  # YYYY-MM

    status: Mapped[LoanStatus] = mapped_column(
        Enum(LoanStatus, native_enum=False), default=LoanStatus.ACTIVE, index=True
    )
    pd_predicted: Mapped[float] = mapped_column(default=0.0)
    pd_current: Mapped[float] = mapped_column(default=0.0)
    dpd: Mapped[int] = mapped_column(Integer, default=0)
    case_score: Mapped[int] = mapped_column(Integer, default=0)

    events: Mapped[list[LoanEvent]] = relationship(
        back_populates="loan", cascade="all, delete-orphan", lazy="raise"
    )
    anomalies: Mapped[list[Anomaly]] = relationship(
        back_populates="loan", cascade="all, delete-orphan", lazy="raise"
    )

    __table_args__ = (
        Index("ix_loans_status_cohort", "status", "cohort"),
        Index("ix_loans_sector_emirate", "sector", "emirate"),
    )


class LoanEvent(Base):
    __tablename__ = "loan_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    loan_id: Mapped[str] = mapped_column(
        ForeignKey("loans.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType, native_enum=False), index=True
    )
    amount_aed: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    loan: Mapped[Loan] = relationship(back_populates="events", lazy="raise")

    __table_args__ = (Index("ix_loan_events_loan_ts", "loan_id", "ts"),)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    loan_id: Mapped[str] = mapped_column(
        ForeignKey("loans.id", ondelete="CASCADE"), index=True
    )
    severity: Mapped[AnomalySeverity] = mapped_column(
        Enum(AnomalySeverity, native_enum=False), index=True
    )
    type: Mapped[AnomalyType] = mapped_column(
        Enum(AnomalyType, native_enum=False), index=True
    )
    score: Mapped[float] = mapped_column()
    message: Mapped[str] = mapped_column(String(512))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    loan: Mapped[Loan] = relationship(back_populates="anomalies", lazy="raise")

    __table_args__ = (
        Index("ix_anomalies_open", "resolved_at", "severity"),
    )
