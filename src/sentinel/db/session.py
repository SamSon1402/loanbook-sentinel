"""Async SQLAlchemy engine + session factory.

The engine is created once at process start. `get_session()` is a
FastAPI-friendly async generator that yields a session and rolls back on
exceptions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sentinel.config import get_settings
from sentinel.db.base import Base

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    """Create all tables. Idempotent. Real deploys use Alembic instead."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Rolls back on exception, always closes."""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
