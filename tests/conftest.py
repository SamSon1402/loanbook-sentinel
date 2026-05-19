"""Pytest fixtures.

We train a *tiny* in-memory Isolation Forest and export it to a temp ONNX file
once per test session — so tests don't depend on `make bootstrap` having run
beforehand. This is also how CI gets a model artifact for the API tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest
from sqlalchemy.ext.asyncio import async_sessionmaker

from sentinel.db.base import Base
from sentinel.ml.features import FEATURE_ORDER


@pytest.fixture(scope="session")
def tiny_onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a fast IsoForest on Gaussian noise and export to ONNX."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, len(FEATURE_ORDER))).astype(np.float32)
    clf = IsolationForest(n_estimators=30, contamination=0.1, random_state=0)
    clf.fit(X)
    onnx = to_onnx(
        clf,
        initial_types=[("X", FloatTensorType([None, len(FEATURE_ORDER)]))],
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(clf): {"score_samples": True}},
    )
    p = tmp_path_factory.mktemp("models") / "anomaly.onnx"
    p.write_bytes(onnx.SerializeToString())
    return p


@pytest.fixture(autouse=True)
def _env(tiny_onnx_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point Settings at a temp DB + the tiny ONNX model."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("SENTINEL_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("SENTINEL_MODEL_PATH", str(tiny_onnx_model))
    monkeypatch.setenv("SENTINEL_SCAN_INTERVAL_SECONDS", "9999")
    monkeypatch.setenv("SENTINEL_LOG_LEVEL", "WARNING")

    # Invalidate the cached settings + engine modules so they re-read env
    import importlib

    from sentinel import config as _config
    from sentinel.db import session as _session

    _config.get_settings.cache_clear()
    importlib.reload(_session)
    yield
    _config.get_settings.cache_clear()


@pytest_asyncio.fixture
async def session() -> AsyncIterator:
    """A fresh async DB session against a per-test SQLite file."""
    from sentinel.db.session import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """An httpx ASGI client running the full FastAPI lifespan."""
    # importing here lets the env fixture run first
    from sentinel.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c
