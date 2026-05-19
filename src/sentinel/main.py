"""FastAPI application factory + lifespan.

`create_app()` wires routes, dependencies, middleware, and Prometheus. The
`lifespan` context manager loads the ONNX model once at process startup,
constructs the singleton `AnomalyDetector` + `Scanner`, and spawns a
background task that re-scans the active book on `SENTINEL_SCAN_INTERVAL_SECONDS`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest

from sentinel.api.routes import anomalies, health, loans, portfolio
from sentinel.config import get_settings
from sentinel.core.logging import configure_logging, get_logger
from sentinel.core.metrics import registry
from sentinel.db.session import SessionLocal, init_db
from sentinel.ml.anomaly_detector import AnomalyDetector
from sentinel.services.scanner import Scanner

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log.info("app.starting", env=settings.env)

    # init DB schema (replace with Alembic in real deploys)
    await init_db()

    # Load ONNX model
    detector = AnomalyDetector(
        model_path=settings.model_path,
        threshold=settings.anomaly_threshold,
    )
    app.state.detector = detector
    app.state.scanner = Scanner(detector, settings)

    # Periodic background scanner
    stop_event = asyncio.Event()
    app.state.stop_event = stop_event

    async def _scanner_loop() -> None:
        while not stop_event.is_set():
            try:
                async with SessionLocal() as session:
                    await app.state.scanner.run_once(session)
            except Exception:
                log.exception("scanner.loop_error")
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=settings.scan_interval_seconds,
                )
            except TimeoutError:
                pass  # normal — wake up to run the next scan

    scanner_task = asyncio.create_task(_scanner_loop(), name="scanner_loop")
    app.state.scanner_task = scanner_task
    log.info(
        "app.started",
        scan_interval=settings.scan_interval_seconds,
        model=str(settings.model_path),
    )

    try:
        yield
    finally:
        log.info("app.stopping")
        stop_event.set()
        scanner_task.cancel()
        try:
            await scanner_task
        except (asyncio.CancelledError, Exception):
            pass
        log.info("app.stopped")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title="LoanBook-Sentinel",
        description="Real-time loan portfolio health & anomaly detection.",
        version="0.3.1",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(health.router)
    app.include_router(loans.router)
    app.include_router(portfolio.router)
    app.include_router(anomalies.router)

    # Prometheus exposition endpoint (single shared registry, see core.metrics)
    if settings.metrics_enabled:
        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(
                content=generate_latest(registry),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    return app


app = create_app()


def cli() -> None:
    """`sentinel` console-script entrypoint."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sentinel.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        reload=not settings.is_production,
    )
