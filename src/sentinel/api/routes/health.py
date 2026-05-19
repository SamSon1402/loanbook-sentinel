"""Liveness + readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from sentinel.api.deps import DetectorDep, SessionDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Liveness — process is up. Used by K8s livenessProbe."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(
    response: Response,
    session: SessionDep,
    detector: DetectorDep,
) -> dict[str, object]:
    """Readiness — DB reachable and model loaded. Used by K8s readinessProbe."""
    checks: dict[str, str] = {}
    is_ready = True

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # pragma: no cover - exercised in integration
        checks["database"] = f"error: {exc.__class__.__name__}"
        is_ready = False

    checks["model"] = "ok" if detector is not None else "not_loaded"
    if detector is None:
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"ready": is_ready, "checks": checks}
