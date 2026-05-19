# syntax=docker/dockerfile:1.7
# ─── builder ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip wheel \
 && pip wheel --wheel-dir /build/wheels .

# ─── runtime ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel -d /app sentinel

WORKDIR /app
COPY --from=builder /build/wheels /wheels
RUN pip install --no-index --find-links /wheels loanbook-sentinel \
 && rm -rf /wheels

# models directory mounted at runtime (or copied via initContainer in K8s)
RUN mkdir -p /app/models && chown -R sentinel:sentinel /app

USER sentinel
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "sentinel.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
