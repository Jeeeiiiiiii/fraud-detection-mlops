# ---------------------------------------------------------------------------
# Inference / Scoring Container
# ---------------------------------------------------------------------------
# Serves the fraud scoring API via FastAPI + Uvicorn.
# Optimised for low-latency inference (< 50 ms p99).
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/false --create-home appuser

WORKDIR /app

COPY serving/ ./serving/

# Configuration defaults
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=fraud-detection-xgb \
    MODEL_VERSION=latest \
    REVIEW_THRESHOLD=0.7 \
    BLOCK_THRESHOLD=0.9 \
    SERVICE_PORT=8080 \
    LOG_LEVEL=INFO

EXPOSE 8080

USER appuser

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Use 1 worker with thread pool for CPU-bound XGBoost inference.
# Scale horizontally via HPA rather than vertically via workers.
ENTRYPOINT ["uvicorn", "serving.predict:app", \
            "--host", "0.0.0.0", "--port", "8080", \
            "--workers", "1", "--log-level", "info"]
