# ---------------------------------------------------------------------------
# Enrichment Service Container
# ---------------------------------------------------------------------------
# Python 3.11 slim base, non-root user, exposes port 8081.
# Multi-stage build keeps the final image small.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by numpy / pyarrow wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Build stage -- install Python deps into a virtual-env
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Final stage
# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy virtual-env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/false --create-home appuser

WORKDIR /app

# Copy application code
COPY features/enrichment_service/ ./features/enrichment_service/
COPY features/feast/ ./features/feast/

# Environment defaults (overridable via Helm / ConfigMap)
ENV FEAST_REPO_PATH=/app/features/feast \
    SERVICE_PORT=8081 \
    LOG_LEVEL=INFO

EXPOSE 8081

USER appuser

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health')"

ENTRYPOINT ["uvicorn", "features.enrichment_service.app:app", \
            "--host", "0.0.0.0", "--port", "8081", \
            "--workers", "2", "--log-level", "info"]
