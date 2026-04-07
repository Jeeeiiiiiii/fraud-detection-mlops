# ---------------------------------------------------------------------------
# Training Container
# ---------------------------------------------------------------------------
# Contains the full training pipeline including Optuna hyperparameter
# search, XGBoost model training, and MLflow logging.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for XGBoost, matplotlib, and numeric workloads
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user
RUN groupadd --gid 1001 trainer && \
    useradd --uid 1001 --gid trainer --shell /bin/false --create-home trainer

WORKDIR /app

COPY training/ ./training/
COPY data/ ./data/

# Default environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    DATA_PATH=/app/data/output/transactions.parquet \
    N_TRIALS=50 \
    EXPERIMENT_NAME=fraud-detection

USER trainer

ENTRYPOINT ["python", "-m", "training.train"]
CMD ["--data-path", "/app/data/output/transactions.parquet", \
     "--mlflow-uri", "http://mlflow:5000", \
     "--n-trials", "50"]
