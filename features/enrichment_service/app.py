#!/usr/bin/env python3
"""
Feature Enrichment Service
==========================
FastAPI microservice that enriches raw transaction JSON with features from
the Feast online store and computes real-time derived features for downstream
fraud scoring.

Endpoints
---------
- POST /enrich          Enrich a single transaction with features.
- POST /enrich/batch    Enrich a batch of transactions.
- GET  /health          Liveness / readiness probe.
- GET  /metrics         Prometheus metrics (text format).

Real-Time Features Computed
---------------------------
- velocity_10min / velocity_1hr   Transaction counts in rolling windows.
- time_since_last_txn             Seconds since this user's previous transaction.
- amount_vs_avg_ratio             Current amount / user's 30-day average amount.
- billing_shipping_match          1 if billing country == shipping country.

Configuration is read from environment variables so the service can be
deployed via Helm / Kubernetes without code changes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit structured JSON log lines for easy ingestion by Loki / ELK."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

logger = logging.getLogger("enrichment_service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/features/feast")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8081"))

# ---------------------------------------------------------------------------
# Prometheus metrics (custom registry to avoid default process/gc collectors
# clashing with other services in the same process during tests)
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

ENRICHMENT_LATENCY = Histogram(
    "enrichment_latency_seconds",
    "Time spent enriching a transaction (seconds).",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY,
)

ENRICHMENT_REQUESTS = Counter(
    "enrichment_requests_total",
    "Total enrichment requests received.",
    labelnames=["endpoint", "status"],
    registry=REGISTRY,
)

ENRICHMENT_ERRORS = Counter(
    "enrichment_errors_total",
    "Total enrichment errors.",
    labelnames=["error_type"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Feast store (lazy-loaded at startup)
# ---------------------------------------------------------------------------

_feast_store = None


def _get_feast_store():
    """Return a cached Feast FeatureStore instance."""
    global _feast_store
    if _feast_store is None:
        try:
            from feast import FeatureStore
            _feast_store = FeatureStore(repo_path=FEAST_REPO_PATH)
            logger.info("Feast FeatureStore initialised from %s", FEAST_REPO_PATH)
        except Exception:
            logger.warning(
                "Feast unavailable -- falling back to stub features. "
                "Set FEAST_REPO_PATH to a valid repo for production use."
            )
    return _feast_store


# ---------------------------------------------------------------------------
# In-memory velocity tracker (production would use Redis sorted-sets or
# a streaming engine like Flink; this serves as a functional placeholder)
# ---------------------------------------------------------------------------

_user_timestamps: dict[str, list[float]] = {}


def _record_velocity(user_id: str, ts: float) -> dict[str, Any]:
    """
    Record a transaction timestamp for the user and return velocity metrics.

    Returns dict with:
      - velocity_10min: number of transactions in the last 10 minutes
      - velocity_1hr:   number of transactions in the last hour
      - time_since_last_txn: seconds since the user's previous transaction
    """
    now = ts
    history = _user_timestamps.setdefault(user_id, [])

    # Compute time since last transaction *before* appending the new one
    time_since_last = float(now - history[-1]) if history else -1.0

    history.append(now)

    # Prune entries older than 1 hour to cap memory growth
    cutoff_1h = now - 3600
    history[:] = [t for t in history if t >= cutoff_1h]

    cutoff_10m = now - 600
    velocity_10min = sum(1 for t in history if t >= cutoff_10m)
    velocity_1hr = len(history)

    return {
        "velocity_10min": velocity_10min,
        "velocity_1hr": velocity_1hr,
        "time_since_last_txn": round(time_since_last, 2),
    }


# ---------------------------------------------------------------------------
# Feature lookup helpers
# ---------------------------------------------------------------------------

def _fetch_feast_features(user_id: str, merchant_id: str) -> dict[str, Any]:
    """
    Retrieve pre-computed features from the Feast online store.

    If Feast is unavailable (e.g. local dev), return sensible defaults so
    the service stays operational for integration testing.
    """
    store = _get_feast_store()
    defaults = {
        "avg_amount_30d": 150.0,
        "max_amount_30d": 500.0,
        "transaction_count_24h": 3,
        "unique_merchants_7d": 5,
        "unique_countries_7d": 1,
        "merchant_fraud_rate_30d": 0.01,
        "merchant_avg_amount": 120.0,
        "merchant_category_risk_score": 0.1,
    }
    if store is None:
        return defaults

    try:
        feature_refs = [
            "user_transaction_features:avg_amount_30d",
            "user_transaction_features:max_amount_30d",
            "user_transaction_features:transaction_count_24h",
            "user_transaction_features:unique_merchants_7d",
            "user_transaction_features:unique_countries_7d",
            "merchant_features:merchant_fraud_rate_30d",
            "merchant_features:merchant_avg_amount",
            "merchant_features:merchant_category_risk_score",
        ]
        entity_rows = [{"user_id": user_id, "merchant_id": merchant_id}]
        response = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        result = response.to_dict()
        # Flatten: each key maps to a list with one element
        return {k: (v[0] if v[0] is not None else defaults.get(k, 0)) for k, v in result.items()}
    except Exception as exc:
        logger.warning("Feast lookup failed, using defaults: %s", exc)
        ENRICHMENT_ERRORS.labels(error_type="feast_lookup").inc()
        return defaults


def _compute_derived_features(
    txn: dict[str, Any],
    feast_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute derived features that combine raw transaction fields with
    historical aggregates.
    """
    amount = txn.get("amount", 0.0)
    avg_amount = feast_features.get("avg_amount_30d", 1.0) or 1.0

    billing = txn.get("billing_country", "")
    shipping = txn.get("shipping_country", "")

    return {
        "amount_vs_avg_ratio": round(amount / avg_amount, 4),
        "billing_shipping_match": int(billing == shipping),
        "is_high_risk_category": int(
            txn.get("merchant_category", "") in {
                "electronics", "gift_cards", "jewelry", "digital_goods",
                "crypto_exchange", "money_transfer", "gambling",
            }
        ),
        "is_odd_hour": int(
            0 <= txn.get("transaction_hour", 12) < 5
        ),
    }


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TransactionIn(BaseModel):
    """Raw transaction payload received from the gateway."""
    transaction_id: str = Field(..., description="Unique transaction identifier.")
    user_id: str = Field(..., description="Cardholder / account owner ID.")
    amount: float = Field(..., gt=0, description="Transaction amount.")
    currency: str = Field(default="USD", description="ISO currency code.")
    merchant_id: str = Field(..., description="Merchant identifier.")
    merchant_category: str = Field(default="unknown", description="Merchant category.")
    card_type: str = Field(default="unknown")
    device_type: str = Field(default="unknown")
    ip_country: str = Field(default="US")
    billing_country: str = Field(default="US")
    shipping_country: str = Field(default="US")
    timestamp: str | None = Field(
        default=None,
        description="ISO-8601 timestamp; defaults to server time if omitted.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_000000000001",
                "user_id": "user_42",
                "amount": 249.99,
                "currency": "USD",
                "merchant_id": "merch_1234",
                "merchant_category": "electronics",
                "card_type": "visa",
                "device_type": "mobile",
                "ip_country": "US",
                "billing_country": "US",
                "shipping_country": "US",
                "timestamp": "2025-06-15T14:32:00Z",
            }
        }


class EnrichedTransaction(BaseModel):
    """Enriched transaction ready for the scoring service."""
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    card_type: str
    device_type: str
    ip_country: str
    billing_country: str
    shipping_country: str
    timestamp: str

    # Velocity features
    velocity_10min: int = 0
    velocity_1hr: int = 0
    time_since_last_txn: float = -1.0

    # Historical aggregates from Feast
    avg_amount_30d: float = 0.0
    max_amount_30d: float = 0.0
    transaction_count_24h: int = 0
    unique_merchants_7d: int = 0
    unique_countries_7d: int = 0
    merchant_fraud_rate_30d: float = 0.0
    merchant_avg_amount: float = 0.0
    merchant_category_risk_score: float = 0.0

    # Derived features
    amount_vs_avg_ratio: float = 0.0
    billing_shipping_match: int = 1
    is_high_risk_category: int = 0
    is_odd_hour: int = 0


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up Feast store on startup."""
    logger.info("Enrichment service starting up -- warming Feast store")
    _get_feast_store()
    yield
    logger.info("Enrichment service shutting down")


app = FastAPI(
    title="Fraud Detection Feature Enrichment Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/enrich", response_model=EnrichedTransaction)
async def enrich_transaction(txn: TransactionIn) -> EnrichedTransaction:
    """
    Enrich a single raw transaction with historical and real-time features.

    The response contains all original fields plus computed features ready
    for the downstream scoring service.
    """
    start = time.monotonic()
    try:
        txn_dict = txn.model_dump()

        # Parse or assign timestamp
        if txn_dict.get("timestamp"):
            try:
                ts_dt = datetime.fromisoformat(txn_dict["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts_dt = datetime.now(timezone.utc)
        else:
            ts_dt = datetime.now(timezone.utc)
        txn_dict["timestamp"] = ts_dt.isoformat()
        txn_dict["transaction_hour"] = ts_dt.hour
        ts_epoch = ts_dt.timestamp()

        # 1. Velocity features (real-time, in-memory)
        velocity = _record_velocity(txn.user_id, ts_epoch)

        # 2. Feast features (online store lookup)
        feast_feats = _fetch_feast_features(txn.user_id, txn.merchant_id)

        # 3. Derived features
        derived = _compute_derived_features(txn_dict, feast_feats)

        enriched = EnrichedTransaction(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            amount=txn.amount,
            currency=txn.currency,
            merchant_id=txn.merchant_id,
            merchant_category=txn.merchant_category,
            card_type=txn.card_type,
            device_type=txn.device_type,
            ip_country=txn.ip_country,
            billing_country=txn.billing_country,
            shipping_country=txn.shipping_country,
            timestamp=txn_dict["timestamp"],
            # Velocity
            velocity_10min=velocity["velocity_10min"],
            velocity_1hr=velocity["velocity_1hr"],
            time_since_last_txn=velocity["time_since_last_txn"],
            # Historical
            avg_amount_30d=feast_feats.get("avg_amount_30d", 0.0),
            max_amount_30d=feast_feats.get("max_amount_30d", 0.0),
            transaction_count_24h=feast_feats.get("transaction_count_24h", 0),
            unique_merchants_7d=feast_feats.get("unique_merchants_7d", 0),
            unique_countries_7d=feast_feats.get("unique_countries_7d", 0),
            merchant_fraud_rate_30d=feast_feats.get("merchant_fraud_rate_30d", 0.0),
            merchant_avg_amount=feast_feats.get("merchant_avg_amount", 0.0),
            merchant_category_risk_score=feast_feats.get("merchant_category_risk_score", 0.0),
            # Derived
            amount_vs_avg_ratio=derived["amount_vs_avg_ratio"],
            billing_shipping_match=derived["billing_shipping_match"],
            is_high_risk_category=derived["is_high_risk_category"],
            is_odd_hour=derived["is_odd_hour"],
        )

        ENRICHMENT_REQUESTS.labels(endpoint="/enrich", status="success").inc()
        return enriched

    except Exception as exc:
        ENRICHMENT_ERRORS.labels(error_type="enrichment_failure").inc()
        ENRICHMENT_REQUESTS.labels(endpoint="/enrich", status="error").inc()
        logger.exception("Enrichment failed for txn=%s", txn.transaction_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        elapsed = time.monotonic() - start
        ENRICHMENT_LATENCY.observe(elapsed)


@app.post("/enrich/batch", response_model=list[EnrichedTransaction])
async def enrich_batch(transactions: list[TransactionIn]) -> list[EnrichedTransaction]:
    """Enrich a batch of transactions. Max 100 per request."""
    if len(transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size must not exceed 100 transactions.",
        )
    results = []
    for txn in transactions:
        enriched = await enrich_transaction(txn)
        results.append(enriched)
    return results


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness and readiness probe."""
    return {"status": "healthy", "service": "enrichment", "version": "1.0.0"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint in OpenMetrics text format."""
    data = generate_latest(REGISTRY)
    return PlainTextResponse(content=data.decode("utf-8"), media_type="text/plain")


# ---------------------------------------------------------------------------
# Entrypoint (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
