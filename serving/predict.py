#!/usr/bin/env python3
"""
Fraud Scoring Service
=====================
FastAPI microservice that scores enriched transactions for fraud probability.

Endpoints
---------
- POST /score           Score a single enriched transaction.
- POST /score/batch     Score up to 200 transactions in a single request.
- GET  /health          Liveness / readiness probe.
- GET  /metrics         Prometheus metrics (text format).

The service loads the XGBoost model from MLflow on startup and caches it
in memory.  Warm-up predictions are executed during startup to ensure
the first real request meets the < 50 ms SLA.

Decision logic:
  - fraud_score >= BLOCK_THRESHOLD  -> action = "BLOCK"
  - fraud_score >= REVIEW_THRESHOLD -> action = "REVIEW"
  - otherwise                       -> action = "ALLOW"

All predictions are logged as structured JSON for the audit trail.
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
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
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
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

logger = logging.getLogger("scoring_service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-xgb")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.7"))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.9"))
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8080"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "200"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

SCORING_LATENCY = Histogram(
    "scoring_latency_seconds",
    "Latency of fraud score computation (seconds).",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

FRAUD_SCORE_DIST = Histogram(
    "fraud_score",
    "Distribution of fraud scores returned by the model.",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

BLOCK_RATE = Counter(
    "block_rate",
    "Number of transactions blocked.",
    registry=REGISTRY,
)

REVIEW_RATE = Counter(
    "review_rate",
    "Number of transactions sent for review.",
    registry=REGISTRY,
)

ALLOW_RATE = Counter(
    "allow_rate",
    "Number of transactions allowed.",
    registry=REGISTRY,
)

SCORING_REQUESTS = Counter(
    "scoring_requests_total",
    "Total scoring requests.",
    labelnames=["endpoint", "status"],
    registry=REGISTRY,
)

SCORING_ERRORS = Counter(
    "scoring_errors_total",
    "Total scoring errors.",
    labelnames=["error_type"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Model loading & caching
# ---------------------------------------------------------------------------

_cached_model = None
_cached_feature_names: list[str] | None = None
_cached_threshold: float = 0.5


def _load_model():
    """
    Load the XGBoost model from the MLflow Model Registry.

    Falls back to a stub predictor if MLflow is unreachable (useful for
    local development and integration tests).
    """
    global _cached_model, _cached_feature_names, _cached_threshold
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if MODEL_VERSION == "latest":
            model_uri = f"models:/{MODEL_NAME}/latest"
        else:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        logger.info("Loading model from %s", model_uri)
        _cached_model = mlflow.xgboost.load_model(model_uri)

        # Try to load model metadata for feature names and threshold
        try:
            client = mlflow.MlflowClient()
            if MODEL_VERSION == "latest":
                versions = client.get_latest_versions(MODEL_NAME)
                if versions:
                    run_id = versions[0].run_id
                else:
                    run_id = None
            else:
                mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)
                run_id = mv.run_id

            if run_id:
                run = client.get_run(run_id)
                _cached_threshold = float(
                    run.data.params.get("optimal_threshold", 0.5)
                )
        except Exception as meta_exc:
            logger.warning("Could not load model metadata: %s", meta_exc)

        logger.info("Model loaded successfully. Threshold: %.2f", _cached_threshold)

    except Exception as exc:
        logger.warning(
            "MLflow unavailable -- using stub model. Error: %s", exc
        )
        _cached_model = None


def _warm_up_model():
    """
    Run a dummy prediction to force XGBoost to JIT-compile tree structures.
    This ensures the first real request meets the < 50 ms SLA.
    """
    if _cached_model is None:
        return
    try:
        n_features = _cached_model.n_features_in_
        dummy = np.zeros((1, n_features), dtype=np.float32)
        _cached_model.predict_proba(dummy)
        logger.info("Model warm-up complete (%d features)", n_features)
    except Exception as exc:
        logger.warning("Warm-up prediction failed: %s", exc)


def _predict(features: np.ndarray) -> np.ndarray:
    """
    Run model prediction. Returns fraud probabilities.

    If no model is loaded, returns a random baseline (for dev/test only).
    """
    if _cached_model is not None:
        return _cached_model.predict_proba(features)[:, 1]
    else:
        # Stub: return low scores to simulate a healthy model
        return np.random.uniform(0.0, 0.3, size=features.shape[0])


# ---------------------------------------------------------------------------
# Feature vector extraction
# ---------------------------------------------------------------------------

# Feature ordering must match training. These are the enriched features
# produced by the enrichment service plus label-encoded categoricals.
FEATURE_KEYS = [
    "merchant_category", "card_type", "device_type",
    "ip_country", "billing_country", "shipping_country",
    "amount", "transaction_hour", "transaction_day_of_week",
    "billing_shipping_match", "ip_billing_match", "ip_shipping_match",
    "amount_log",
]

# Simple hash-based encoding for categoricals at serving time
# (production would use the same LabelEncoder artefact from training)
def _encode_categorical(value: str) -> int:
    return hash(value) % 10000


def _extract_features(txn: dict[str, Any]) -> np.ndarray:
    """Convert an enriched transaction dict to a numeric feature vector."""
    categoricals = ["merchant_category", "card_type", "device_type",
                    "ip_country", "billing_country", "shipping_country"]

    features = []
    for key in FEATURE_KEYS:
        if key in categoricals:
            features.append(float(_encode_categorical(str(txn.get(key, "unknown")))))
        elif key == "amount_log":
            features.append(float(np.log1p(txn.get("amount", 0.0))))
        elif key == "transaction_hour":
            ts_str = txn.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                features.append(float(dt.hour))
            except (ValueError, AttributeError):
                features.append(12.0)
        elif key == "transaction_day_of_week":
            ts_str = txn.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                features.append(float(dt.weekday()))
            except (ValueError, AttributeError):
                features.append(3.0)
        elif key == "ip_billing_match":
            features.append(float(txn.get("ip_country", "") == txn.get("billing_country", "")))
        elif key == "ip_shipping_match":
            features.append(float(txn.get("ip_country", "") == txn.get("shipping_country", "")))
        else:
            features.append(float(txn.get(key, 0.0)))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ScoringRequest(BaseModel):
    """Enriched transaction payload from the enrichment service."""
    transaction_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    merchant_id: str = ""
    merchant_category: str = "unknown"
    card_type: str = "unknown"
    device_type: str = "unknown"
    ip_country: str = "US"
    billing_country: str = "US"
    shipping_country: str = "US"
    timestamp: str = ""

    # Enriched features (optional -- the model can work without them)
    velocity_10min: int = 0
    velocity_1hr: int = 0
    time_since_last_txn: float = -1.0
    avg_amount_30d: float = 0.0
    max_amount_30d: float = 0.0
    transaction_count_24h: int = 0
    unique_merchants_7d: int = 0
    unique_countries_7d: int = 0
    merchant_fraud_rate_30d: float = 0.0
    merchant_avg_amount: float = 0.0
    merchant_category_risk_score: float = 0.0
    amount_vs_avg_ratio: float = 0.0
    billing_shipping_match: int = 1
    is_high_risk_category: int = 0
    is_odd_hour: int = 0


class ScoringResponse(BaseModel):
    """Fraud scoring result."""
    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0,1].")
    action: str = Field(..., description="BLOCK, REVIEW, or ALLOW.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence.")
    threshold_used: float = Field(..., description="Decision threshold applied.")
    model_version: str = Field(..., description="Model version used for scoring.")
    scored_at: str = Field(..., description="ISO-8601 timestamp of scoring.")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm up model before accepting requests."""
    logger.info("Scoring service starting -- loading model")
    _load_model()
    _warm_up_model()
    logger.info(
        "Ready. Thresholds: REVIEW >= %.2f, BLOCK >= %.2f",
        REVIEW_THRESHOLD, BLOCK_THRESHOLD,
    )
    yield
    logger.info("Scoring service shutting down")


app = FastAPI(
    title="Fraud Detection Scoring Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _score_single(txn_dict: dict[str, Any]) -> ScoringResponse:
    """Score a single transaction and return the response."""
    features = _extract_features(txn_dict).reshape(1, -1)
    fraud_score = float(_predict(features)[0])

    # Determine action
    if fraud_score >= BLOCK_THRESHOLD:
        action = "BLOCK"
        BLOCK_RATE.inc()
    elif fraud_score >= REVIEW_THRESHOLD:
        action = "REVIEW"
        REVIEW_RATE.inc()
    else:
        action = "ALLOW"
        ALLOW_RATE.inc()

    # Confidence: how far the score is from the nearest threshold boundary
    if action == "BLOCK":
        confidence = min(1.0, (fraud_score - BLOCK_THRESHOLD) / (1.0 - BLOCK_THRESHOLD + 1e-9) * 0.5 + 0.5)
    elif action == "REVIEW":
        confidence = 0.5 + (fraud_score - REVIEW_THRESHOLD) / (BLOCK_THRESHOLD - REVIEW_THRESHOLD + 1e-9) * 0.3
    else:
        confidence = 1.0 - fraud_score

    confidence = round(max(0.0, min(1.0, confidence)), 4)

    FRAUD_SCORE_DIST.observe(fraud_score)

    response = ScoringResponse(
        transaction_id=txn_dict.get("transaction_id", "unknown"),
        fraud_score=round(fraud_score, 6),
        action=action,
        confidence=confidence,
        threshold_used=BLOCK_THRESHOLD if action == "BLOCK" else REVIEW_THRESHOLD,
        model_version=MODEL_VERSION,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )

    # Audit trail: log every prediction as structured JSON
    logger.info(
        json.dumps({
            "event": "prediction",
            "transaction_id": response.transaction_id,
            "user_id": txn_dict.get("user_id", ""),
            "amount": txn_dict.get("amount", 0),
            "fraud_score": response.fraud_score,
            "action": response.action,
            "confidence": response.confidence,
            "model_version": response.model_version,
        })
    )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoringResponse)
async def score_transaction(request: ScoringRequest) -> ScoringResponse:
    """Score a single enriched transaction for fraud."""
    start = time.monotonic()
    try:
        result = _score_single(request.model_dump())
        SCORING_REQUESTS.labels(endpoint="/score", status="success").inc()
        return result
    except Exception as exc:
        SCORING_ERRORS.labels(error_type="scoring_failure").inc()
        SCORING_REQUESTS.labels(endpoint="/score", status="error").inc()
        logger.exception("Scoring failed for txn=%s", request.transaction_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        SCORING_LATENCY.observe(time.monotonic() - start)


@app.post("/score/batch", response_model=list[ScoringResponse])
async def score_batch(requests: list[ScoringRequest]) -> list[ScoringResponse]:
    """
    Score a batch of enriched transactions.

    For optimal throughput, the model runs vectorised inference on the
    entire batch rather than scoring one-by-one.
    """
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(requests)} exceeds maximum {MAX_BATCH_SIZE}.",
        )

    start = time.monotonic()
    try:
        # Build feature matrix for vectorised inference
        txn_dicts = [r.model_dump() for r in requests]
        feature_matrix = np.vstack([
            _extract_features(t) for t in txn_dicts
        ])

        fraud_scores = _predict(feature_matrix)

        results = []
        for txn_dict, score in zip(txn_dicts, fraud_scores):
            score_f = float(score)

            if score_f >= BLOCK_THRESHOLD:
                action = "BLOCK"
                BLOCK_RATE.inc()
            elif score_f >= REVIEW_THRESHOLD:
                action = "REVIEW"
                REVIEW_RATE.inc()
            else:
                action = "ALLOW"
                ALLOW_RATE.inc()

            if action == "BLOCK":
                confidence = min(1.0, (score_f - BLOCK_THRESHOLD) / (1.0 - BLOCK_THRESHOLD + 1e-9) * 0.5 + 0.5)
            elif action == "REVIEW":
                confidence = 0.5 + (score_f - REVIEW_THRESHOLD) / (BLOCK_THRESHOLD - REVIEW_THRESHOLD + 1e-9) * 0.3
            else:
                confidence = 1.0 - score_f
            confidence = round(max(0.0, min(1.0, confidence)), 4)

            FRAUD_SCORE_DIST.observe(score_f)

            results.append(ScoringResponse(
                transaction_id=txn_dict.get("transaction_id", "unknown"),
                fraud_score=round(score_f, 6),
                action=action,
                confidence=confidence,
                threshold_used=BLOCK_THRESHOLD if action == "BLOCK" else REVIEW_THRESHOLD,
                model_version=MODEL_VERSION,
                scored_at=datetime.now(timezone.utc).isoformat(),
            ))

            # Audit trail
            logger.info(json.dumps({
                "event": "prediction",
                "transaction_id": txn_dict.get("transaction_id", ""),
                "fraud_score": round(score_f, 6),
                "action": action,
            }))

        SCORING_REQUESTS.labels(endpoint="/score/batch", status="success").inc()
        return results

    except Exception as exc:
        SCORING_ERRORS.labels(error_type="batch_scoring_failure").inc()
        SCORING_REQUESTS.labels(endpoint="/score/batch", status="error").inc()
        logger.exception("Batch scoring failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        SCORING_LATENCY.observe(time.monotonic() - start)


# ---------------------------------------------------------------------------
# Compatibility endpoint for payment-service integration
# ---------------------------------------------------------------------------
# The payment-service calls POST /predict with a simplified payload.
# This wraps the internal scoring logic and returns the format it expects.
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Simplified request from payment-service."""
    amount: float
    merchant_category: str = "unknown"
    customer_id: str = ""
    ip_address: str = "unknown"
    billing_country: str = "US"
    shipping_country: str = "US"
    device_id: str = "unknown"


@app.post("/predict")
async def predict_for_payment_service(request: PredictRequest) -> dict[str, Any]:
    """
    Simplified scoring endpoint for the payment-service.

    Accepts the payment-service payload format and returns:
      {"decision": "ALLOW|REVIEW|BLOCK", "score": 0.xx, "risk_factors": [...], "model_version": "..."}
    """
    start = time.monotonic()
    try:
        txn_dict = {
            "transaction_id": f"pay_{int(time.time() * 1000)}",
            "user_id": request.customer_id,
            "amount": request.amount,
            "merchant_category": request.merchant_category,
            "card_type": "unknown",
            "device_type": "desktop" if request.device_id == "unknown" else "mobile",
            "ip_country": request.billing_country,
            "billing_country": request.billing_country,
            "shipping_country": request.shipping_country,
            "billing_shipping_match": int(request.billing_country == request.shipping_country),
            "ip_billing_match": 1,
            "ip_shipping_match": int(request.billing_country == request.shipping_country),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = _score_single(txn_dict)

        # Derive risk factors from transaction features
        risk_factors = []
        if request.amount > 5000:
            risk_factors.append("high_amount")
        elif request.amount > 1000:
            risk_factors.append("elevated_amount")
        if request.billing_country != request.shipping_country:
            risk_factors.append("geo_mismatch")
        if request.device_id == "unknown":
            risk_factors.append("unknown_device")
        if request.merchant_category in ("gambling", "crypto", "gift_cards"):
            risk_factors.append("high_risk_category")

        SCORING_REQUESTS.labels(endpoint="/predict", status="success").inc()

        return {
            "decision": result.action,
            "score": result.fraud_score,
            "risk_factors": risk_factors,
            "model_version": result.model_version,
        }

    except Exception as exc:
        SCORING_ERRORS.labels(error_type="predict_failure").inc()
        SCORING_REQUESTS.labels(endpoint="/predict", status="error").inc()
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        SCORING_LATENCY.observe(time.monotonic() - start)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness and readiness probe."""
    model_loaded = _cached_model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "scoring",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "review_threshold": REVIEW_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    data = generate_latest(REGISTRY)
    return PlainTextResponse(content=data.decode("utf-8"), media_type="text/plain")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
