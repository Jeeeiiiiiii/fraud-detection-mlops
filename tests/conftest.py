"""
Shared pytest fixtures for the fraud-detection-mlops test suite.

Provides:
  - sample_transaction:     A single valid transaction dict.
  - sample_transactions:    A batch of diverse transaction dicts (normal + suspicious).
  - mock_feast_store:       A patched Feast FeatureStore that returns deterministic features.
  - mock_model:             A stub XGBoost model that returns predictable fraud scores.
  - enrichment_client:      A FastAPI TestClient for the enrichment service.
  - scoring_client:         A FastAPI TestClient for the scoring service.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Sample transaction data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transaction() -> dict[str, Any]:
    """Return a single valid transaction payload matching the enrichment API schema."""
    return {
        "transaction_id": "txn_test_000001",
        "user_id": "user_42",
        "amount": 149.99,
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


@pytest.fixture
def sample_transactions() -> list[dict[str, Any]]:
    """
    Return a batch of 5 diverse transactions covering various risk profiles.

    Includes:
      - Normal domestic transaction
      - High-amount electronics purchase
      - Geographic mismatch (billing != shipping)
      - Odd-hour small transaction
      - Gift card purchase from high-risk IP
    """
    return [
        {
            "transaction_id": "txn_test_norm_001",
            "user_id": "user_100",
            "amount": 45.00,
            "currency": "USD",
            "merchant_id": "merch_001",
            "merchant_category": "grocery",
            "card_type": "visa",
            "device_type": "mobile",
            "ip_country": "US",
            "billing_country": "US",
            "shipping_country": "US",
            "timestamp": "2025-06-15T10:00:00Z",
        },
        {
            "transaction_id": "txn_test_high_002",
            "user_id": "user_200",
            "amount": 4999.99,
            "currency": "USD",
            "merchant_id": "merch_002",
            "merchant_category": "electronics",
            "card_type": "amex",
            "device_type": "desktop",
            "ip_country": "US",
            "billing_country": "US",
            "shipping_country": "US",
            "timestamp": "2025-06-15T15:30:00Z",
        },
        {
            "transaction_id": "txn_test_geo_003",
            "user_id": "user_300",
            "amount": 299.50,
            "currency": "EUR",
            "merchant_id": "merch_003",
            "merchant_category": "clothing",
            "card_type": "mastercard",
            "device_type": "desktop",
            "ip_country": "NG",
            "billing_country": "US",
            "shipping_country": "GB",
            "timestamp": "2025-06-15T12:00:00Z",
        },
        {
            "transaction_id": "txn_test_odd_004",
            "user_id": "user_400",
            "amount": 1.50,
            "currency": "USD",
            "merchant_id": "merch_004",
            "merchant_category": "digital_goods",
            "card_type": "discover",
            "device_type": "tablet",
            "ip_country": "US",
            "billing_country": "US",
            "shipping_country": "US",
            "timestamp": "2025-06-15T03:15:00Z",
        },
        {
            "transaction_id": "txn_test_gift_005",
            "user_id": "user_500",
            "amount": 500.00,
            "currency": "USD",
            "merchant_id": "merch_005",
            "merchant_category": "gift_cards",
            "card_type": "visa",
            "device_type": "unknown",
            "ip_country": "RU",
            "billing_country": "US",
            "shipping_country": "CN",
            "timestamp": "2025-06-15T02:00:00Z",
        },
    ]


# ---------------------------------------------------------------------------
# Mock Feast feature store
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_feast_store():
    """
    Patch the Feast FeatureStore so tests do not require a running Redis/Feast
    instance. Returns deterministic feature values.
    """
    mock_store = MagicMock()

    def _get_online_features(features, entity_rows):
        """Return a mock response with realistic feature values."""
        result = MagicMock()
        n = len(entity_rows)
        result.to_dict.return_value = {
            "user_id": [row.get("user_id", "unknown") for row in entity_rows],
            "merchant_id": [row.get("merchant_id", "unknown") for row in entity_rows],
            "avg_amount_30d": [150.0] * n,
            "max_amount_30d": [500.0] * n,
            "transaction_count_24h": [5] * n,
            "unique_merchants_7d": [8] * n,
            "unique_countries_7d": [1] * n,
            "merchant_fraud_rate_30d": [0.01] * n,
            "merchant_avg_amount": [120.0] * n,
            "merchant_category_risk_score": [0.1] * n,
        }
        return result

    mock_store.get_online_features = _get_online_features

    with patch(
        "features.enrichment_service.app._get_feast_store",
        return_value=mock_store,
    ):
        yield mock_store


# ---------------------------------------------------------------------------
# Mock XGBoost model
# ---------------------------------------------------------------------------

class StubFraudModel:
    """
    A deterministic stub that mimics the XGBoost model interface.

    Scoring logic:
      - Returns a low score (0.1) for amounts < 100
      - Returns a medium score (0.5) for amounts 100-1000
      - Returns a high score (0.95) for amounts > 1000
    This lets tests verify threshold-based decision logic.
    """

    n_features_in_ = 13  # matches FEATURE_KEYS length in predict.py

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [P(legit), P(fraud)] for each row."""
        n = X.shape[0]
        scores = np.zeros(n)
        # Use the 7th feature (index 6) which is 'amount' in FEATURE_KEYS ordering
        for i in range(n):
            amount_feature = X[i, 6] if X.shape[1] > 6 else 100.0
            if amount_feature < 100:
                scores[i] = 0.1
            elif amount_feature < 1000:
                scores[i] = 0.5
            else:
                scores[i] = 0.95
        return np.column_stack([1 - scores, scores])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction at 0.5 threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@pytest.fixture
def mock_model():
    """
    Patch the scoring service to use the StubFraudModel instead of loading
    from MLflow.
    """
    stub = StubFraudModel()
    with patch("serving.predict._cached_model", stub), \
         patch("serving.predict._cached_threshold", 0.5):
        yield stub


# ---------------------------------------------------------------------------
# FastAPI test clients
# ---------------------------------------------------------------------------

@pytest.fixture
def enrichment_client(mock_feast_store):
    """Return a FastAPI TestClient for the enrichment service."""
    from fastapi.testclient import TestClient
    from features.enrichment_service.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def scoring_client(mock_model):
    """Return a FastAPI TestClient for the scoring service."""
    from fastapi.testclient import TestClient
    from serving.predict import app

    with TestClient(app) as client:
        yield client
