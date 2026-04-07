"""
Tests for the Feature Enrichment Service.

Covers:
  - Successful enrichment of a valid transaction
  - Error handling for missing required fields
  - Velocity feature computation accuracy
  - Billing/shipping country match logic
"""

from __future__ import annotations

import copy
import time
from typing import Any

import pytest


class TestEnrichValidTransaction:
    """Tests for POST /enrich with a well-formed transaction."""

    def test_enrich_valid_transaction(
        self, enrichment_client, sample_transaction
    ):
        """A valid transaction should be enriched with all expected feature fields."""
        response = enrichment_client.post("/enrich", json=sample_transaction)

        assert response.status_code == 200
        data = response.json()

        # Original fields are preserved
        assert data["transaction_id"] == sample_transaction["transaction_id"]
        assert data["user_id"] == sample_transaction["user_id"]
        assert data["amount"] == sample_transaction["amount"]
        assert data["merchant_id"] == sample_transaction["merchant_id"]

        # Velocity features are present
        assert "velocity_10min" in data
        assert "velocity_1hr" in data
        assert "time_since_last_txn" in data
        assert isinstance(data["velocity_10min"], int)
        assert isinstance(data["velocity_1hr"], int)

        # Historical features from Feast are present
        assert "avg_amount_30d" in data
        assert "max_amount_30d" in data
        assert "transaction_count_24h" in data
        assert "unique_merchants_7d" in data
        assert "unique_countries_7d" in data
        assert "merchant_fraud_rate_30d" in data
        assert "merchant_avg_amount" in data
        assert "merchant_category_risk_score" in data

        # Derived features are present
        assert "amount_vs_avg_ratio" in data
        assert "billing_shipping_match" in data
        assert "is_high_risk_category" in data
        assert "is_odd_hour" in data

    def test_enrich_returns_correct_types(
        self, enrichment_client, sample_transaction
    ):
        """Enriched features should have the correct Python types."""
        response = enrichment_client.post("/enrich", json=sample_transaction)
        data = response.json()

        assert isinstance(data["velocity_10min"], int)
        assert isinstance(data["velocity_1hr"], int)
        assert isinstance(data["time_since_last_txn"], (int, float))
        assert isinstance(data["avg_amount_30d"], (int, float))
        assert isinstance(data["amount_vs_avg_ratio"], (int, float))
        assert isinstance(data["billing_shipping_match"], int)
        assert data["billing_shipping_match"] in (0, 1)
        assert isinstance(data["is_high_risk_category"], int)
        assert data["is_high_risk_category"] in (0, 1)
        assert isinstance(data["is_odd_hour"], int)
        assert data["is_odd_hour"] in (0, 1)


class TestEnrichMissingFields:
    """Tests for missing or invalid fields in the enrichment request."""

    def test_enrich_missing_fields_returns_error(self, enrichment_client):
        """A request missing required fields should return a 422 validation error."""
        # Missing transaction_id, user_id, amount, merchant_id (all required)
        response = enrichment_client.post("/enrich", json={})
        assert response.status_code == 422

    def test_enrich_missing_transaction_id(self, enrichment_client, sample_transaction):
        """Missing transaction_id should return 422."""
        txn = copy.deepcopy(sample_transaction)
        del txn["transaction_id"]
        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 422

    def test_enrich_missing_user_id(self, enrichment_client, sample_transaction):
        """Missing user_id should return 422."""
        txn = copy.deepcopy(sample_transaction)
        del txn["user_id"]
        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 422

    def test_enrich_missing_amount(self, enrichment_client, sample_transaction):
        """Missing amount should return 422."""
        txn = copy.deepcopy(sample_transaction)
        del txn["amount"]
        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 422

    def test_enrich_invalid_amount_zero(self, enrichment_client, sample_transaction):
        """Amount of 0 should return 422 (amount must be > 0)."""
        txn = copy.deepcopy(sample_transaction)
        txn["amount"] = 0
        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 422

    def test_enrich_invalid_amount_negative(self, enrichment_client, sample_transaction):
        """Negative amount should return 422."""
        txn = copy.deepcopy(sample_transaction)
        txn["amount"] = -50.0
        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 422


class TestVelocityFeatures:
    """Tests for real-time velocity feature computation."""

    def test_velocity_features_computed_correctly(
        self, enrichment_client, sample_transaction
    ):
        """
        Sending multiple transactions for the same user in quick succession
        should produce increasing velocity counts.
        """
        # Use a unique user to avoid cross-test contamination
        txn = copy.deepcopy(sample_transaction)
        txn["user_id"] = "user_velocity_test"

        # First transaction: velocity should be 1
        txn["transaction_id"] = "txn_vel_001"
        r1 = enrichment_client.post("/enrich", json=txn)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["velocity_10min"] == 1
        assert d1["velocity_1hr"] == 1
        assert d1["time_since_last_txn"] == -1.0  # No previous transaction

        # Second transaction: velocity should be 2
        txn["transaction_id"] = "txn_vel_002"
        r2 = enrichment_client.post("/enrich", json=txn)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["velocity_10min"] == 2
        assert d2["velocity_1hr"] == 2
        assert d2["time_since_last_txn"] >= 0  # Should be small positive

        # Third transaction: velocity should be 3
        txn["transaction_id"] = "txn_vel_003"
        r3 = enrichment_client.post("/enrich", json=txn)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["velocity_10min"] == 3
        assert d3["velocity_1hr"] == 3


class TestBillingShippingMatch:
    """Tests for the billing_shipping_match derived feature."""

    def test_billing_shipping_match(self, enrichment_client, sample_transaction):
        """When billing and shipping countries match, the flag should be 1."""
        txn = copy.deepcopy(sample_transaction)
        txn["billing_country"] = "US"
        txn["shipping_country"] = "US"
        txn["user_id"] = "user_match_test_same"

        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 200
        assert response.json()["billing_shipping_match"] == 1

    def test_billing_shipping_mismatch(self, enrichment_client, sample_transaction):
        """When billing and shipping countries differ, the flag should be 0."""
        txn = copy.deepcopy(sample_transaction)
        txn["billing_country"] = "US"
        txn["shipping_country"] = "GB"
        txn["user_id"] = "user_match_test_diff"

        response = enrichment_client.post("/enrich", json=txn)
        assert response.status_code == 200
        assert response.json()["billing_shipping_match"] == 0


class TestHealthAndMetrics:
    """Tests for operational endpoints."""

    def test_health_endpoint(self, enrichment_client):
        """The /health endpoint should return 200 with service info."""
        response = enrichment_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "enrichment"

    def test_metrics_endpoint(self, enrichment_client):
        """The /metrics endpoint should return Prometheus text format."""
        response = enrichment_client.get("/metrics")
        assert response.status_code == 200
        assert "enrichment_latency_seconds" in response.text
