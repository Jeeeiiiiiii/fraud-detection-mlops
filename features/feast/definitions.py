"""
Feast Feature Definitions for Fraud Detection
==============================================
Defines entities, feature views, and on-demand feature views used by the
fraud-detection scoring pipeline.

Entity hierarchy:
  - user        — the cardholder / account owner
  - transaction — a single payment event
  - merchant    — the seller / payee
  - device      — the device fingerprint used for a transaction

Feature Views:
  - user_transaction_features  (batch, from BigQuery / Parquet)
  - device_features            (batch)
  - merchant_features          (batch)
  - velocity_features          (stream / push source)

On-Demand Feature Views (computed at request time):
  - on_demand_fraud_signals    (billing/shipping match, amount ratio, odd-hour)
"""

from __future__ import annotations

from datetime import timedelta

from feast import (
    BigQuerySource,
    Entity,
    Feature,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Bool, Float32, Float64, Int64, String

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

user_entity = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.STRING,
    description="A cardholder or account owner identified by user_id.",
)

transaction_entity = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="A single payment event.",
)

merchant_entity = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    value_type=ValueType.STRING,
    description="A merchant / seller.",
)

device_entity = Entity(
    name="device",
    join_keys=["device_id"],
    value_type=ValueType.STRING,
    description="A device fingerprint.",
)

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

# Production: BigQuery tables.  Switch to FileSource for local dev.

user_txn_source = BigQuerySource(
    name="user_transaction_source",
    table="fraud_detection.user_transaction_features",
    timestamp_field="feature_timestamp",
    created_timestamp_column="created_at",
    description="Pre-computed user-level transaction aggregates.",
)

device_source = BigQuerySource(
    name="device_source",
    table="fraud_detection.device_features",
    timestamp_field="feature_timestamp",
    created_timestamp_column="created_at",
    description="Device-level features.",
)

merchant_source = BigQuerySource(
    name="merchant_source",
    table="fraud_detection.merchant_features",
    timestamp_field="feature_timestamp",
    created_timestamp_column="created_at",
    description="Merchant-level features.",
)

velocity_push_source = PushSource(
    name="velocity_push_source",
    schema=[
        Field(name="user_id", dtype=String),
        Field(name="transactions_last_10min", dtype=Int64),
        Field(name="transactions_last_1h", dtype=Int64),
        Field(name="amount_last_10min", dtype=Float64),
        Field(name="feature_timestamp", dtype=Int64),
    ],
    description="Real-time velocity features pushed from the streaming pipeline.",
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user_entity],
    ttl=timedelta(hours=24),  # Features are refreshed daily
    schema=[
        Field(name="transaction_count_1h", dtype=Int64,
              description="Number of transactions by this user in the last hour."),
        Field(name="transaction_count_24h", dtype=Int64,
              description="Number of transactions by this user in the last 24 hours."),
        Field(name="avg_amount_30d", dtype=Float64,
              description="Average transaction amount over the last 30 days."),
        Field(name="max_amount_30d", dtype=Float64,
              description="Maximum single transaction amount in the last 30 days."),
        Field(name="unique_merchants_7d", dtype=Int64,
              description="Count of distinct merchants used in the last 7 days."),
        Field(name="unique_countries_7d", dtype=Int64,
              description="Count of distinct transaction countries in the last 7 days."),
    ],
    source=user_txn_source,
    online=True,
    tags={"team": "fraud", "tier": "critical"},
)

device_features = FeatureView(
    name="device_features",
    entities=[device_entity],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="device_first_seen_days", dtype=Int64,
              description="Days since device was first observed."),
        Field(name="device_transaction_count", dtype=Int64,
              description="Total transactions from this device."),
        Field(name="device_unique_users", dtype=Int64,
              description="Number of distinct users from this device."),
    ],
    source=device_source,
    online=True,
    tags={"team": "fraud"},
)

merchant_features = FeatureView(
    name="merchant_features",
    entities=[merchant_entity],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="merchant_fraud_rate_30d", dtype=Float64,
              description="Fraud rate at this merchant over the last 30 days."),
        Field(name="merchant_avg_amount", dtype=Float64,
              description="Average transaction amount at this merchant."),
        Field(name="merchant_category_risk_score", dtype=Float64,
              description="Risk score (0-1) for the merchant's category."),
    ],
    source=merchant_source,
    online=True,
    tags={"team": "fraud"},
)

velocity_features = FeatureView(
    name="velocity_features",
    entities=[user_entity],
    ttl=timedelta(minutes=15),  # Very fresh — pushed from stream
    schema=[
        Field(name="transactions_last_10min", dtype=Int64,
              description="Transaction count in the last 10 minutes."),
        Field(name="transactions_last_1h", dtype=Int64,
              description="Transaction count in the last hour."),
        Field(name="amount_last_10min", dtype=Float64,
              description="Total amount transacted in the last 10 minutes."),
    ],
    source=velocity_push_source,
    online=True,
    tags={"team": "fraud", "tier": "critical"},
)

# ---------------------------------------------------------------------------
# On-Demand Feature Views (computed at request time)
# ---------------------------------------------------------------------------

# Request source defines fields supplied at inference time
transaction_request_source = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="billing_country", dtype=String),
        Field(name="shipping_country", dtype=String),
        Field(name="amount", dtype=Float64),
        Field(name="transaction_hour", dtype=Int64),
    ],
)


@on_demand_feature_view(
    sources=[
        transaction_request_source,
        user_transaction_features,
    ],
    schema=[
        Field(name="billing_shipping_match", dtype=Int64),
        Field(name="amount_vs_user_avg_ratio", dtype=Float64),
        Field(name="is_odd_hour", dtype=Int64),
    ],
)
def on_demand_fraud_signals(inputs: dict) -> dict:
    """
    Compute request-time features that depend on both the raw transaction
    and historical user features.

    Features:
      - billing_shipping_match: 1 if billing == shipping country, else 0
      - amount_vs_user_avg_ratio: current amount / user's 30-day avg amount
      - is_odd_hour: 1 if the transaction occurs between 00:00 and 05:00 UTC
    """
    import pandas as pd

    df = pd.DataFrame(inputs)

    billing_shipping_match = (
        df["billing_country"] == df["shipping_country"]
    ).astype(int)

    # Guard against division by zero when user has no history
    avg_amount = df["avg_amount_30d"].replace(0, 1.0)
    amount_vs_user_avg_ratio = df["amount"] / avg_amount

    is_odd_hour = ((df["transaction_hour"] >= 0) & (df["transaction_hour"] < 5)).astype(int)

    return {
        "billing_shipping_match": billing_shipping_match,
        "amount_vs_user_avg_ratio": amount_vs_user_avg_ratio,
        "is_odd_hour": is_odd_hour,
    }
