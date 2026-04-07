#!/usr/bin/env python3
"""
Transaction Data Generator
===========================
Generates realistic synthetic transaction data for fraud-detection model
training and evaluation.

Usage
-----
    python generate_transactions.py --output-dir ./output --n-transactions 200000

The generator creates a baseline of legitimate transactions and then injects
fraud patterns (velocity attacks, geographic mismatches, amount anomalies,
account takeovers, and card-testing) to reach an approximate 2 % fraud rate.

Outputs
-------
- ``transactions.csv``     Comma-separated file.
- ``transactions.parquet`` Apache Parquet (Snappy compressed).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Local import — fraud_patterns lives in the same package.
from fraud_patterns import inject_all_patterns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "clothing", "travel", "entertainment", "healthcare", "education",
    "subscription", "utilities", "insurance", "charity", "gift_cards",
    "jewelry", "digital_goods", "home_improvement", "automotive", "sports",
]

COUNTRIES = [
    "US", "GB", "CA", "DE", "FR", "BR", "IN", "JP", "AU", "MX",
    "ZA", "KR", "AE", "SG", "NL", "IT", "ES", "SE", "NO", "NZ",
]

CARD_TYPES = ["visa", "mastercard", "amex", "discover"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "BRL", "JPY"]

# Spending distribution parameters per category (mean, std for log-normal)
CATEGORY_AMOUNT_PARAMS: dict[str, tuple[float, float]] = {
    "grocery":          (3.5, 0.6),
    "restaurant":       (3.2, 0.5),
    "gas_station":      (3.6, 0.3),
    "online_retail":    (3.8, 0.8),
    "electronics":      (5.5, 1.0),
    "clothing":         (4.0, 0.7),
    "travel":           (5.8, 0.9),
    "entertainment":    (3.0, 0.6),
    "healthcare":       (4.5, 1.0),
    "education":        (4.2, 0.8),
    "subscription":     (2.5, 0.5),
    "utilities":        (4.0, 0.4),
    "insurance":        (5.0, 0.5),
    "charity":          (3.0, 1.0),
    "gift_cards":       (3.5, 0.6),
    "jewelry":          (5.5, 1.2),
    "digital_goods":    (2.8, 0.7),
    "home_improvement": (4.5, 0.8),
    "automotive":       (5.0, 0.9),
    "sports":           (3.8, 0.7),
}

# Hour-of-day weights (higher = more transactions at that hour, UTC)
HOUR_WEIGHTS = np.array([
    0.3, 0.2, 0.15, 0.1, 0.1, 0.15, 0.3, 0.6,
    0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4,
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# Legitimate transaction generator
# ---------------------------------------------------------------------------

def _generate_user_pool(n_users: int) -> pd.DataFrame:
    """Pre-generate a pool of users with home country and preferred card."""
    return pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(1, n_users + 1)],
        "home_country": np.random.choice(COUNTRIES, size=n_users, p=None),
        "preferred_card": np.random.choice(CARD_TYPES, size=n_users),
        "preferred_device": np.random.choice(DEVICE_TYPES, size=n_users),
    })


def _generate_merchant_pool(n_merchants: int) -> pd.DataFrame:
    """Pre-generate a pool of merchants."""
    categories = np.random.choice(MERCHANT_CATEGORIES, size=n_merchants)
    return pd.DataFrame({
        "merchant_id": [f"merch_{i}" for i in range(1, n_merchants + 1)],
        "merchant_category": categories,
    })


def generate_legitimate_transactions(
    n_transactions: int,
    n_users: int = 50_000,
    n_merchants: int = 5_000,
    base_time: datetime | None = None,
    days_span: int = 30,
) -> pd.DataFrame:
    """
    Generate a DataFrame of legitimate (non-fraud) transactions.

    Parameters
    ----------
    n_transactions : int
        Total number of legitimate transactions.
    n_users : int
        Size of the user pool.
    n_merchants : int
        Size of the merchant pool.
    base_time : datetime, optional
        Start of the time window; defaults to 30 days ago.
    days_span : int
        Number of days the transactions span.

    Returns
    -------
    pd.DataFrame
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=days_span))
    users = _generate_user_pool(n_users)
    merchants = _generate_merchant_pool(n_merchants)

    logger.info("Generating %d legitimate transactions ...", n_transactions)

    # Sample users and merchants
    user_indices = np.random.randint(0, n_users, size=n_transactions)
    merchant_indices = np.random.randint(0, n_merchants, size=n_transactions)

    sampled_users = users.iloc[user_indices].reset_index(drop=True)
    sampled_merchants = merchants.iloc[merchant_indices].reset_index(drop=True)

    # Generate amounts based on merchant category
    amounts = np.zeros(n_transactions)
    for cat, params in CATEGORY_AMOUNT_PARAMS.items():
        mask = sampled_merchants["merchant_category"].values == cat
        count = mask.sum()
        if count > 0:
            amounts[mask] = np.round(
                np.random.lognormal(mean=params[0], sigma=params[1], size=count), 2
            )
    # Fill any remaining (shouldn't happen, but safety net)
    zero_mask = amounts == 0
    if zero_mask.any():
        amounts[zero_mask] = np.round(
            np.random.lognormal(mean=3.5, sigma=0.7, size=zero_mask.sum()), 2
        )

    # Generate timestamps with realistic hourly distribution
    hours = np.random.choice(24, size=n_transactions, p=HOUR_WEIGHTS)
    day_offsets = np.random.randint(0, days_span, size=n_transactions)
    minute_offsets = np.random.randint(0, 60, size=n_transactions)
    second_offsets = np.random.randint(0, 60, size=n_transactions)

    timestamps = pd.to_datetime([
        base_time + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offsets, hours, minute_offsets, second_offsets)
    ])

    # Currencies — users mostly transact in their home currency
    currency_map = {
        "US": "USD", "CA": "CAD", "GB": "GBP", "AU": "AUD", "BR": "BRL",
        "JP": "JPY", "DE": "EUR", "FR": "EUR", "NL": "EUR", "IT": "EUR",
        "ES": "EUR", "SE": "EUR", "NO": "EUR",
    }
    home_countries = sampled_users["home_country"].values
    currencies = np.array([
        currency_map.get(c, "USD") for c in home_countries
    ])

    # For legitimate transactions, IP / billing / shipping usually match
    ip_countries = home_countries.copy()
    billing_countries = home_countries.copy()
    shipping_countries = home_countries.copy()

    # ~5 % of legit transactions have a minor mismatch (travelling, gifts)
    travel_mask = np.random.random(n_transactions) < 0.05
    ip_countries[travel_mask] = np.random.choice(COUNTRIES, size=travel_mask.sum())

    gift_mask = np.random.random(n_transactions) < 0.03
    shipping_countries[gift_mask] = np.random.choice(COUNTRIES, size=gift_mask.sum())

    # Device: mostly preferred, with ~10 % variation
    devices = sampled_users["preferred_device"].values.copy()
    device_switch_mask = np.random.random(n_transactions) < 0.10
    devices[device_switch_mask] = np.random.choice(DEVICE_TYPES, size=device_switch_mask.sum())

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:012d}" for i in range(n_transactions)],
        "user_id": sampled_users["user_id"].values,
        "amount": amounts,
        "currency": currencies,
        "merchant_id": sampled_merchants["merchant_id"].values,
        "merchant_category": sampled_merchants["merchant_category"].values,
        "card_type": sampled_users["preferred_card"].values,
        "device_type": devices,
        "ip_country": ip_countries,
        "billing_country": billing_countries,
        "shipping_country": shipping_countries,
        "timestamp": timestamps,
        "is_fraud": 0,
        "fraud_pattern": "none",
    })

    logger.info("Legitimate transactions generated: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Combine and save
# ---------------------------------------------------------------------------

def generate_dataset(
    n_transactions: int = 200_000,
    fraud_rate: float = 0.02,
    output_dir: str = "./output",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full dataset with legitimate + fraud transactions.

    Parameters
    ----------
    n_transactions : int
        Approximate total number of transactions (legit + fraud).
    fraud_rate : float
        Target fraud rate (0-1).
    output_dir : str
        Directory to write CSV and Parquet files.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Complete dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_fraud_target = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud_target

    base_time = datetime(2025, 1, 1)

    # Generate legitimate transactions
    legit_df = generate_legitimate_transactions(
        n_transactions=n_legit,
        base_time=base_time,
        days_span=30,
    )

    # Generate fraud transactions — scale to reach target count
    # inject_all_patterns with scale=1.0 produces ~4000-5000 fraud txns
    # We adjust scale to get close to n_fraud_target
    scale = max(0.1, n_fraud_target / 4500)
    fraud_df = inject_all_patterns(base_time=base_time, scale=scale)

    # If we have more fraud than target, sample down
    if len(fraud_df) > n_fraud_target:
        fraud_df = fraud_df.sample(n=n_fraud_target, random_state=seed).reset_index(drop=True)
    # If we have fewer, generate more geo_mismatch to fill
    elif len(fraud_df) < n_fraud_target:
        from fraud_patterns import geo_mismatch
        extra = geo_mismatch(
            n_transactions=n_fraud_target - len(fraud_df),
            base_time=base_time,
        )
        fraud_df = pd.concat([fraud_df, extra], ignore_index=True)

    # Combine
    full_df = pd.concat([legit_df, fraud_df], ignore_index=True)
    full_df.sort_values("timestamp", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    actual_fraud_rate = full_df["is_fraud"].mean()
    logger.info(
        "Dataset summary: %d total transactions, %d fraud (%.2f%%)",
        len(full_df),
        full_df["is_fraud"].sum(),
        actual_fraud_rate * 100,
    )

    # Log fraud pattern distribution
    fraud_dist = full_df[full_df["is_fraud"] == 1]["fraud_pattern"].value_counts()
    logger.info("Fraud pattern distribution:\n%s", fraud_dist.to_string())

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "transactions.csv"
    parquet_path = output_path / "transactions.parquet"

    full_df.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s (%.1f MB)", csv_path, csv_path.stat().st_size / 1e6)

    full_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Saved Parquet: %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)

    return full_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for fraud detection.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Directory to write output files (default: ./output)",
    )
    parser.add_argument(
        "--n-transactions", type=int, default=200_000,
        help="Approximate number of transactions to generate (default: 200000)",
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=0.02,
        help="Target fraud rate, 0-1 (default: 0.02)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_dataset(
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
