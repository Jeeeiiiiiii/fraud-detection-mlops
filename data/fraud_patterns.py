"""
Fraud Pattern Injection Module
==============================
Provides functions to inject realistic fraud patterns into clean transaction data.
Each pattern type mimics real-world fraud behaviors observed in payment systems.

Pattern Types:
  - velocity_attack: Many transactions in a short time window (card stolen, rapid use)
  - geo_mismatch: Billing/shipping/IP country inconsistencies
  - amount_anomaly: Transactions far outside a user's normal spending range
  - account_takeover: Device change + password reset + large purchase pattern
  - card_testing: Many small-value transactions to test if a stolen card is active
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_RISK_MERCHANT_CATEGORIES = [
    "electronics", "gift_cards", "jewelry", "crypto_exchange",
    "money_transfer", "gambling", "digital_goods",
]

COUNTRIES = [
    "US", "GB", "CA", "DE", "FR", "BR", "NG", "RU", "CN", "IN",
    "JP", "AU", "MX", "ZA", "KR", "AE", "SG", "PH", "VN", "UA",
]

HIGH_RISK_COUNTRIES = ["NG", "RU", "CN", "UA", "VN", "PH"]

DEVICE_TYPES = ["desktop", "mobile", "tablet", "unknown"]
CARD_TYPES = ["visa", "mastercard", "amex", "discover"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "BRL", "JPY"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between *start* and *end*."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def _generate_transaction_id() -> str:
    """Generate a realistic-looking transaction ID."""
    return f"txn_{random.randint(10**11, 10**12 - 1)}"


def _pick_merchant(category: str | None = None) -> tuple[str, str]:
    """Return (merchant_id, merchant_category)."""
    cat = category or random.choice(HIGH_RISK_MERCHANT_CATEGORIES)
    mid = f"merch_{random.randint(1000, 9999)}"
    return mid, cat


# ---------------------------------------------------------------------------
# Pattern: Velocity Attack
# ---------------------------------------------------------------------------

def velocity_attack(
    n_attacks: int = 50,
    txns_per_attack: int | tuple[int, int] = (5, 20),
    window_minutes: int = 10,
    base_time: datetime | None = None,
    amount_range: tuple[float, float] = (50.0, 500.0),
) -> pd.DataFrame:
    """
    Generate velocity-attack fraud transactions.

    A velocity attack involves a stolen card being used many times in a very
    short window (typically < 10 minutes) before the cardholder notices.

    Parameters
    ----------
    n_attacks : int
        Number of distinct velocity attacks to generate.
    txns_per_attack : int or (min, max)
        Transactions per attack burst.
    window_minutes : int
        Time window in minutes for each burst.
    base_time : datetime, optional
        Anchor time; defaults to now - 30 days.
    amount_range : (float, float)
        (min, max) amount per fraudulent transaction.

    Returns
    -------
    pd.DataFrame
        Fraudulent transactions with ``is_fraud=1`` and ``fraud_pattern='velocity_attack'``.
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=30))
    records: list[dict[str, Any]] = []

    for _ in range(n_attacks):
        user_id = f"user_{random.randint(1, 50000)}"
        card = random.choice(CARD_TYPES)
        device = random.choice(DEVICE_TYPES)
        country = random.choice(COUNTRIES)
        merchant_id, merchant_cat = _pick_merchant()

        if isinstance(txns_per_attack, tuple):
            n_txns = random.randint(*txns_per_attack)
        else:
            n_txns = txns_per_attack

        attack_start = _random_timestamp(base_time, base_time + timedelta(days=30))

        for i in range(n_txns):
            offset_sec = random.randint(0, window_minutes * 60)
            ts = attack_start + timedelta(seconds=offset_sec)
            amount = round(random.uniform(*amount_range), 2)

            records.append({
                "transaction_id": _generate_transaction_id(),
                "user_id": user_id,
                "amount": amount,
                "currency": random.choice(CURRENCIES),
                "merchant_id": merchant_id,
                "merchant_category": merchant_cat,
                "card_type": card,
                "device_type": device,
                "ip_country": country,
                "billing_country": country,
                "shipping_country": country,
                "timestamp": ts,
                "is_fraud": 1,
                "fraud_pattern": "velocity_attack",
            })

    logger.info("Generated %d velocity-attack transactions from %d attacks", len(records), n_attacks)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pattern: Geographic Mismatch
# ---------------------------------------------------------------------------

def geo_mismatch(
    n_transactions: int = 500,
    base_time: datetime | None = None,
    amount_range: tuple[float, float] = (100.0, 2000.0),
) -> pd.DataFrame:
    """
    Generate geographic-mismatch fraud transactions.

    The billing country, shipping country, and/or IP country are intentionally
    mismatched, mimicking stolen card details used from a different location.

    Parameters
    ----------
    n_transactions : int
        Number of fraudulent transactions to generate.
    base_time : datetime, optional
        Anchor time; defaults to now - 30 days.
    amount_range : (float, float)
        (min, max) amount per transaction.

    Returns
    -------
    pd.DataFrame
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=30))
    records: list[dict[str, Any]] = []

    for _ in range(n_transactions):
        billing = random.choice(COUNTRIES)
        # Ensure at least one mismatch
        shipping = random.choice([c for c in COUNTRIES if c != billing])
        ip_country = random.choice([billing, shipping, random.choice(HIGH_RISK_COUNTRIES)])

        merchant_id, merchant_cat = _pick_merchant()
        ts = _random_timestamp(base_time, base_time + timedelta(days=30))
        amount = round(random.uniform(*amount_range), 2)

        records.append({
            "transaction_id": _generate_transaction_id(),
            "user_id": f"user_{random.randint(1, 50000)}",
            "amount": amount,
            "currency": random.choice(CURRENCIES),
            "merchant_id": merchant_id,
            "merchant_category": merchant_cat,
            "card_type": random.choice(CARD_TYPES),
            "device_type": random.choice(DEVICE_TYPES),
            "ip_country": ip_country,
            "billing_country": billing,
            "shipping_country": shipping,
            "timestamp": ts,
            "is_fraud": 1,
            "fraud_pattern": "geo_mismatch",
        })

    logger.info("Generated %d geo-mismatch fraud transactions", len(records))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pattern: Amount Anomaly
# ---------------------------------------------------------------------------

def amount_anomaly(
    n_transactions: int = 400,
    base_time: datetime | None = None,
    high_amount_range: tuple[float, float] = (2000.0, 15000.0),
) -> pd.DataFrame:
    """
    Generate amount-anomaly fraud transactions.

    These are unusually large purchases that deviate significantly from the
    user's typical spending, often targeting high-value electronics or
    gift cards.

    Parameters
    ----------
    n_transactions : int
        Number of fraudulent transactions.
    base_time : datetime, optional
        Anchor time.
    high_amount_range : (float, float)
        (min, max) for the anomalous amounts.

    Returns
    -------
    pd.DataFrame
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=30))
    records: list[dict[str, Any]] = []

    for _ in range(n_transactions):
        ts = _random_timestamp(base_time, base_time + timedelta(days=30))
        # Skew toward very high amounts with a log-normal tail
        amount = round(
            np.random.lognormal(mean=np.log(3000), sigma=0.6), 2
        )
        amount = max(high_amount_range[0], min(amount, high_amount_range[1]))

        merchant_id, merchant_cat = _pick_merchant(
            random.choice(["electronics", "jewelry", "gift_cards"])
        )

        country = random.choice(COUNTRIES)
        records.append({
            "transaction_id": _generate_transaction_id(),
            "user_id": f"user_{random.randint(1, 50000)}",
            "amount": amount,
            "currency": random.choice(CURRENCIES),
            "merchant_id": merchant_id,
            "merchant_category": merchant_cat,
            "card_type": random.choice(CARD_TYPES),
            "device_type": random.choice(DEVICE_TYPES),
            "ip_country": country,
            "billing_country": country,
            "shipping_country": country,
            "timestamp": ts,
            "is_fraud": 1,
            "fraud_pattern": "amount_anomaly",
        })

    logger.info("Generated %d amount-anomaly fraud transactions", len(records))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pattern: Account Takeover
# ---------------------------------------------------------------------------

def account_takeover(
    n_attacks: int = 80,
    base_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Generate account-takeover fraud transactions.

    Simulates a scenario where a fraudster gains access to a legitimate
    account, changes the device, and makes large purchases from a new
    location.  Typically preceded by a period of inactivity.

    Parameters
    ----------
    n_attacks : int
        Number of distinct account takeover events.
    base_time : datetime, optional
        Anchor time.

    Returns
    -------
    pd.DataFrame
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=30))
    records: list[dict[str, Any]] = []

    for _ in range(n_attacks):
        user_id = f"user_{random.randint(1, 50000)}"
        # Attacker uses a new device and a high-risk IP country
        new_device = "unknown"
        new_ip = random.choice(HIGH_RISK_COUNTRIES)
        original_country = random.choice(["US", "GB", "CA", "DE", "FR", "AU"])

        attack_start = _random_timestamp(base_time, base_time + timedelta(days=28))
        # 1-3 large purchases within a few hours
        n_purchases = random.randint(1, 3)
        for i in range(n_purchases):
            ts = attack_start + timedelta(minutes=random.randint(0, 180))
            amount = round(random.uniform(500, 8000), 2)
            merchant_id, merchant_cat = _pick_merchant(
                random.choice(["electronics", "gift_cards", "digital_goods"])
            )

            records.append({
                "transaction_id": _generate_transaction_id(),
                "user_id": user_id,
                "amount": amount,
                "currency": random.choice(CURRENCIES),
                "merchant_id": merchant_id,
                "merchant_category": merchant_cat,
                "card_type": random.choice(CARD_TYPES),
                "device_type": new_device,
                "ip_country": new_ip,
                "billing_country": original_country,
                "shipping_country": random.choice([original_country, new_ip]),
                "timestamp": ts,
                "is_fraud": 1,
                "fraud_pattern": "account_takeover",
            })

    logger.info("Generated %d account-takeover transactions from %d attacks", len(records), n_attacks)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pattern: Card Testing
# ---------------------------------------------------------------------------

def card_testing(
    n_attacks: int = 60,
    txns_per_attack: int | tuple[int, int] = (10, 50),
    base_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Generate card-testing fraud transactions.

    Fraudsters test stolen card numbers by making many very small
    transactions (often $0.50-$2.00) against low-security merchants.
    If the card is not declined, they proceed with larger purchases.

    Parameters
    ----------
    n_attacks : int
        Number of distinct card-testing sessions.
    txns_per_attack : int or (min, max)
        Number of test transactions per session.
    base_time : datetime, optional
        Anchor time.

    Returns
    -------
    pd.DataFrame
    """
    base_time = base_time or (datetime.utcnow() - timedelta(days=30))
    records: list[dict[str, Any]] = []

    for _ in range(n_attacks):
        user_id = f"user_{random.randint(1, 50000)}"
        card = random.choice(CARD_TYPES)
        device = random.choice(DEVICE_TYPES)
        ip_country = random.choice(HIGH_RISK_COUNTRIES)

        if isinstance(txns_per_attack, tuple):
            n_txns = random.randint(*txns_per_attack)
        else:
            n_txns = txns_per_attack

        session_start = _random_timestamp(base_time, base_time + timedelta(days=30))

        for i in range(n_txns):
            offset_sec = random.randint(0, 300)  # within 5 minutes
            ts = session_start + timedelta(seconds=offset_sec)
            # Very small test amounts
            amount = round(random.uniform(0.50, 2.00), 2)
            merchant_id, merchant_cat = _pick_merchant(
                random.choice(["digital_goods", "charity", "subscription"])
            )

            records.append({
                "transaction_id": _generate_transaction_id(),
                "user_id": user_id,
                "amount": amount,
                "currency": "USD",
                "merchant_id": merchant_id,
                "merchant_category": merchant_cat,
                "card_type": card,
                "device_type": device,
                "ip_country": ip_country,
                "billing_country": random.choice(COUNTRIES),
                "shipping_country": random.choice(COUNTRIES),
                "timestamp": ts,
                "is_fraud": 1,
                "fraud_pattern": "card_testing",
            })

    logger.info("Generated %d card-testing transactions from %d attacks", len(records), n_attacks)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

ALL_PATTERNS = {
    "velocity_attack": velocity_attack,
    "geo_mismatch": geo_mismatch,
    "amount_anomaly": amount_anomaly,
    "account_takeover": account_takeover,
    "card_testing": card_testing,
}


def inject_all_patterns(
    base_time: datetime | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    """
    Generate fraud transactions for **all** pattern types with default
    parameters scaled by *scale*.

    Parameters
    ----------
    base_time : datetime, optional
        Anchor time for all patterns.
    scale : float
        Multiplier applied to the default counts. ``scale=0.5`` produces
        roughly half the default fraud volume.

    Returns
    -------
    pd.DataFrame
        Combined fraud DataFrame sorted by timestamp.
    """
    frames: list[pd.DataFrame] = []
    default_counts = {
        "velocity_attack": {"n_attacks": int(50 * scale)},
        "geo_mismatch": {"n_transactions": int(500 * scale)},
        "amount_anomaly": {"n_transactions": int(400 * scale)},
        "account_takeover": {"n_attacks": int(80 * scale)},
        "card_testing": {"n_attacks": int(60 * scale)},
    }

    for name, func in ALL_PATTERNS.items():
        kwargs: dict[str, Any] = default_counts.get(name, {})
        if base_time is not None:
            kwargs["base_time"] = base_time
        frames.append(func(**kwargs))

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("Total fraud transactions generated: %d", len(combined))
    return combined
