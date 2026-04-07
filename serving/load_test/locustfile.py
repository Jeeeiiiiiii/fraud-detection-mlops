#!/usr/bin/env python3
"""
Locust Load Test for the Fraud Scoring Service
===============================================
Simulates realistic production traffic patterns against the scoring API.

Target metrics:
  - Sustained throughput: 1000 req/s
  - p99 latency: < 50 ms
  - Error rate: < 0.1%

Usage
-----
    # Start Locust web UI
    locust -f locustfile.py --host http://localhost:8080

    # Headless mode for CI/CD
    locust -f locustfile.py \
        --host http://localhost:8080 \
        --headless \
        --users 200 \
        --spawn-rate 20 \
        --run-time 5m \
        --csv results/load_test

    # With custom thresholds (for CI gate)
    locust -f locustfile.py \
        --host http://localhost:8080 \
        --headless \
        --users 200 \
        --spawn-rate 20 \
        --run-time 2m \
        --csv results/load_test \
        --check-fail-ratio 0.001 \
        --check-p99-latency 50
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

# ---------------------------------------------------------------------------
# Synthetic transaction generators
# ---------------------------------------------------------------------------

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "clothing", "travel", "entertainment", "healthcare", "gift_cards",
    "jewelry", "digital_goods",
]
COUNTRIES = ["US", "GB", "CA", "DE", "FR", "BR", "IN", "JP", "AU", "MX"]
CARD_TYPES = ["visa", "mastercard", "amex", "discover"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]


def _generate_normal_transaction() -> dict:
    """Generate a normal (low-risk) transaction payload."""
    country = random.choice(COUNTRIES[:5])  # Mostly domestic
    return {
        "transaction_id": f"lt_txn_{random.randint(10**8, 10**9)}",
        "user_id": f"user_{random.randint(1, 50000)}",
        "amount": round(random.lognormvariate(3.5, 0.7), 2),
        "currency": "USD",
        "merchant_id": f"merch_{random.randint(1, 5000)}",
        "merchant_category": random.choice(MERCHANT_CATEGORIES[:6]),
        "card_type": random.choice(CARD_TYPES),
        "device_type": random.choice(DEVICE_TYPES),
        "ip_country": country,
        "billing_country": country,
        "shipping_country": country,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "velocity_10min": random.randint(0, 3),
        "velocity_1hr": random.randint(0, 10),
        "time_since_last_txn": random.uniform(60, 7200),
        "avg_amount_30d": random.uniform(50, 300),
        "billing_shipping_match": 1,
        "is_high_risk_category": 0,
        "is_odd_hour": 0,
    }


def _generate_suspicious_transaction() -> dict:
    """Generate a high-risk (potentially fraudulent) transaction payload."""
    return {
        "transaction_id": f"lt_txn_{random.randint(10**8, 10**9)}",
        "user_id": f"user_{random.randint(1, 50000)}",
        "amount": round(random.uniform(2000, 15000), 2),
        "currency": "USD",
        "merchant_id": f"merch_{random.randint(1, 5000)}",
        "merchant_category": random.choice(["electronics", "gift_cards", "jewelry"]),
        "card_type": random.choice(CARD_TYPES),
        "device_type": "unknown",
        "ip_country": random.choice(["NG", "RU", "CN"]),
        "billing_country": random.choice(COUNTRIES[:3]),
        "shipping_country": random.choice(COUNTRIES[5:]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "velocity_10min": random.randint(5, 20),
        "velocity_1hr": random.randint(15, 50),
        "time_since_last_txn": random.uniform(1, 30),
        "avg_amount_30d": random.uniform(50, 200),
        "billing_shipping_match": 0,
        "is_high_risk_category": 1,
        "is_odd_hour": 1,
    }


def _generate_batch(size: int) -> list[dict]:
    """Generate a batch of mixed transactions."""
    txns = []
    for _ in range(size):
        if random.random() < 0.02:  # ~2% fraud rate
            txns.append(_generate_suspicious_transaction())
        else:
            txns.append(_generate_normal_transaction())
    return txns


# ---------------------------------------------------------------------------
# Locust user classes
# ---------------------------------------------------------------------------

class FraudScoringUser(HttpUser):
    """
    Simulates a client sending transactions for fraud scoring.

    Traffic mix:
      - 85% single /score requests (normal transactions)
      - 10% single /score requests (suspicious transactions)
      - 5%  /score/batch requests (batch of 10-50)
    """
    wait_time = between(0.01, 0.05)  # Aggressive for 1000 req/s target

    @task(85)
    def score_normal(self):
        """Score a normal transaction."""
        payload = _generate_normal_transaction()
        with self.client.post(
            "/score",
            json=payload,
            catch_response=True,
            name="/score [normal]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("action") not in ("ALLOW", "REVIEW", "BLOCK"):
                    response.failure(f"Invalid action: {data.get('action')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(10)
    def score_suspicious(self):
        """Score a suspicious transaction."""
        payload = _generate_suspicious_transaction()
        with self.client.post(
            "/score",
            json=payload,
            catch_response=True,
            name="/score [suspicious]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("action") not in ("ALLOW", "REVIEW", "BLOCK"):
                    response.failure(f"Invalid action: {data.get('action')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def score_batch(self):
        """Score a batch of transactions."""
        batch_size = random.randint(10, 50)
        payload = _generate_batch(batch_size)
        with self.client.post(
            "/score/batch",
            json=payload,
            catch_response=True,
            name="/score/batch",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data) != batch_size:
                    response.failure(
                        f"Expected {batch_size} results, got {len(data)}"
                    )
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Periodic health check (simulates load balancer probes)."""
        self.client.get("/health", name="/health")


# ---------------------------------------------------------------------------
# Custom event hooks for CI/CD quality gates
# ---------------------------------------------------------------------------

_test_start_time = None
_results_summary = {
    "total_requests": 0,
    "total_failures": 0,
    "p99_latency_ms": 0.0,
}


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global _test_start_time
    _test_start_time = time.time()
    print("=" * 60)
    print("Fraud Scoring Load Test Starting")
    print(f"Target: 1000 req/s, p99 < 50ms, error rate < 0.1%")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary and check SLA gates."""
    stats = environment.runner.stats
    total = stats.total

    total_requests = total.num_requests
    total_failures = total.num_failures
    failure_rate = (total_failures / max(total_requests, 1)) * 100

    # Calculate p99 from response time percentiles
    p99 = total.get_response_time_percentile(0.99) or 0
    p95 = total.get_response_time_percentile(0.95) or 0
    p50 = total.get_response_time_percentile(0.50) or 0
    avg = total.avg_response_time or 0

    elapsed = time.time() - (_test_start_time or time.time())
    rps = total_requests / max(elapsed, 1)

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"  Duration:        {elapsed:.1f}s")
    print(f"  Total requests:  {total_requests:,}")
    print(f"  Total failures:  {total_failures:,}")
    print(f"  Failure rate:    {failure_rate:.3f}%")
    print(f"  Avg RPS:         {rps:.1f}")
    print(f"  Avg latency:     {avg:.1f}ms")
    print(f"  p50 latency:     {p50:.1f}ms")
    print(f"  p95 latency:     {p95:.1f}ms")
    print(f"  p99 latency:     {p99:.1f}ms")
    print("=" * 60)

    # SLA checks
    sla_pass = True
    if p99 > 50:
        print(f"  FAIL: p99 latency {p99:.1f}ms > 50ms SLA")
        sla_pass = False
    else:
        print(f"  PASS: p99 latency {p99:.1f}ms <= 50ms SLA")

    if failure_rate > 0.1:
        print(f"  FAIL: failure rate {failure_rate:.3f}% > 0.1% SLA")
        sla_pass = False
    else:
        print(f"  PASS: failure rate {failure_rate:.3f}% <= 0.1% SLA")

    if sla_pass:
        print("\n  ALL SLA CHECKS PASSED")
    else:
        print("\n  SLA CHECKS FAILED -- model may not meet production requirements")

    print("=" * 60)
