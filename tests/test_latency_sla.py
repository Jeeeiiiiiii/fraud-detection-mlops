"""
Latency SLA Tests
=================
Validates that the fraud scoring service meets latency requirements:
  - Single prediction:    < 50ms  (p99 SLA)
  - Batch prediction:     < 200ms for 50 transactions
  - Concurrent requests:  < 50ms p99 under 10 concurrent users

These tests exercise the actual FastAPI endpoints using the TestClient,
with the model stubbed via conftest fixtures.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scoring_payload(
    transaction_id: str = "txn_latency_001",
    amount: float = 100.0,
) -> dict[str, Any]:
    """Create a minimal scoring request payload."""
    return {
        "transaction_id": transaction_id,
        "user_id": "user_latency_test",
        "amount": amount,
        "currency": "USD",
        "merchant_id": "merch_001",
        "merchant_category": "grocery",
        "card_type": "visa",
        "device_type": "mobile",
        "ip_country": "US",
        "billing_country": "US",
        "shipping_country": "US",
        "timestamp": "2025-06-15T14:00:00Z",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSinglePredictionLatency:
    """Verify single-transaction scoring latency is within SLA."""

    def test_single_prediction_under_50ms(self, scoring_client):
        """
        A single /score request must complete in under 50ms.

        We run 20 warm-up requests to eliminate cold-start effects, then
        measure 100 requests and assert the p99 is below 50ms.
        """
        payload = _make_scoring_payload()

        # Warm-up: prime the model and any caches
        for _ in range(20):
            scoring_client.post("/score", json=payload)

        # Measurement phase
        latencies = []
        n_requests = 100
        for i in range(n_requests):
            p = _make_scoring_payload(transaction_id=f"txn_lat_single_{i:04d}")
            start = time.perf_counter()
            response = scoring_client.post("/score", json=p)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert response.status_code == 200

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(0.50 * n_requests)]
        p95 = latencies_sorted[int(0.95 * n_requests)]
        p99 = latencies_sorted[int(0.99 * n_requests)]
        avg = sum(latencies) / len(latencies)

        print(f"\nSingle prediction latency (n={n_requests}):")
        print(f"  avg: {avg:.2f}ms")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # Note: TestClient runs in-process (no network), so latencies will
        # be lower than production. We use 50ms as the ceiling; actual
        # production SLA verification happens in the load test (Locust).
        assert p99 < 50.0, (
            f"Single prediction p99 latency {p99:.2f}ms exceeds 50ms SLA. "
            f"Avg: {avg:.2f}ms, p95: {p95:.2f}ms"
        )


class TestBatchPredictionLatency:
    """Verify batch scoring latency scales linearly."""

    def test_batch_prediction_latency(self, scoring_client):
        """
        Scoring a batch of 50 transactions should complete in under 200ms.

        Batch inference is vectorised (numpy), so it should be significantly
        faster per-transaction than individual requests.
        """
        batch = [
            _make_scoring_payload(
                transaction_id=f"txn_batch_{i:04d}",
                amount=float(10 + i * 5),
            )
            for i in range(50)
        ]

        # Warm-up
        scoring_client.post("/score/batch", json=batch[:5])

        # Measurement
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            response = scoring_client.post("/score/batch", json=batch)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 50

        avg = sum(latencies) / len(latencies)
        p99 = sorted(latencies)[int(0.99 * len(latencies))]

        print(f"\nBatch prediction latency (50 txns, n=10 runs):")
        print(f"  avg: {avg:.2f}ms")
        print(f"  p99: {p99:.2f}ms")
        print(f"  per-txn avg: {avg / 50:.2f}ms")

        assert p99 < 200.0, (
            f"Batch prediction (50 txns) p99 latency {p99:.2f}ms exceeds 200ms. "
            f"Avg: {avg:.2f}ms"
        )


class TestConcurrentRequestsLatency:
    """Verify scoring latency under concurrent load."""

    def test_concurrent_requests_latency(self, scoring_client):
        """
        Under 10 concurrent users, p99 latency should remain below 50ms.

        Uses a thread pool to simulate concurrent callers. Each thread sends
        10 sequential requests and records latencies.
        """
        n_workers = 10
        requests_per_worker = 10
        all_latencies: list[float] = []

        def _worker(worker_id: int) -> list[float]:
            latencies = []
            for i in range(requests_per_worker):
                payload = _make_scoring_payload(
                    transaction_id=f"txn_conc_{worker_id}_{i:03d}",
                )
                start = time.perf_counter()
                response = scoring_client.post("/score", json=payload)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
                assert response.status_code == 200
            return latencies

        # Warm-up
        for _ in range(10):
            scoring_client.post("/score", json=_make_scoring_payload())

        # Concurrent execution
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker, wid) for wid in range(n_workers)]
            for future in as_completed(futures):
                all_latencies.extend(future.result())

        total_requests = len(all_latencies)
        all_latencies_sorted = sorted(all_latencies)
        avg = sum(all_latencies) / total_requests
        p50 = all_latencies_sorted[int(0.50 * total_requests)]
        p95 = all_latencies_sorted[int(0.95 * total_requests)]
        p99 = all_latencies_sorted[int(0.99 * total_requests)]

        print(f"\nConcurrent latency ({n_workers} workers, {total_requests} total):")
        print(f"  avg: {avg:.2f}ms")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        assert p99 < 50.0, (
            f"Concurrent p99 latency {p99:.2f}ms exceeds 50ms SLA "
            f"under {n_workers} concurrent workers. "
            f"Avg: {avg:.2f}ms, p95: {p95:.2f}ms"
        )
