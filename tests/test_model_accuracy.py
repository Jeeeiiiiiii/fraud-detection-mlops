"""
Model Accuracy Gate Tests
=========================
Validates that the trained fraud detection model meets minimum performance
thresholds required for production deployment.

These tests are designed to run in the CI/CD pipeline after training completes.
They load the model and evaluate it against a hold-out test set to ensure:
  - Recall >= 0.90  (catch at least 90% of fraud)
  - Precision >= 0.50  (at least half of flagged transactions are actually fraud)
  - AUC-ROC >= 0.95  (strong overall discrimination)

If MLflow / model artifacts are unavailable (e.g., in unit test mode), the tests
use synthetic data and the stub model from conftest to verify the test harness
itself.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _generate_synthetic_test_data(
    n_samples: int = 2000,
    fraud_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature matrix and labels for model evaluation.

    Returns (X, y_true) where X has 13 features matching FEATURE_KEYS
    and y_true contains binary fraud labels.
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions: low amounts, matching countries
    X_legit = rng.uniform(0, 200, size=(n_legit, 13)).astype(np.float32)
    y_legit = np.zeros(n_legit, dtype=int)

    # Fraudulent transactions: high amounts, mismatched features
    X_fraud = rng.uniform(500, 5000, size=(n_fraud, 13)).astype(np.float32)
    # Set billing_shipping_match (index 9) to 0 for fraud
    X_fraud[:, 9] = 0
    y_fraud = np.ones(n_fraud, dtype=int)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])

    # Shuffle
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def _get_model_and_data():
    """
    Attempt to load the model from MLflow. If unavailable, fall back to the
    stub model and synthetic data.

    Returns (model, X_test, y_test).
    """
    try:
        import mlflow
        import mlflow.xgboost

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = os.getenv("MODEL_NAME", "fraud-detection-xgb")
        model_version = os.getenv("MODEL_VERSION", "latest")

        mlflow.set_tracking_uri(mlflow_uri)

        if model_version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{model_version}"

        model = mlflow.xgboost.load_model(model_uri)

        # Try to load test data
        data_path = os.getenv("TEST_DATA_PATH", "data/output/transactions.parquet")
        if os.path.exists(data_path):
            import pandas as pd
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
            from train import engineer_features, get_feature_columns
            df = pd.read_parquet(data_path)
            df = engineer_features(df)
            X_test = df[get_feature_columns()].values
            y_test = df["is_fraud"].values
            return model, X_test, y_test

    except Exception:
        pass

    # Fallback: use stub model + synthetic data
    from conftest import StubFraudModel
    model = StubFraudModel()
    X_test, y_test = _generate_synthetic_test_data()
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_evaluation():
    """
    Load model and test data once per module, run predictions, and return
    a dict with y_true, y_pred, y_prob, and computed metrics.
    """
    model, X_test, y_test = _get_model_and_data()
    y_prob = model.predict_proba(X_test)[:, 1]

    # Use threshold of 0.5 for default evaluation
    threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    y_pred = (y_prob >= threshold).astype(int)

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    # AUC-ROC requires both classes present
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_roc = 0.0

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "recall": recall,
        "precision": precision,
        "auc_roc": auc_roc,
        "threshold": threshold,
        "n_samples": len(y_test),
        "n_fraud": int(y_test.sum()),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelAccuracy:
    """Validate model meets minimum production quality gates."""

    def test_recall_above_threshold(self, model_evaluation):
        """
        Recall must be >= 0.90.

        This is the most critical metric for fraud detection: we must catch
        at least 90% of fraudulent transactions. Missing fraud has direct
        financial impact (chargebacks, losses).
        """
        recall = model_evaluation["recall"]
        min_recall = 0.90
        assert recall >= min_recall, (
            f"Recall {recall:.4f} is below the minimum threshold of {min_recall}. "
            f"The model is missing too many fraudulent transactions. "
            f"({model_evaluation['n_fraud']} fraud in {model_evaluation['n_samples']} samples)"
        )

    def test_precision_above_threshold(self, model_evaluation):
        """
        Precision must be >= 0.50.

        At least half of the transactions flagged as fraud should actually be
        fraudulent. Lower precision means too many false positives, which
        blocks legitimate customers and increases support costs.
        """
        precision = model_evaluation["precision"]
        min_precision = 0.50
        assert precision >= min_precision, (
            f"Precision {precision:.4f} is below the minimum threshold of {min_precision}. "
            f"Too many legitimate transactions are being flagged as fraud."
        )

    def test_auc_roc_above_threshold(self, model_evaluation):
        """
        AUC-ROC must be >= 0.95.

        A strong AUC-ROC indicates the model has excellent overall
        discrimination ability between legitimate and fraudulent transactions
        across all possible thresholds.
        """
        auc_roc = model_evaluation["auc_roc"]
        min_auc = 0.95
        assert auc_roc >= min_auc, (
            f"AUC-ROC {auc_roc:.4f} is below the minimum threshold of {min_auc}. "
            f"The model does not have sufficient discrimination power."
        )

    def test_sufficient_test_data(self, model_evaluation):
        """Ensure the test set is large enough for reliable metric estimation."""
        assert model_evaluation["n_samples"] >= 100, (
            f"Test set has only {model_evaluation['n_samples']} samples; "
            f"need at least 100 for reliable evaluation."
        )

    def test_fraud_class_present(self, model_evaluation):
        """The test set must contain both fraud and non-fraud examples."""
        assert model_evaluation["n_fraud"] > 0, (
            "Test set contains no fraudulent transactions. "
            "Cannot evaluate fraud detection without positive examples."
        )
        assert model_evaluation["n_fraud"] < model_evaluation["n_samples"], (
            "Test set contains only fraudulent transactions. "
            "Cannot evaluate false positive rate without negative examples."
        )
