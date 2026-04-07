#!/usr/bin/env python3
"""
Model Evaluation Script
=======================
Comprehensive evaluation of a trained fraud-detection model loaded from
the MLflow Model Registry.

Capabilities
------------
- Per-threshold analysis table (precision, recall, F1 at 20 thresholds).
- Segment analysis by merchant_category, amount_range, and geography.
- Champion-challenger comparison against the current production model.
- JSON report output for downstream CI/CD gating decisions.

Usage
-----
    python evaluate.py \
        --model-name fraud-detection-xgb \
        --model-version 2 \
        --data-path ./data/output/transactions.parquet \
        --mlflow-uri http://mlflow:5000 \
        --output-path ./evaluation_report.json \
        --champion-version 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Reuse the same feature engineering as training
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import engineer_features, get_feature_columns, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Per-threshold analysis
# ---------------------------------------------------------------------------

def per_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Compute precision, recall, F1, and counts at multiple thresholds.

    Returns a list of dicts suitable for tabular display or JSON export.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred)), 4),
            "f1": round(float(f1_score(y_true, y_pred)), 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "flagged_pct": round(float(y_pred.mean()) * 100, 2),
        })

    return rows


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def segment_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate model performance across key business segments:
    - merchant_category
    - amount_range (bucketed)
    - geography (billing_country)
    """
    y_pred = (y_prob >= threshold).astype(int)
    results: dict[str, Any] = {}

    # -- By merchant category --
    cat_results = []
    if "merchant_category" in df.columns:
        for cat in df["merchant_category"].unique():
            mask = df["merchant_category"] == cat
            if mask.sum() < 10:
                continue
            cat_results.append({
                "segment": str(cat),
                "n_samples": int(mask.sum()),
                "fraud_rate": round(float(y_true[mask].mean()), 4),
                "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
            })
    results["by_merchant_category"] = sorted(cat_results, key=lambda x: x["f1"])

    # -- By amount range --
    amount_bins = [0, 10, 50, 200, 1000, 5000, float("inf")]
    amount_labels = ["0-10", "10-50", "50-200", "200-1K", "1K-5K", "5K+"]
    if "amount" in df.columns:
        df_eval = df.copy()
        df_eval["amount_range"] = pd.cut(
            df_eval["amount"], bins=amount_bins, labels=amount_labels, right=False,
        )
        amount_results = []
        for label in amount_labels:
            mask = df_eval["amount_range"] == label
            if mask.sum() < 10:
                continue
            amount_results.append({
                "segment": label,
                "n_samples": int(mask.sum()),
                "fraud_rate": round(float(y_true[mask].mean()), 4),
                "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
            })
        results["by_amount_range"] = amount_results
    else:
        results["by_amount_range"] = []

    # -- By geography (billing country, top 10 by volume) --
    geo_results = []
    if "billing_country" in df.columns:
        top_countries = df["billing_country"].value_counts().head(10).index
        for country in top_countries:
            mask = df["billing_country"] == country
            if mask.sum() < 10:
                continue
            geo_results.append({
                "segment": str(country),
                "n_samples": int(mask.sum()),
                "fraud_rate": round(float(y_true[mask].mean()), 4),
                "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
            })
    results["by_geography"] = geo_results

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_registry(
    model_name: str,
    version: int | str | None = None,
    stage: str | None = None,
) -> Any:
    """
    Load a model from the MLflow Model Registry.

    Specify either a version number or a stage ('Production', 'Staging').
    If neither is given, the latest version is loaded.
    """
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info("Loading model from %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)
    return model


# ---------------------------------------------------------------------------
# Champion-challenger comparison
# ---------------------------------------------------------------------------

def champion_challenger(
    y_true: np.ndarray,
    challenger_prob: np.ndarray,
    champion_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Compare challenger (new) model against champion (current production).

    Returns a comparison dict with metrics for both and a recommendation.
    """
    c_pred = (challenger_prob >= threshold).astype(int)
    p_pred = (champion_prob >= threshold).astype(int)

    def _metrics(y_t, y_p, y_pr):
        return {
            "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
            "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
            "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_t, y_pr)), 4),
            "pr_auc": round(float(average_precision_score(y_t, y_pr)), 4),
        }

    challenger_metrics = _metrics(y_true, c_pred, challenger_prob)
    champion_metrics = _metrics(y_true, p_pred, champion_prob)

    # Recommendation: challenger must beat champion on recall without
    # significant precision degradation (> 5% drop)
    recall_improved = challenger_metrics["recall"] > champion_metrics["recall"]
    precision_acceptable = (
        challenger_metrics["precision"]
        >= champion_metrics["precision"] * 0.95
    )

    recommendation = "PROMOTE" if (recall_improved and precision_acceptable) else "REJECT"

    return {
        "challenger": challenger_metrics,
        "champion": champion_metrics,
        "recommendation": recommendation,
        "recall_delta": round(
            challenger_metrics["recall"] - champion_metrics["recall"], 4
        ),
        "precision_delta": round(
            challenger_metrics["precision"] - champion_metrics["precision"], 4
        ),
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    model_version: int | str,
    data_path: str,
    mlflow_uri: str = "http://localhost:5000",
    output_path: str = "./evaluation_report.json",
    champion_version: int | str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run the full evaluation pipeline and write a JSON report.
    """
    mlflow.set_tracking_uri(mlflow_uri)

    # -- Load data --
    df_raw = load_data(data_path)
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y_true = df["is_fraud"].values

    # -- Load challenger model --
    challenger_model = load_model_from_registry(model_name, version=model_version)
    challenger_prob = challenger_model.predict_proba(X)[:, 1]

    # Determine optimal threshold from model metadata if not provided
    if threshold is None:
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version(model_name, str(model_version))
            run = client.get_run(mv.run_id)
            threshold = float(run.data.params.get("optimal_threshold", 0.5))
        except Exception:
            threshold = 0.5
    logger.info("Using threshold: %.2f", threshold)

    # -- Per-threshold analysis --
    threshold_table = per_threshold_analysis(y_true, challenger_prob)

    # -- Segment analysis --
    segments = segment_analysis(df_raw, y_true, challenger_prob, threshold)

    # -- Champion-challenger (if champion version specified) --
    comparison = None
    if champion_version is not None:
        try:
            champion_model = load_model_from_registry(
                model_name, version=champion_version
            )
            champion_prob = champion_model.predict_proba(X)[:, 1]
            comparison = champion_challenger(
                y_true, challenger_prob, champion_prob, threshold
            )
            logger.info("Champion-challenger comparison: %s", comparison["recommendation"])
        except Exception as exc:
            logger.warning("Could not load champion model v%s: %s", champion_version, exc)
            comparison = {"error": str(exc)}

    # -- Overall metrics --
    y_pred = (challenger_prob >= threshold).astype(int)
    overall = {
        "threshold": float(threshold),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, challenger_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, challenger_prob)), 4),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "fraud_rate": round(float(y_true.mean()), 4),
    }

    # -- Assemble report --
    report: dict[str, Any] = {
        "model_name": model_name,
        "model_version": str(model_version),
        "evaluation_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "overall_metrics": overall,
        "per_threshold_analysis": threshold_table,
        "segment_analysis": segments,
    }
    if comparison is not None:
        report["champion_challenger"] = comparison

    # -- Write report --
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report written to %s", output)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fraud detection model from MLflow registry.",
    )
    parser.add_argument(
        "--model-name", type=str, default="fraud-detection-xgb",
        help="Registered model name in MLflow.",
    )
    parser.add_argument(
        "--model-version", type=str, required=True,
        help="Model version to evaluate (challenger).",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to evaluation dataset (CSV or Parquet).",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--output-path", type=str, default="./evaluation_report.json",
        help="Output path for the JSON evaluation report.",
    )
    parser.add_argument(
        "--champion-version", type=str, default=None,
        help="Champion model version for comparison. Omit to skip comparison.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold. If omitted, read from model metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        model_name=args.model_name,
        model_version=args.model_version,
        data_path=args.data_path,
        mlflow_uri=args.mlflow_uri,
        output_path=args.output_path,
        champion_version=args.champion_version,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
