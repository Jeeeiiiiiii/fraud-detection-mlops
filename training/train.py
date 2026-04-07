#!/usr/bin/env python3
"""
XGBoost Fraud Classifier -- Training Script
============================================
End-to-end training pipeline for the fraud detection model:

1. Load and validate data (CSV or Parquet).
2. Feature engineering and preprocessing.
3. Handle severe class imbalance with SMOTE + scale_pos_weight.
4. Bayesian hyperparameter optimisation via Optuna (default 50 trials).
5. Train final model with the best hyperparameters.
6. Log everything to MLflow: parameters, metrics, plots, and model artefact.
7. Register the model in the MLflow Model Registry.

Optimisation objective: maximise **recall** while maintaining precision >= 0.5.

Usage
-----
    python train.py \
        --data-path ./data/output/transactions.parquet \
        --mlflow-uri http://mlflow:5000 \
        --n-trials 50 \
        --experiment-name fraud-detection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# Silence Optuna's verbose trial logging unless DEBUG
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "merchant_category", "card_type", "device_type",
    "ip_country", "billing_country", "shipping_country",
]

NUMERIC_FEATURES = [
    "amount", "transaction_hour", "transaction_day_of_week",
    "billing_shipping_match", "ip_billing_match", "ip_shipping_match",
    "amount_log",
]

_label_encoders: dict[str, LabelEncoder] = {}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw transaction columns.

    This function is deterministic and can be reused at serving time.
    """
    df = df.copy()

    # Temporal features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["transaction_hour"] = df["timestamp"].dt.hour
        df["transaction_day_of_week"] = df["timestamp"].dt.dayofweek
    else:
        df.setdefault("transaction_hour", 12)
        df.setdefault("transaction_day_of_week", 3)

    # Geographic match flags
    df["billing_shipping_match"] = (
        df["billing_country"] == df["shipping_country"]
    ).astype(int)
    df["ip_billing_match"] = (
        df["ip_country"] == df["billing_country"]
    ).astype(int)
    df["ip_shipping_match"] = (
        df["ip_country"] == df["shipping_country"]
    ).astype(int)

    # Log-transform amount to reduce skew
    df["amount_log"] = np.log1p(df["amount"])

    # Label-encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        le = _label_encoders.get(col)
        if le is None:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            _label_encoders[col] = le
        else:
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, _k=known, _le=le: (
                    _le.transform([x])[0] if x in _k else len(_le.classes_)
                )
            )

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used by the model."""
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV or Parquet transaction data."""
    path = Path(data_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    logger.info("Loaded %d transactions from %s", len(df), data_path)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Engineer features, split into train/test, and apply SMOTE to the
    training set to address class imbalance.

    Returns (X_train, X_test, y_train, y_test, feature_names).
    """
    df = engineer_features(df)
    feature_cols = get_feature_columns()

    X = df[feature_cols].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    logger.info(
        "Train: %d samples (%d fraud, %.2f%%), Test: %d samples (%d fraud, %.2f%%)",
        len(y_train), y_train.sum(), y_train.mean() * 100,
        len(y_test), y_test.sum(), y_test.mean() * 100,
    )

    # SMOTE on training set only to avoid data leakage
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state, sampling_strategy=0.3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE: %d samples (%d fraud, %.2f%%)",
            len(y_train), y_train.sum(), y_train.mean() * 100,
        )
    except ImportError:
        logger.warning(
            "imbalanced-learn not installed -- skipping SMOTE. "
            "Install with: pip install imbalanced-learn"
        )

    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_folds: int = 5,
):
    """
    Return an Optuna objective that performs stratified k-fold CV and
    maximises recall while keeping precision above 0.5.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        recalls = []
        precisions = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )

            y_pred = model.predict(X_v)
            recalls.append(recall_score(y_v, y_pred))
            precisions.append(precision_score(y_v, y_pred, zero_division=0))

        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)

        # Penalise if precision drops below 0.5 (too many false positives)
        if mean_precision < 0.5:
            return mean_recall * 0.5

        return mean_recall

    return objective


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(y_true, y_prob, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_precision_recall(y_true, y_prob, path: str) -> None:
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_arr, precision_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_arr, precision_arr, label=f"AUC-PR = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, feature_names: list[str], path: str) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(feature_names)),
        importances[indices[::-1]],
        align="center",
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    mlflow_uri: str = "http://localhost:5000",
    n_trials: int = 50,
    experiment_name: str = "fraud-detection",
    register_model: bool = True,
) -> dict[str, Any]:
    """
    Execute the full training pipeline and return a metrics dictionary.

    Steps:
      1. Load + prepare data.
      2. Run Optuna hyperparameter search (maximise recall).
      3. Train final model with best params.
      4. Evaluate on hold-out test set.
      5. Log everything to MLflow and register the model.
    """
    # -- MLflow setup --
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # -- Load data --
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # -- Scale pos weight (ratio of negatives to positives) --
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.2f", scale_pos_weight)

    # -- Optuna hyperparameter search --
    logger.info("Starting Optuna study with %d trials ...", n_trials)
    study = optuna.create_study(direction="maximize", study_name="fraud-xgb")
    study.optimize(
        create_objective(X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
    })
    logger.info("Best params: %s", json.dumps(best_params, indent=2))
    logger.info("Best trial recall: %.4f", study.best_value)

    # -- Train final model --
    with mlflow.start_run(run_name="fraud-xgb-final") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("n_train_samples", len(y_train))
        mlflow.log_param("n_test_samples", len(y_test))

        final_model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            verbosity=0,
        )
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # -- Evaluate --
        y_prob = final_model.predict_proba(X_test)[:, 1]
        # Use a threshold tuned for high recall
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1_recall = 0.0
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            r = recall_score(y_test, y_pred_t)
            p = precision_score(y_test, y_pred_t, zero_division=0)
            f1 = f1_score(y_test, y_pred_t)
            # Pick threshold that maximises recall while keeping precision >= 0.5
            score = r if p >= 0.5 else r * 0.5
            if score > best_f1_recall:
                best_f1_recall = score
                best_threshold = t

        y_pred = (y_prob >= best_threshold).astype(int)

        metrics = {
            "threshold": float(best_threshold),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
        }
        logger.info("Test metrics: %s", json.dumps(metrics, indent=2))

        mlflow.log_metrics(metrics)
        mlflow.log_param("optimal_threshold", best_threshold)

        # -- Plots --
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            pr_path = os.path.join(tmpdir, "precision_recall_curve.png")
            fi_path = os.path.join(tmpdir, "feature_importance.png")

            _plot_confusion_matrix(y_test, y_pred, cm_path)
            _plot_roc_curve(y_test, y_prob, roc_path)
            _plot_precision_recall(y_test, y_prob, pr_path)
            _plot_feature_importance(final_model, feature_names, fi_path)

            mlflow.log_artifact(cm_path, "plots")
            mlflow.log_artifact(roc_path, "plots")
            mlflow.log_artifact(pr_path, "plots")
            mlflow.log_artifact(fi_path, "plots")

        # -- Log classification report --
        report = classification_report(y_test, y_pred, target_names=["legit", "fraud"])
        mlflow.log_text(report, "classification_report.txt")
        logger.info("Classification report:\n%s", report)

        # -- Save model --
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="fraud-detection-xgb" if register_model else None,
        )
        logger.info("Model logged to MLflow run %s", run.info.run_id)

        # -- Save label encoders and feature names for serving --
        model_meta = {
            "feature_names": feature_names,
            "optimal_threshold": best_threshold,
            "label_encoders": {
                col: le.classes_.tolist()
                for col, le in _label_encoders.items()
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(model_meta, f, indent=2)
            mlflow.log_artifact(f.name, "model_meta")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost fraud detection model.",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to transactions CSV or Parquet file.",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="http://localhost:5000",
        help="MLflow tracking server URI (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna hyperparameter search trials (default: 50).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="fraud-detection",
        help="MLflow experiment name (default: fraud-detection).",
    )
    parser.add_argument(
        "--no-register", action="store_true",
        help="Skip model registration in MLflow Model Registry.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(
        data_path=args.data_path,
        mlflow_uri=args.mlflow_uri,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
        register_model=not args.no_register,
    )


if __name__ == "__main__":
    main()
