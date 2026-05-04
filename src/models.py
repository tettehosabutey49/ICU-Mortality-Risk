"""
Model training, evaluation, and persistence for ICU mortality prediction.

Provides a unified interface over LightGBM, XGBoost, and a logistic regression
baseline so that notebooks can swap models with a single argument change.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb

# --------------------------------------------------------------------------- #
# Default hyperparameters — tuned for the class-imbalanced ICU dataset         #
# --------------------------------------------------------------------------- #

# scale_pos_weight ≈ (N_negative / N_positive) ≈ 11 for ~8 % mortality rate.
# Computed dynamically in get_model() to adapt to any split.
LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "aucpr",  # AUC-PR is more informative than AUC-ROC under imbalance
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

LR_DEFAULTS: dict[str, Any] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}

MODELS_DIR = Path("models")


def get_model(name: str, y_train: pd.Series | None = None) -> Any:
    """Return an untrained model instance by name.

    *y_train* is used to compute scale_pos_weight for tree models so the
    class-imbalance correction is data-driven rather than hard-coded.

    Args:
        name: One of 'lgbm', 'xgb', 'lr'.
        y_train: Training labels (needed for imbalance weight calculation).
    """
    if y_train is not None:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        spw = neg / pos
    else:
        spw = 11.0  # sensible default for ~8 % mortality

    if name == "lgbm":
        return lgb.LGBMClassifier(**{**LGBM_DEFAULTS, "scale_pos_weight": spw})
    if name == "xgb":
        return xgb.XGBClassifier(**{**XGB_DEFAULTS, "scale_pos_weight": spw})
    if name == "lr":
        return LogisticRegression(**LR_DEFAULTS)
    raise ValueError(f"Unknown model name '{name}'. Choose from: lgbm, xgb, lr.")


def train(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 50,
) -> Any:
    """Fit model with early stopping on the validation set.

    Early stopping prevents overfitting without manual epoch tuning — the
    model stops when validation performance plateaus for *early_stopping_rounds*
    consecutive trees.
    """
    fit_kwargs: dict[str, Any] = {}
    if isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            if isinstance(model, lgb.LGBMClassifier)
            else None,
        }
        if isinstance(model, xgb.XGBClassifier):
            fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]

    model.fit(X_train, y_train, **{k: v for k, v in fit_kwargs.items() if v is not None})
    return model


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series, split_label: str = "test") -> dict[str, float]:
    """Compute a clinical-quality evaluation suite for the given split.

    Metrics chosen for clinical relevance:
    - AUC-ROC: overall discrimination across all thresholds
    - AUC-PR: more informative than AUC-ROC under class imbalance
    - Brier score: proper scoring rule measuring calibration
    """
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "split": split_label,
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
    }

    print(f"\n=== {split_label.upper()} ===")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k:>10}: {v:.4f}")
    print(classification_report(y, pred, target_names=["survived", "died"]))

    return metrics


def save_model(model: Any, name: str) -> Path:
    """Persist a fitted model to models/<name>.pkl."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> Any:
    """Load a previously saved model from models/<name>.pkl."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
