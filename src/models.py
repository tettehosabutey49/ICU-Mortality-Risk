"""
Model training, evaluation, and persistence for ICU mortality prediction.

Provides a unified interface over LightGBM, XGBoost, Random Forest, and
Logistic Regression so notebooks can swap models with a single argument change.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
import xgboost as xgb

# --------------------------------------------------------------------------- #
# Default hyperparameters — named constants so every choice is auditable        #
# --------------------------------------------------------------------------- #

# scale_pos_weight ≈ N_negative/N_positive ≈ 11 for ~8% mortality.
# Computed dynamically in get_model() from actual y_train to adapt to any split.
LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
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
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",    # AUC-PR more informative than AUC-ROC under imbalance
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",     # faster on CPU; hist = approximate quantile sketching
}

RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 30,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

LR_DEFAULTS: dict[str, Any] = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}

MODELS_DIR = Path("models")

# RandomizedSearch parameter distributions — one dict per model family.
# Ranges chosen to span a wide-but-plausible space without exhaustive enumeration.
PARAM_GRIDS: dict[str, dict] = {
    "lr": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    },
    "rf": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [10, 20, 30, 50],
        "max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    },
    "lgbm": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.8, 0.9],
    },
}


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def get_model(name: str, y_train: pd.Series | None = None) -> Any:
    """Return an untrained model instance by name.

    scale_pos_weight for tree models is computed from y_train when provided,
    making the class-imbalance correction data-driven rather than hard-coded.

    Args:
        name: One of 'lgbm', 'xgb', 'rf', 'lr'.
        y_train: Training labels for imbalance weight computation.
    """
    if y_train is not None:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        spw = float(neg / pos)
    else:
        spw = 11.0  # sensible default for ~8% mortality

    if name == "lgbm":
        return lgb.LGBMClassifier(**{**LGBM_DEFAULTS, "scale_pos_weight": spw})
    if name == "xgb":
        return xgb.XGBClassifier(**{**XGB_DEFAULTS, "scale_pos_weight": spw})
    if name == "rf":
        return RandomForestClassifier(**RF_DEFAULTS)
    if name == "lr":
        return LogisticRegression(**LR_DEFAULTS)
    raise ValueError(f"Unknown model '{name}'. Choose from: lgbm, xgb, rf, lr.")


def tune(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Any:
    """Hyperparameter-tune a model using RandomizedSearchCV.

    Why RandomizedSearch over GridSearch:
    With 5 hyperparameters × 4 values each, GridSearch requires 4^5 = 1 024 fits
    per fold. RandomizedSearchCV samples n_iter=20 random points, reducing compute
    by ~98% with negligible performance loss — Bergstra & Bengio (2012) showed that
    random search finds equally good solutions because only a few hyperparameters
    have meaningful marginal impact.

    Scored on average_precision (AUC-PR) rather than accuracy because AUC-PR
    correctly reflects the precision-recall trade-off under class imbalance.
    """
    base_model = get_model(name, y_train=y_train)
    param_grid = PARAM_GRIDS[name]

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV AUC-PR ({name}): {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_


def train(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_rounds: int = 50,
) -> Any:
    """Fit a model, using early stopping for tree models when a validation set is provided.

    Early stopping halts training when validation AUC-PR stops improving for
    early_stopping_rounds consecutive trees, preventing overfitting without
    manual tuning of n_estimators.  LR and RF are fit without early stopping.
    """
    if isinstance(model, lgb.LGBMClassifier) and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
    elif isinstance(model, xgb.XGBClassifier) and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                xgb.callback.EarlyStopping(rounds=early_stopping_rounds, metric_name="aucpr"),
            ],
        )
    else:
        model.fit(X_train, y_train)
    return model


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_label: str = "val",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full evaluation suite covering discrimination, calibration, and classification.

    Metrics:
    - roc_auc: discrimination across all thresholds (biased towards majority class under imbalance)
    - pr_auc: precision-recall AUC — the primary metric for imbalanced clinical data
    - f1 / precision / recall: threshold-dependent classification metrics
    - brier: proper scoring rule measuring calibration quality

    Returns a flat dict so results from multiple models can be concatenated into a
    comparison DataFrame without reshaping.
    """
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return {
        "split":     split_label,
        "roc_auc":   round(float(roc_auc_score(y, prob)), 4),
        "pr_auc":    round(float(average_precision_score(y, prob)), 4),
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "brier":     round(float(brier_score_loss(y, prob)), 4),
    }


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
