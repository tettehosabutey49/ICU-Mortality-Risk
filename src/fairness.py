"""
Fairness audit for the ICU mortality risk model.

Evaluates demographic parity and equalized odds across protected attributes
(gender, ethnicity, age group) using Fairlearn's MetricFrame.

Clinical context: algorithmic bias in severity scoring has been documented
in critical care literature (Sjoding et al., NEJM 2020; Obermeyer et al.,
Science 2019).  This module ensures we can quantify and report any disparities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, roc_auc_score

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds based on NIST SP 1270 / EEOC 4/5ths rule guidance.
# A demographic parity difference > 0.10 typically warrants investigation.
DISPARITY_FLAG_THRESHOLD = 0.10


def run_metric_frame(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> MetricFrame:
    """Build a Fairlearn MetricFrame across all protected attributes simultaneously.

    MetricFrame disaggregates each metric by every combination of sensitive
    feature values, making intersectional bias visible (e.g. elderly women).
    """
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return mf


def summarise_disparities(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_col: pd.Series,
    col_name: str = "group",
) -> pd.DataFrame:
    """Compute demographic parity and equalized odds differences for one attribute.

    Returns a one-row DataFrame with named disparity metrics, making it easy
    to concatenate results across multiple protected attributes into a report.
    """
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_col)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_col)

    flag = "FLAG" if max(abs(dp_diff), abs(eo_diff)) > DISPARITY_FLAG_THRESHOLD else "OK"

    return pd.DataFrame(
        [{
            "attribute": col_name,
            "demographic_parity_diff": round(dp_diff, 4),
            "equalized_odds_diff": round(eo_diff, 4),
            "status": flag,
        }]
    )


def plot_metric_by_group(
    mf: MetricFrame,
    metric: str = "false_negative_rate",
    attribute: str = "ethnicity",
    save: bool = True,
) -> None:
    """Bar chart of a chosen metric disaggregated by a protected attribute.

    False negative rate is highlighted by default — in clinical settings, a
    high FNR for a subgroup means sicker patients in that group are more
    likely to be undertriaged.
    """
    by_group = mf.by_group[metric]
    fig, ax = plt.subplots(figsize=(10, 5))
    by_group.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.axhline(mf.overall[metric], color="crimson", linestyle="--", linewidth=1.5, label="Overall")
    ax.set_title(f"{metric.replace('_', ' ').title()} by {attribute.title()}", fontsize=13)
    ax.set_xlabel(attribute.title(), fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"fairness_{metric}_by_{attribute}.png", dpi=150, bbox_inches="tight")
    plt.show()
