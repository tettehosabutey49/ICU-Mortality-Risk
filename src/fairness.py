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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def compute_subgroup_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_col: pd.Series,
) -> pd.DataFrame:
    """Per-subgroup accuracy, precision, recall, F1, and AUC-ROC.

    Groups with fewer than 10 samples or no positive cases are excluded because
    AUC-ROC is undefined for single-class subgroups (and estimates would be
    unreliable at very small n).
    """
    rows = []
    for grp in sorted(sensitive_col.dropna().unique()):
        mask = sensitive_col == grp
        yt = y_true[mask].values
        yp = y_pred[mask]
        ypr = y_prob[mask]
        if mask.sum() < 10 or len(np.unique(yt)) < 2:
            continue
        rows.append({
            "group": str(grp),
            "n": int(mask.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "auc_roc": roc_auc_score(yt, ypr),
        })
    return pd.DataFrame(rows).set_index("group")


def plot_grouped_bar_charts(
    metrics_df: pd.DataFrame,
    sensitive_name: str,
    save: bool = True,
) -> None:
    """Five bar charts (accuracy/precision/recall/F1/AUC-ROC), one per metric.

    Each chart annotates the gap between best and worst group in red when > 0.05
    (clinically meaningful disparity) and green otherwise.
    """
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    palette = plt.cm.Set2.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for i, (metric, title) in enumerate(zip(metric_cols, titles)):
        if metric not in metrics_df.columns:
            continue
        ax = axes_flat[i]
        vals = metrics_df[metric]
        colors = [palette[j % len(palette)] for j in range(len(vals))]
        ax.bar(vals.index, vals.values, color=colors, edgecolor="white", alpha=0.88)

        gap = vals.max() - vals.min()
        gap_color = "darkred" if gap > 0.05 else "darkgreen"
        ax.annotate(
            f"Gap: {gap:.3f}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            color=gap_color, fontweight="bold",
        )
        ax.set_title(f"{title} by {sensitive_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel(title, fontsize=9)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, min(1.05, vals.max() * 1.22))

    axes_flat[5].set_visible(False)
    fig.suptitle(
        f"Section 1 — Performance Disparities by {sensitive_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fname = f"fairness_perf_disparity_{sensitive_name.lower().replace(' ', '_')}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def bootstrap_group_diff(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_col: pd.Series,
    metric: str = "recall",
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CI for each group's metric difference vs overall.

    A CI that excludes zero is treated as statistically significant — i.e., the
    group's performance differs from the population in a way unlikely to be
    explained by sampling variance alone.
    """
    _metric_fns: dict[str, Any] = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    }
    fn = _metric_fns.get(metric, _metric_fns["recall"])
    rng = np.random.default_rng(random_state)

    idx = np.arange(len(y_true))
    y_true_arr = np.asarray(y_true)
    groups = sensitive_col.dropna().unique()
    diffs: dict[str, list[float]] = {str(g): [] for g in groups}

    for _ in range(n_bootstrap):
        boot = rng.choice(idx, size=len(idx), replace=True)
        yt_b = y_true_arr[boot]
        yp_b = y_pred[boot]
        sf_b = sensitive_col.iloc[boot].values
        overall = fn(yt_b, yp_b)
        for g in groups:
            mask = sf_b == g
            if mask.sum() < 5 or yt_b[mask].sum() == 0:
                diffs[str(g)].append(0.0)
            else:
                diffs[str(g)].append(fn(yt_b[mask], yp_b[mask]) - overall)

    rows = []
    for g, vals in diffs.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({
            "group": g,
            "mean_diff": round(float(np.mean(vals)), 4),
            "ci_lower": round(float(lo), 4),
            "ci_upper": round(float(hi), 4),
            "significant": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows).set_index("group")


def plot_metricframe_heatmap(
    mf: MetricFrame,
    save: bool = True,
    fname: str = "fairness_metricframe_heatmap.png",
) -> None:
    """Heatmap of Fairlearn MetricFrame by_group results.

    Diverging red-green palette: high FNR/FPR are red (bad), high accuracy is green.
    Annotated with the numeric value so readers don't need to decode the colour scale.
    """
    import seaborn as sns  # local import keeps the top-level optional

    data = mf.by_group.astype(float)
    fig, ax = plt.subplots(figsize=(max(8, len(data.columns) * 2.2), max(5, len(data) * 0.65)))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="RdYlGn_r",
        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(
        "Fairlearn MetricFrame — Performance by Demographic Group",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def run_threshold_optimizer(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_test: pd.Series,
    constraint: str = "equalized_odds",
) -> dict:
    """Post-process a trained model with Fairlearn's ThresholdOptimizer.

    ThresholdOptimizer learns group-specific classification thresholds that
    minimise the equalized-odds constraint violation.  It does NOT retrain the
    base model — it only adjusts the decision boundary per group, so the base
    model's learned parameters are unchanged.

    Returns a dict with before/after fairness metrics and the fitted optimizer
    so callers can generate mitigated predictions on new data.
    """
    from fairlearn.postprocessing import ThresholdOptimizer  # optional dep

    opt = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        objective="accuracy_score",
        predict_method="predict_proba",
    )
    opt.fit(X_train, y_train, sensitive_features=sensitive_train)

    y_pred_base = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    y_pred_mitigated = opt.predict(X_test, sensitive_features=sensitive_test)

    before_eod = equalized_odds_difference(y_test, y_pred_base, sensitive_features=sensitive_test)
    after_eod = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_test)

    return {
        "before": {
            "equalized_odds_diff": round(float(before_eod), 4),
            "f1": round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_base, zero_division=0)), 4),
        },
        "after": {
            "equalized_odds_diff": round(float(after_eod), 4),
            "f1": round(float(f1_score(y_test, y_pred_mitigated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_mitigated, zero_division=0)), 4),
        },
        "mitigated_predictions": y_pred_mitigated,
        "optimizer": opt,
    }
