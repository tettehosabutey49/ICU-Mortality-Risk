"""
SHAP-based global and local explainability for the ICU mortality model.

Provides publication-quality plots and a local explanation function suitable
for embedding in the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Number of features to show in summary/bar plots — beyond ~20 the chart
# becomes unreadable on a standard screen.
TOP_N_FEATURES = 20


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using the tree explainer (fast for gradient boosters).

    TreeExplainer is exact (not approximate) for tree-based models and runs
    in O(T * L) time rather than the exponential exact Shapley calculation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def plot_global_summary(shap_values: shap.Explanation, X: pd.DataFrame, save: bool = True) -> None:
    """Beeswarm plot of the top-N most impactful features across the cohort.

    Beeswarm is preferred over bar chart for global summaries because it
    simultaneously encodes direction (positive vs negative SHAP) and
    feature value magnitude via colour.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.title("SHAP Global Feature Importance — ICU Mortality Model", fontsize=14, pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_beeswarm_global.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_bar_importance(shap_values: shap.Explanation, save: bool = True) -> None:
    """Mean |SHAP| bar chart — useful for a quick ranked feature list."""
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.bar(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save: bool = True,
) -> None:
    """Dependence plot showing how a single feature's SHAP value varies with its magnitude.

    Interaction_feature='auto' lets SHAP pick the strongest interaction
    partner, which often surfaces unexpected clinical relationships.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values[:, interaction_feature] if interaction_feature != "auto" else shap_values,
        show=False,
        ax=ax,
    )
    ax.set_title(f"SHAP Dependence: {feature}", fontsize=13)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    plt.tight_layout()
    if save:
        fname = f"shap_dependence_{feature.replace(' ', '_')}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def explain_patient(
    model,
    X_single: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Return a dict of {feature: shap_value} for a single patient row.

    Used by the Streamlit app to render individual-level explanations at
    inference time without recomputing the full explainer.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_single)
    names = feature_names or X_single.columns.tolist()
    return dict(zip(names, sv.values[0]))
