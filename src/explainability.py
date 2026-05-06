"""
SHAP-based global and local explainability for the ICU mortality model.

Provides publication-quality plots and a local explanation function suitable
for embedding in the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Number of features shown in summary/bar plots — beyond ~25 the chart becomes unreadable.
TOP_N_FEATURES = 25


def compute_shap_values(model: Any, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using the TreeExplainer (exact for gradient-boosted trees).

    TreeExplainer is O(T × L) per sample — orders of magnitude faster than the
    kernel explainer while remaining exact.  For binary classification LightGBM/XGBoost
    models the result is a single-output Explanation for the positive class (mortality).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    # Some SHAP versions return shape (n, p, 2) for binary classifiers — extract class 1.
    if isinstance(sv, shap.Explanation) and sv.values.ndim == 3:
        sv = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=sv.data,
            feature_names=sv.feature_names,
        )
    return sv


def plot_global_summary(
    shap_values: shap.Explanation,
    clinical_notes: dict[str, str] | None = None,
    max_display: int = TOP_N_FEATURES,
    save: bool = True,
) -> None:
    """Beeswarm plot of the most impactful features with optional clinical annotations.

    Beeswarm is preferred over bar chart for global summaries because it encodes
    both the direction of each feature's effect (positive vs. negative SHAP) and
    the feature value magnitude via colour — all in a single panel.

    clinical_notes: mapping from feature name to a short clinical interpretation
        shown in an annotation box beside the top 5 features.
    """
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    ax = fig.axes[0]
    fig.set_size_inches(13, 9)

    if clinical_notes:
        mean_abs = pd.Series(
            np.abs(shap_values.values).mean(axis=0),
            index=shap_values.feature_names,
        ).sort_values(ascending=False)
        top5 = mean_abs.head(5).index.tolist()
        lines = [f"{i+1}. {f}:\n   {clinical_notes.get(f, 'high-impact predictor')}"
                 for i, f in enumerate(top5)]
        note = "Top 5 Clinical Notes:\n\n" + "\n".join(lines)
        fig.text(
            0.68, 0.96, note,
            fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      alpha=0.90, edgecolor="grey"),
            fontfamily="monospace",
        )

    ax.set_title(
        "SHAP Global Feature Importance — ICU Mortality Model\n"
        "Red = high feature value  ·  Blue = low feature value",
        fontsize=12, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_beeswarm_global.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_bar_importance(
    shap_values: shap.Explanation,
    max_display: int = TOP_N_FEATURES,
    save: bool = True,
) -> None:
    """Mean |SHAP| bar chart — ranked feature importance for quick reference."""
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title("Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str,
    interaction_feature: str | None = None,
    ax: plt.Axes | None = None,
    save: bool = True,
) -> None:
    """Scatter plot of one feature's SHAP value vs. its magnitude.

    Colours each point by the strongest interaction partner when
    interaction_feature is None (auto-detected by SHAP).  This reveals
    non-linear effects and synergistic interactions between features.
    """
    show = ax is None
    if ax is not None:
        color_arg = shap_values[:, interaction_feature] if interaction_feature else shap_values
        shap.plots.scatter(shap_values[:, feature], color=color_arg, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {feature}", fontsize=10, fontweight="bold")
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel(f"SHAP for {feature}", fontsize=9)
    else:
        color_arg = shap_values[:, interaction_feature] if interaction_feature else shap_values
        shap.plots.scatter(shap_values[:, feature], color=color_arg, show=False)
        plt.title(f"SHAP Dependence: {feature}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            fname = f"shap_dependence_{feature.replace(' ', '_').replace('/', '_')}.png"
            plt.gcf().savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
        plt.show()


def plot_waterfall(
    shap_values: shap.Explanation,
    patient_idx: int,
    title: str = "",
    max_display: int = 15,
    save_path: str | Path | None = None,
) -> None:
    """Waterfall plot for a single patient — shows cumulative SHAP contributions.

    Each bar shows how one feature pushes the prediction above or below the
    expected value (baseline average log-odds), ending at the patient's
    final predicted log-odds.  This is the canonical individual explanation
    format for clinical case notes.
    """
    shap.plots.waterfall(shap_values[patient_idx], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_shap_vs_native(
    shap_values: shap.Explanation,
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """Side-by-side comparison of SHAP mean |SHAP| vs. XGBoost gain-based importance.

    Returns a DataFrame with both rankings so callers can analyse agreement and
    divergence between the two importance measures.
    """
    shap_imp = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    native_imp = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    # Normalise both to [0, 1] for visual comparison
    shap_norm   = shap_imp   / shap_imp.max()
    native_norm = native_imp / native_imp.max()

    merged = pd.DataFrame({
        "shap_rank":   range(1, len(shap_imp) + 1),
        "shap_norm":   shap_norm.values,
        "native_norm": native_norm.reindex(shap_imp.index).fillna(0).values,
    }, index=shap_imp.index)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    for ax, col, label, color in [
        (axes[0], "shap_norm",   "SHAP Mean |SHAP| (normalised)", "#4878d0"),
        (axes[1], "native_norm", "XGBoost Gain Importance (normalised)", "#ee854a"),
    ]:
        ax.barh(merged.index, merged[col], color=color, alpha=0.85, edgecolor="white")
        ax.invert_yaxis()
        ax.set_xlabel(label, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Feature", fontsize=10)
    fig.suptitle(
        "SHAP vs. XGBoost Native Importance\n"
        "Disagreements reveal features whose interactions SHAP captures but gain misses",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_vs_native_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    return merged


def explain_patient(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Return {feature: shap_value} for a single patient row.

    Used by the Streamlit dashboard at inference time without recomputing
    SHAP values for the entire dataset.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_single)
    if sv.values.ndim == 3:
        values = sv.values[0, :, 1]
    else:
        values = sv.values[0]
    names = feature_names or X_single.columns.tolist()
    return dict(zip(names, values.tolist()))
