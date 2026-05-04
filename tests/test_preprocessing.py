"""
Unit tests for src/preprocessing.py.

Focuses on invariants that must hold regardless of input scale:
 - drop_high_missingness correctly identifies the right columns
 - impute leaves no NaN values behind
 - split_data preserves the positive class ratio across all three splits
 - scale_numeric does not leak statistics from val/test into the scaler
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    HIGH_MISSINGNESS_THRESHOLD,
    drop_high_missingness,
    impute,
    scale_numeric,
    split_data,
)


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Minimal synthetic ICU-like DataFrame (50 rows, mix of numeric/categorical)."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n).astype(float),
            "bmi": rng.normal(25, 5, size=n),
            "apache_score": rng.integers(0, 50, size=n).astype(float),
            "ethnicity": rng.choice(["Asian", "White", "Black", "Hispanic"], size=n),
            # column with 80 % missing — should be dropped
            "sparse_lab": np.where(rng.random(n) < 0.80, np.nan, rng.normal(5, 1, n)),
            "hospital_death": rng.integers(0, 2, size=n),
        }
    )


class TestDropHighMissingness:
    def test_drops_sparse_column(self, small_df):
        cleaned, dropped = drop_high_missingness(small_df)
        assert "sparse_lab" in dropped

    def test_retains_dense_columns(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        assert "age" in cleaned.columns
        assert "bmi" in cleaned.columns

    def test_threshold_boundary(self, small_df):
        # At exactly the threshold the column should be retained (strict >)
        small_df["boundary_col"] = np.where(
            np.random.default_rng(1).random(len(small_df)) < HIGH_MISSINGNESS_THRESHOLD,
            np.nan,
            1.0,
        )
        _, dropped = drop_high_missingness(small_df)
        assert "boundary_col" not in dropped


class TestImpute:
    def test_no_nulls_after_impute(self, small_df):
        # Drop the always-sparse column first, then impute
        cleaned, _ = drop_high_missingness(small_df)
        result = impute(cleaned)
        assert result.isnull().sum().sum() == 0

    def test_does_not_modify_original(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        original_nulls = cleaned.isnull().sum().sum()
        _ = impute(cleaned)
        assert cleaned.isnull().sum().sum() == original_nulls  # original unchanged


class TestSplitData:
    def test_split_sizes_sum_to_total(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(filled)

    def test_positive_rate_is_stratified(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        overall_rate = filled["hospital_death"].mean()
        for split_y in [y_train, y_val, y_test]:
            # Allow ±10 percentage points due to small sample size
            assert abs(split_y.mean() - overall_rate) < 0.10


class TestScaleNumeric:
    def test_train_mean_near_zero(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        X_train_s, _, _, _ = scale_numeric(X_train, X_val, X_test)
        num_cols = X_train_s.select_dtypes(include=[np.number]).columns
        assert (X_train_s[num_cols].mean().abs() < 0.1).all()

    def test_scaler_not_fit_on_test(self, small_df):
        """Verify scaler is fit only on train — test stats should differ from 0."""
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        _, _, X_test_s, scaler = scale_numeric(X_train, X_val, X_test)
        # Test mean should NOT be 0 (it wasn't used to fit the scaler)
        num_cols = X_test_s.select_dtypes(include=[np.number]).columns
        means = X_test_s[num_cols].mean().abs()
        # At least one column should have a non-zero mean in test
        assert (means > 0.01).any()
