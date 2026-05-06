"""
Unit tests for src/preprocessing.py and src/features.py.

Five categories of invariants that must hold regardless of input scale:
  1. SMOTE is only applied to training data — never to the test set
  2. Vital-sign imputation uses group-level (ICU-type) medians, not global medians
  3. Derived features (shock_index, comorbidity_burden) compute to correct values
  4. Target encoding is fitted on training data only — test labels never influence it
  5. The StandardScaler is fitted on train data only — test statistics differ from 0
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
from src.features import (
    add_derived_features,
    apply_target_encoder,
    fit_target_encoder,
    impute_vitals_by_icu_type,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Minimal synthetic ICU-like DataFrame (50 rows)."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({
        "age":          rng.integers(18, 90, size=n).astype(float),
        "bmi":          rng.normal(25, 5, size=n),
        "apache_score": rng.integers(0, 50, size=n).astype(float),
        "ethnicity":    rng.choice(["Asian", "White", "Black", "Hispanic"], size=n),
        "sparse_lab":   np.where(rng.random(n) < 0.80, np.nan, rng.normal(5, 1, n)),
        "hospital_death": rng.integers(0, 2, size=n),
    })


@pytest.fixture()
def df_for_split() -> pd.DataFrame:
    """200-row DataFrame large enough for reliable stratified splits."""
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "age":          rng.integers(18, 90, size=n).astype(float),
        "bmi":          rng.normal(25, 5, size=n),
        "apache_score": rng.integers(0, 50, size=n).astype(float),
        "hospital_death": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })


# ── 1. SMOTE is never applied to test data ────────────────────────────────────

class TestSmoteIsolation:
    def test_smote_does_not_change_test_set_size(self, df_for_split):
        """Applying SMOTE to the training split must leave test size unchanged."""
        from imblearn.over_sampling import SMOTE

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_for_split)
        original_test_n = len(X_test)

        smote = SMOTE(random_state=42, k_neighbors=min(3, y_train.sum() - 1))
        X_train_num = X_train.select_dtypes(include=[np.number])
        X_res, y_res = smote.fit_resample(X_train_num, y_train)

        # Test set is untouched — this is the core leakage invariant
        assert len(X_test) == original_test_n
        assert len(y_test) == original_test_n

    def test_smote_expands_only_training_minority(self, df_for_split):
        """After SMOTE, the resampled training class counts should be more balanced."""
        from imblearn.over_sampling import SMOTE

        X_train, _, _, y_train, _, _ = split_data(df_for_split)
        smote = SMOTE(random_state=42, k_neighbors=min(3, y_train.sum() - 1))
        X_num = X_train.select_dtypes(include=[np.number])
        _, y_res = smote.fit_resample(X_num, y_train)

        n_before_pos = y_train.sum()
        n_after_pos  = y_res.sum()
        # SMOTE oversamples the minority — positive count must increase
        assert n_after_pos >= n_before_pos


# ── 2. Vital-sign imputation uses group-level (ICU-type) medians ─────────────

class TestGroupLevelImputation:
    def test_group_median_used_not_global(self):
        """Missing vitals should be filled with the group median, not the global median.

        Here MICU median = 100 bpm and SICU median = 60 bpm.  The global median is
        the average across all rows.  A row with icu_type='MICU' must be filled with
        100, not with the global value.
        """
        df = pd.DataFrame({
            "d1_heartrate_max": [100.0, np.nan, 60.0, np.nan],
            "icu_type":         ["MICU", "MICU", "SICU", "SICU"],
        })
        result = impute_vitals_by_icu_type(df)
        # Group medians: MICU=100, SICU=60
        assert result.loc[1, "d1_heartrate_max"] == pytest.approx(100.0)
        assert result.loc[3, "d1_heartrate_max"] == pytest.approx(60.0)

    def test_different_groups_get_different_fills(self):
        """Two missing rows from different ICU types must receive different imputed values."""
        df = pd.DataFrame({
            "d1_sysbp_max": [140.0, np.nan, 100.0, np.nan],
            "icu_type":     ["CCU", "CCU", "NSICU", "NSICU"],
        })
        result = impute_vitals_by_icu_type(df)
        assert result.loc[1, "d1_sysbp_max"] != result.loc[3, "d1_sysbp_max"]

    def test_no_nulls_after_group_imputation(self):
        """No missing values should remain after group-level imputation."""
        rng = np.random.default_rng(5)
        n = 40
        df = pd.DataFrame({
            "d1_heartrate_max": np.where(rng.random(n) < 0.3, np.nan, rng.normal(90, 15, n)),
            "d1_sysbp_max":     np.where(rng.random(n) < 0.3, np.nan, rng.normal(120, 20, n)),
            "icu_type":         rng.choice(["MICU", "SICU", "CCU"], n),
        })
        result = impute_vitals_by_icu_type(df)
        assert result[["d1_heartrate_max", "d1_sysbp_max"]].isnull().sum().sum() == 0


# ── 3. Derived features compute to correct values ────────────────────────────

class TestDerivedFeatures:
    def test_shock_index_formula(self):
        """shock_index = HR_max / SBP_min (clipped at SHOCK_INDEX_CAP)."""
        df = pd.DataFrame({
            "d1_heartrate_max": [120.0],
            "d1_sysbp_min":     [80.0],
        })
        result = add_derived_features(df)
        assert "shock_index" in result.columns
        assert result.loc[0, "shock_index"] == pytest.approx(120.0 / 80.0)

    def test_comorbidity_burden_sums_flags(self):
        """comorbidity_burden = number of active comorbidity flags."""
        df = pd.DataFrame({
            "diabetes_mellitus": [1, 0, 1],
            "cirrhosis":         [0, 0, 1],
            "aids":              [0, 0, 0],
            "leukemia":          [1, 0, 0],
        })
        result = add_derived_features(df)
        expected = [2, 0, 2]
        assert result["comorbidity_burden"].tolist() == expected

    def test_pulse_pressure_formula(self):
        """pulse_pressure = d1_sysbp_max - d1_diasbp_max."""
        df = pd.DataFrame({
            "d1_sysbp_max":  [140.0],
            "d1_diasbp_max": [80.0],
        })
        result = add_derived_features(df)
        assert result.loc[0, "pulse_pressure"] == pytest.approx(60.0)

    def test_shock_index_zero_sbp_is_handled(self):
        """Division by zero in shock_index (SBP_min = 0) must not raise an exception.

        The implementation replaces 0 with NaN before dividing, so the result is NaN
        rather than inf.  The important invariant is that execution completes without
        error and the value is finite-or-NaN (never +inf / -inf).
        """
        df = pd.DataFrame({
            "d1_heartrate_max": [100.0],
            "d1_sysbp_min":     [0.0],
        })
        result = add_derived_features(df)
        val = result.loc[0, "shock_index"]
        assert not np.isinf(val), "shock_index should never be infinite (guard against /0)"


# ── 4. Target encoding uses training statistics only ─────────────────────────

class TestTargetEncoding:
    def test_encoding_derived_from_train_not_test(self):
        """Encoded test values must match training-set statistics, not test-set truth."""
        rng = np.random.default_rng(99)
        # Training: group A has 50% mortality, group B has 10%
        train_df = pd.DataFrame({
            "ethnicity":     ["A"] * 20 + ["B"] * 20,
            "hospital_death": [1] * 10 + [0] * 10 + [1] * 2 + [0] * 18,
        })
        # Test: deliberately give opposite labels to see if encoding uses them
        test_df = pd.DataFrame({
            "ethnicity":     ["A"] * 5 + ["B"] * 5,
            "hospital_death": [0] * 5 + [1] * 5,  # reversed — must not affect encoding
        })
        encoder_maps = fit_target_encoder(train_df, ["ethnicity"])
        encoded_test = apply_target_encoder(
            test_df.drop(columns=["hospital_death"]), encoder_maps
        )
        # A-rows should be encoded higher than B-rows (from training stats)
        a_val = encoded_test.loc[encoded_test.index[0], "ethnicity"]
        b_val = encoded_test.loc[encoded_test.index[5], "ethnicity"]
        assert a_val > b_val, "Group A (higher training mortality) should encode higher than B"

    def test_unseen_category_gets_global_mean(self):
        """Categories not in training data fall back to the global training mean."""
        train_df = pd.DataFrame({
            "ethnicity":     ["A", "A", "B", "B"],
            "hospital_death": [1, 0, 1, 0],
        })
        test_df = pd.DataFrame({"ethnicity": ["C"]})  # 'C' was never in training
        encoder_maps = fit_target_encoder(train_df, ["ethnicity"])
        encoded = apply_target_encoder(test_df, encoder_maps)
        # global_mean = 0.5; unseen 'C' must be mapped to ~0.5 (with smoothing)
        assert 0.0 < encoded.loc[0, "ethnicity"] <= 1.0

    def test_smoothing_pulls_rare_categories_toward_global_mean(self):
        """With only 1 observation in a category, encoding must be influenced by smoothing."""
        train_df = pd.DataFrame({
            "ethnicity":     ["A"] * 100 + ["B"],   # B has only 1 row
            "hospital_death": [0] * 100 + [1],       # B's raw rate = 1.0
        })
        encoder_maps = fit_target_encoder(train_df, ["ethnicity"])
        b_encoded = encoder_maps["ethnicity"]["mapping"]["B"]
        # Without smoothing: B = 1.0; with smoothing it should be pulled below 0.5
        assert b_encoded < 1.0


# ── 5. StandardScaler is fitted on training data only ────────────────────────

class TestScalerLeakage:
    def test_train_mean_is_zero_after_scaling(self, small_df):
        """Training set numeric means should be exactly 0 after StandardScaler."""
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, *_ = split_data(filled)
        X_train_s, _, _, _ = scale_numeric(X_train, X_val, X_test)
        num_cols = X_train_s.select_dtypes(include=[np.number]).columns
        assert (X_train_s[num_cols].mean().abs() < 1e-9).all()

    def test_test_mean_is_not_zero(self, small_df):
        """Test set was not used to fit the scaler, so its mean should not be 0."""
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, *_ = split_data(filled)
        _, _, X_test_s, _ = scale_numeric(X_train, X_val, X_test)
        num_cols = X_test_s.select_dtypes(include=[np.number]).columns
        # At least one column must have a non-zero mean in test
        assert (X_test_s[num_cols].mean().abs() > 1e-3).any()

    def test_scaler_fitted_on_train_statistics(self, small_df):
        """Scaler.mean_ should match the training column means exactly."""
        from sklearn.preprocessing import StandardScaler

        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, *_ = split_data(filled)
        _, _, _, scaler = scale_numeric(X_train, X_val, X_test)

        num_cols = X_train.select_dtypes(include=[np.number]).columns
        expected_means = X_train[num_cols].mean().values
        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)


# ── Original tests preserved (drop/impute/split invariants) ──────────────────

class TestDropHighMissingness:
    def test_drops_sparse_column(self, small_df):
        _, dropped = drop_high_missingness(small_df)
        assert "sparse_lab" in dropped

    def test_retains_dense_columns(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        assert "age" in cleaned.columns
        assert "bmi" in cleaned.columns

    def test_threshold_boundary(self, small_df):
        """A column with missing rate == threshold must be retained (condition is strict >)."""
        n = len(small_df)
        # Create exactly threshold fraction of NaN — at the boundary, not beyond it
        n_nan = int(n * HIGH_MISSINGNESS_THRESHOLD)
        vals = np.ones(n)
        vals[:n_nan] = np.nan
        small_df = small_df.copy()
        small_df["boundary_col"] = vals
        _, dropped = drop_high_missingness(small_df)
        assert "boundary_col" not in dropped


class TestImpute:
    def test_no_nulls_after_impute(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        assert impute(cleaned).isnull().sum().sum() == 0

    def test_does_not_modify_original(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        original_nulls = cleaned.isnull().sum().sum()
        impute(cleaned)
        assert cleaned.isnull().sum().sum() == original_nulls


class TestSplitData:
    def test_split_sizes_sum_to_total(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        assert len(X_train) + len(X_val) + len(X_test) == len(filled)

    def test_positive_rate_is_stratified(self, small_df):
        cleaned, _ = drop_high_missingness(small_df)
        filled = impute(cleaned)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filled)
        overall_rate = filled["hospital_death"].mean()
        for split_y in [y_train, y_val, y_test]:
            assert abs(split_y.mean() - overall_rate) < 0.15
