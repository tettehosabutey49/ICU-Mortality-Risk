"""
Preprocessing utilities for the ICU Mortality Risk dataset.

Handles missing-value imputation, type casting, outlier capping, and
train/validation/test splitting so that every notebook starts from a
clean, reproducible state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constants — all thresholds are named so readers know *why* the value exists  #
# --------------------------------------------------------------------------- #

# Features missing >60 % of values carry too little signal to impute reliably.
HIGH_MISSINGNESS_THRESHOLD = 0.60

# Winsorise continuous vitals at 1st / 99th percentile to suppress transcription
# outliers that are physiologically impossible (e.g. HR = 0 or HR = 400).
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the *training* split reserved for validation


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV and enforce sensible dtypes immediately on read.

    Categorical columns encoded as integers (e.g. ethnicity codes) must be
    kept as object/category to prevent the scaler from treating them as
    ordinal quantities.
    """
    df = pd.read_csv(path, low_memory=False)
    # hospital_death must be integer for sklearn compatibility
    if "hospital_death" in df.columns:
        df["hospital_death"] = df["hospital_death"].astype(int)
    return df


def drop_high_missingness(df: pd.DataFrame, threshold: float = HIGH_MISSINGNESS_THRESHOLD) -> pd.DataFrame:
    """Remove columns whose missing-value rate exceeds *threshold*.

    Rationale: features that are absent for the majority of patients cannot
    be imputed without introducing more noise than signal.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    return df.drop(columns=cols_to_drop), cols_to_drop


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns, mode-impute categorical columns.

    Median is preferred over mean for ICU vitals because the distributions
    are typically right-skewed (e.g. creatinine, lactate).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def winsorise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cap extreme values at the 1st and 99th percentiles.

    Physiological measurements in electronic health records frequently
    contain transcription errors (e.g. a systolic BP of 0).  Winsorising
    is safer than removing rows because it preserves the full cohort size.
    """
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(WINSOR_LOWER)
        upper = df[col].quantile(WINSOR_UPPER)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "hospital_death",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / validation / test split preserving class ratios.

    Stratification is critical because mortality rate (~8 %) is low — a
    random split could place too few positive cases in the test set and
    inflate apparent AUC.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # VAL_SIZE is applied to the remaining training data, not the full set
    val_fraction_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_trainval,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and transform all splits.

    The scaler is fit *only* on training data to prevent data leakage.
    Returns the fitted scaler so it can be persisted alongside the model.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test, scaler
