import os

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import load_data, preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Data.csv")


@pytest.fixture(scope="module")
def preprocessed():
    df = load_data(DATA_PATH)
    return preprocess(df)


def test_leaky_and_identifier_columns_are_dropped(preprocessed):
    X, y_class, y_reg = preprocessed
    # closed_at is the classic leak in this dataset (a company only has a
    # close date if it closed) -- it's caught by the generic "_at" suffix
    # rule, but pin it explicitly so a future refactor can't silently drop
    # that rule and reintroduce the leak without a test failing.
    leaky_or_identifier_columns = [
        "closed_at", "founded_at", "first_funding_at", "last_funding_at",
        "labels", "id", "object_id", "name", "zip_code", "status",
        "funding_total_usd", "Unnamed: 0", "Unnamed: 6", "state_code.1",
    ]
    for col in leaky_or_identifier_columns:
        assert col not in X.columns, f"leaky/identifier column '{col}' should not be in the feature matrix"


def test_no_missing_values_after_imputation(preprocessed):
    X, y_class, y_reg = preprocessed
    assert X.isnull().sum().sum() == 0


def test_all_features_are_numeric(preprocessed):
    X, y_class, y_reg = preprocessed
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"column '{col}' should be numeric after encoding"


def test_target_class_is_binary_and_matches_acquired_label():
    df = load_data(DATA_PATH)
    X, y_class, y_reg = preprocess(df)
    assert set(y_class.unique()) <= {0, 1}
    # Sanity check against the raw data: count of 1s should equal count of
    # rows whose original status was 'acquired'.
    raw = load_data(DATA_PATH)
    assert y_class.sum() == (raw["status"] == "acquired").sum()


def test_regression_target_is_numeric_and_not_in_features(preprocessed):
    X, y_class, y_reg = preprocessed
    assert pd.api.types.is_numeric_dtype(y_reg)
    assert "funding_total_usd" not in X.columns


def test_row_count_preserved(preprocessed):
    X, y_class, y_reg = preprocessed
    raw = load_data(DATA_PATH)
    assert len(X) == len(raw)
    assert len(y_class) == len(raw)
    assert len(y_reg) == len(raw)
