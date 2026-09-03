import json

import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

from src.persistence import save_models, load_models, models_exist


@pytest.fixture
def tiny_fitted_models():
    # Tiny fake models -- persistence logic doesn't care what the model
    # class is, just that it round-trips correctly through joblib.
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_class = np.array([1, 0, 1, 0])
    y_reg = np.array([10.0, 20.0, 30.0, 40.0])

    rf_class = LogisticRegression().fit(X, y_class)
    rf_reg = LinearRegression().fit(X, y_reg)
    return rf_class, rf_reg


def test_models_exist_is_false_for_empty_dir(tmp_path):
    assert models_exist(model_dir=str(tmp_path)) is False


def test_load_models_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python runner.py"):
        load_models(model_dir=str(tmp_path))


def test_save_then_load_round_trips_correctly(tmp_path, tiny_fitted_models):
    rf_class, rf_reg = tiny_fitted_models
    feature_names = ["feature_a", "feature_b"]
    metrics = {"classification": {"accuracy": 0.9}, "regression": {"mae": 123.0}}

    save_models(rf_class, rf_reg, feature_names, metrics, model_dir=str(tmp_path))
    assert models_exist(model_dir=str(tmp_path)) is True

    loaded_class, loaded_reg, loaded_features, loaded_metrics = load_models(model_dir=str(tmp_path))

    assert loaded_features == feature_names
    assert loaded_metrics == metrics

    X = np.array([[0, 1], [1, 1]])
    np.testing.assert_array_equal(loaded_class.predict(X), rf_class.predict(X))
    np.testing.assert_array_equal(loaded_reg.predict(X), rf_reg.predict(X))


def test_saved_metrics_file_is_valid_json(tmp_path, tiny_fitted_models):
    rf_class, rf_reg = tiny_fitted_models
    metrics = {"classification": {"accuracy": 0.9}, "regression": {"mae": 123.0}}
    save_models(rf_class, rf_reg, ["a", "b"], metrics, model_dir=str(tmp_path))

    with open(tmp_path / "metrics.json") as f:
        loaded = json.load(f)
    assert loaded == metrics
