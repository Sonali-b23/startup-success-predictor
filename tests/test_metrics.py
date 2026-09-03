import numpy as np
import pytest

from src.metrics import classification_metrics, regression_metrics


def test_classification_metrics_on_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.95])

    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["recall_class_0"] == 1.0
    assert m["recall_class_1"] == 1.0


def test_classification_metrics_majority_baseline_matches_class_balance():
    # 3 out of 4 are class 1 -> majority baseline should be 0.75
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1])  # always predicts majority class
    y_proba = np.array([0.6, 0.6, 0.6, 0.6])

    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["majority_class_baseline_accuracy"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)  # matches baseline since it always predicts majority


def test_classification_metrics_catches_zero_recall_for_missed_class():
    # Model never predicts class 0 at all -- recall for class 0 must be 0.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_proba = np.array([0.5, 0.5, 0.9, 0.9])

    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["recall_class_0"] == 0.0


def test_regression_metrics_known_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((10 + 10 + 0) / 3)
    assert m["mse"] == pytest.approx((100 + 100 + 0) / 3)
    assert m["rmse"] == pytest.approx(m["mse"] ** 0.5)
