"""
Evaluation metrics for both models, computed on a held-out test set only.

Kept separate from train.py so the same metric computation can be unit
tested and reused by both runner.py (prints/saves them after training) and
app.py (reads the saved metrics to show real, honest numbers in the UI
instead of no metrics at all, or numbers from an unvalidated full-data fit).
"""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(y_true, y_pred, y_proba):
    """
    Returns a JSON-serializable dict of classification metrics, including
    the majority-class baseline accuracy so "72% accuracy" can be read next
    to "64% for just guessing the majority class every time" rather than in
    isolation.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    majority_share = max(y_true.mean(), 1 - y_true.mean())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "majority_class_baseline_accuracy": float(majority_share),
        "precision_class_0": report["0"]["precision"],
        "recall_class_0": report["0"]["recall"],
        "f1_class_0": report["0"]["f1-score"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
        "f1_class_1": report["1"]["f1-score"],
    }


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": mse ** 0.5,
    }
