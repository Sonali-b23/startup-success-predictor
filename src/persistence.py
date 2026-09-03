"""
Save/load the trained models and the exact feature column order they were
trained on. Separated out so app.py can *load* a model that was trained and
evaluated once by runner.py, instead of retraining on every cold start with
no held-out validation of its own.
"""
import json
import os

import joblib

DEFAULT_MODEL_DIR = "models"


def save_models(rf_class, rf_reg, feature_names, metrics, model_dir=DEFAULT_MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_class, os.path.join(model_dir, "rf_classifier.joblib"))
    joblib.dump(rf_reg, os.path.join(model_dir, "rf_regressor.joblib"))
    joblib.dump(list(feature_names), os.path.join(model_dir, "feature_names.joblib"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def models_exist(model_dir=DEFAULT_MODEL_DIR):
    required = ["rf_classifier.joblib", "rf_regressor.joblib", "feature_names.joblib"]
    return all(os.path.exists(os.path.join(model_dir, name)) for name in required)


def load_models(model_dir=DEFAULT_MODEL_DIR):
    """
    Raises FileNotFoundError (with a clear message) if the models haven't
    been trained yet, rather than silently retraining -- callers (app.py)
    should catch this and tell the user to run `python runner.py` first.
    """
    if not models_exist(model_dir):
        raise FileNotFoundError(
            f"No trained models found in '{model_dir}/'. Run `python runner.py` first "
            "to train and save the models before launching the app."
        )
    rf_class = joblib.load(os.path.join(model_dir, "rf_classifier.joblib"))
    rf_reg = joblib.load(os.path.join(model_dir, "rf_regressor.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))

    metrics = None
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return rf_class, rf_reg, feature_names, metrics
