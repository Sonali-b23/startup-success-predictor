import os

import matplotlib.pyplot as plt
import pandas as pd
import shap


def create_explainer(model):
    """
    Builds a SHAP TreeExplainer for the given model. Pure -- no file I/O --
    so this is safe to call from app.py on every session without side
    effects (unlike generate_summary_plot below, which writes a PNG and
    should only run as part of the training pipeline, not on every app
    cold start).
    """
    return shap.TreeExplainer(model)


def generate_summary_plot(explainer, X, output_dir='outputs', sample_size=100, random_state=42):
    """
    Generates and saves the SHAP feature-importance summary plot. This is a
    training-time artifact (it writes outputs/shap_summary.png) -- call it
    from runner.py only. The Streamlit app builds its own per-prediction
    waterfall explanations via create_explainer() instead, without touching
    this file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n--- Generating SHAP Explanations ---")
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state) if isinstance(X, pd.DataFrame) \
        else pd.DataFrame(X[:sample_size])

    shap_values = explainer.shap_values(X_sample)

    # For RandomForestClassifier, shap_values is a list of arrays (one per class).
    # We take the SHAP values for the positive class (acquired=1).
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Success Prediction)')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"SHAP feature importance plot saved to: {plot_path}")
    return plot_path


def explain_model(model, X, feature_names=None, output_dir='outputs'):
    """
    Backwards-compatible convenience wrapper combining create_explainer +
    generate_summary_plot, kept for any external caller expecting the old
    single-function interface. runner.py calls the two pieces separately.
    """
    explainer = create_explainer(model)
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)
    generate_summary_plot(explainer, X_df, output_dir=output_dir)
    return explainer, None
