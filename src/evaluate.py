import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def explain_model(model, X, feature_names=None, output_dir='outputs'):
    """
    Generates SHAP values and feature importance plots to explain the model's predictions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\n--- Generating SHAP Explanations ---")
    # TreeExplainer is highly optimized for Random Forest
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample (e.g. 100 rows) to keep computation fast
    X_sample = X.sample(n=min(100, len(X)), random_state=42) if isinstance(X, pd.DataFrame) else pd.DataFrame(X[:100], columns=feature_names)
    
    shap_values = explainer.shap_values(X_sample)
    
    # For RandomForestClassifier, shap_values is a list of arrays (one for each class).
    # We take the SHAP values for the positive class (acquired=1).
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
    
    # Generate Summary Plot (Bar Chart)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Success Prediction)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"SHAP feature importance plot saved to: {plot_path}")
    return explainer, shap_values_to_plot