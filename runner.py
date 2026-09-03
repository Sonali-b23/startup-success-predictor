import json

import pandas as pd

from src.evaluate import create_explainer, generate_summary_plot
from src.persistence import save_models
from src.preprocessing import load_data, preprocess
from src.train import train_models

print("Loading data...")
df = load_data('data/Data.csv')

print("Preprocessing data...")
X, y_class, y_reg = preprocess(df)
original_columns = X.columns.tolist()
X_df = pd.DataFrame(X, columns=original_columns)

print("Training models (with held-out evaluation)...")
rf_class, rf_reg, metrics = train_models(X_df, y_class, y_reg)

print("\n--- Metrics summary (held-out test data only) ---")
print(json.dumps(metrics, indent=2))

print("\nRunning SHAP explainability...")
explainer = create_explainer(rf_class)
generate_summary_plot(explainer, X_df)

print("\nSaving trained models...")
save_models(rf_class, rf_reg, original_columns, metrics)
print("Models and metrics saved to models/ -- run `streamlit run app.py` to launch the app.")

print("\nDone!")
