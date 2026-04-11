from src.preprocessing import load_data, preprocess
from src.train import train_models
from src.evaluate import explain_model
import pandas as pd

print("Loading data...")
df = load_data('data/Data.csv')
print("Preprocessing data...")
X, y_class, y_reg = preprocess(df)
original_columns = X.columns.tolist()
print("Training models...")
rf_class, rf_reg = train_models(X, y_class, y_reg)
print("Running SHAP explainability...")
# Pass X as DataFrame to keep column names for the plot
X_df = pd.DataFrame(X, columns=original_columns)
explain_model(rf_class, X_df, feature_names=original_columns)
print("Done!")
