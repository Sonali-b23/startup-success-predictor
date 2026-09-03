import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from src.preprocessing import load_data, preprocess
from src.evaluate import create_explainer
from src.persistence import load_models

st.set_page_config(page_title="Startup Success Predictor", layout="wide", page_icon="🚀")


@st.cache_resource(show_spinner="Loading trained models...")
def setup_pipeline():
    """
    Loads the models trained (and honestly evaluated on held-out data) by
    runner.py, plus the preprocessed dataset needed to fill in default
    values for the ~30 features the UI doesn't expose sliders for.

    Deliberately does NOT retrain here: the old version retrained on every
    cold start using 100% of the data with zero held-out validation, so the
    app's live model was never actually checked for overfitting. Now the
    only model this app ever serves is the one runner.py validated.
    """
    rf_class, rf_reg, feature_names, metrics = load_models()

    df = load_data('data/Data.csv')
    X, _, _ = preprocess(df)
    X_df = pd.DataFrame(X, columns=feature_names)

    explainer = create_explainer(rf_class)  # no file I/O -- safe to build every session

    return rf_class, rf_reg, X_df, explainer, metrics


try:
    rf_class, rf_reg, X_df, explainer, metrics = setup_pipeline()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()


# UI DESIGN
st.title("🚀 Startup Success & Funding Predictor")
st.markdown("Enter a startup's core metrics below to dynamically predict if it will be **Acquired** or **Closed**, and estimate its **Funding Potential** using our Machine Learning engine.")

if metrics:
    c = metrics["classification"]
    r = metrics["regression"]
    st.caption(
        f"📊 Model performance (held-out test data, from the last training run): "
        f"**{c['accuracy']:.1%} accuracy** (vs. {c['majority_class_baseline_accuracy']:.1%} for always guessing the majority class), "
        f"**ROC-AUC {c['roc_auc']:.2f}**, "
        f"recall on \"closed\" outcomes: {c['recall_class_0']:.0%}, "
        f"funding MAE: ${r['mae']:,.0f}. "
        f"Re-run `python runner.py` to refresh these numbers on new data."
    )

st.sidebar.header("📊 Startup Profile")
st.sidebar.markdown("Adjust the sliders to build your ideal startup.")

# Top features identified during EDA & SHAP analysis
relationships = st.sidebar.slider("Number of Relationships (Partners/Investors)", min_value=0, max_value=50, value=3)
milestones = st.sidebar.slider("Number of Milestones Achieved", min_value=0, max_value=10, value=1)
funding_rounds = st.sidebar.slider("Total Funding Rounds", min_value=1, max_value=15, value=2)
is_top500 = st.sidebar.selectbox("Is the Startup in a Top 500 Network?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
has_roundB = st.sidebar.selectbox("Has the Startup Secured Round B Funding?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
has_VC = st.sidebar.selectbox("Does it have VC backing?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Build the input row (start with the median of the dataset so other 40 variables don't crash)
input_data = X_df.median().to_dict()

# Override the medians with the user's specific inputs
input_data['relationships'] = relationships
input_data['milestones'] = milestones
input_data['funding_rounds'] = funding_rounds
input_data['is_top500'] = is_top500
input_data['has_roundB'] = has_roundB
input_data['has_VC'] = has_VC

# Convert to DataFrame
input_df = pd.DataFrame([input_data])


st.markdown("---")
# Action Button
if st.button("🔮 Predict Startup Outcome", use_container_width=True):
    col1, col2 = st.columns(2)

    # 1. Predict Success/Failure
    class_pred = rf_class.predict(input_df)[0]
    class_prob_success = rf_class.predict_proba(input_df)[0][1]

    # 2. Predict Funding
    reg_pred = rf_reg.predict(input_df)[0]

    with col1:
        st.subheader("🎯 Viability Prediction")
        if class_pred == 1:
            st.success(f"**ACQUIRED (SUCCESS)**\n\nProbability of Success: {class_prob_success:.1%}")
        else:
            st.error(f"**CLOSED (FAILURE)**\n\nProbability of Success: {class_prob_success:.1%}")

    with col2:
        st.subheader("💰 Estimated Funding Potential")
        st.info(f"**${reg_pred:,.2f}**")

    st.markdown("---")
    st.markdown("### 🧠 AI Explainability (Why?)")
    st.markdown("This SHAP Waterfall chart breaks down the exact mathematical impact of your inputs. It shows how the AI started at a baseline probability, and how your specific inputs pushed the prediction either toward Success (Red) or Failure (Blue).")

    # Generate SHAP Waterfall for the single prediction
    fig, ax = plt.subplots(figsize=(10, 5))
    shap_val_single = explainer(input_df)

    # Extract the explanations for the positive class (Acquired)
    if len(shap_val_single.shape) == 3:
        shap_val_single = shap_val_single[..., 1]

    shap.plots.waterfall(shap_val_single[0], show=False)
    st.pyplot(fig)
