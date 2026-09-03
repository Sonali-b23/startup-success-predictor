from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.metrics import classification_metrics, regression_metrics


def train_models(X, y_class, y_reg):
    """
    Trains both models with a held-out test split and returns them along
    with an honest metrics dict computed ONLY on that held-out data --
    never on data the model was trained on. Also runs 5-fold stratified
    cross-validation on the classifier for a more robust ROC-AUC estimate
    than a single train/test split gives (Random Forests can look quite
    different from one split to the next on a dataset this size).
    """
    # Split for Classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    # Split for Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("\n--- Training Classification Model ---")
    rf_model_class = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model_class.fit(X_train_c, y_train_c)

    class_preds = rf_model_class.predict(X_test_c)
    class_proba = rf_model_class.predict_proba(X_test_c)[:, 1]
    class_metrics = classification_metrics(y_test_c, class_preds, class_proba)

    print(f"Test Accuracy:  {class_metrics['accuracy']:.3f}")
    print(f"Test ROC-AUC:   {class_metrics['roc_auc']:.3f}")
    print(f"Majority-class baseline accuracy: {class_metrics['majority_class_baseline_accuracy']:.3f}")
    print(f"Class 0 (closed)   -- precision: {class_metrics['precision_class_0']:.2f}, recall: {class_metrics['recall_class_0']:.2f}")
    print(f"Class 1 (acquired) -- precision: {class_metrics['precision_class_1']:.2f}, recall: {class_metrics['recall_class_1']:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        X, y_class, cv=cv, scoring='roc_auc',
    )
    class_metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
    class_metrics["cv_roc_auc_std"] = float(cv_scores.std())
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    print("\n--- Training Regression Model ---")
    rf_model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_reg.fit(X_train_r, y_train_r)

    reg_preds = rf_model_reg.predict(X_test_r)
    reg_metrics = regression_metrics(y_test_r, reg_preds)
    print(f"Mean Absolute Error (MAE): {reg_metrics['mae']:,.2f}")
    print(f"Mean Squared Error (MSE):  {reg_metrics['mse']:,.2f}")

    metrics = {"classification": class_metrics, "regression": reg_metrics}

    # The models above are evaluated honestly on held-out data -- that's
    # what `metrics` reflects, and it's what gets persisted and shown in the
    # app. For the model that actually ships, refit on *all* available data
    # (standard practice once validation is done: more training data ->
    # generally a better final model, and we already know from the split
    # above roughly how well this architecture generalizes). This is the
    # deliberate, documented version of what the app used to do silently
    # with zero validation at all.
    print("\n--- Refitting final models on full dataset for deployment ---")
    final_rf_class = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    final_rf_class.fit(X, y_class)

    final_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    final_rf_reg.fit(X, y_reg)

    return final_rf_class, final_rf_reg, metrics
