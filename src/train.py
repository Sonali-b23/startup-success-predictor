from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error

def train_models(X, y_class, y_reg):
    # Split for Classificaiton
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    # Split for Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    print("\n--- Training Classification Model ---")
    rf_model_class = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model_class.fit(X_train_c, y_train_c)
    
    class_preds = rf_model_class.predict(X_test_c)
    print("Accuracy:", accuracy_score(y_test_c, class_preds))
    print("\nClassification Report:\n", classification_report(y_test_c, class_preds))
    
    print("\n--- Training Regression Model ---")
    rf_model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_reg.fit(X_train_r, y_train_r)
    
    reg_preds = rf_model_reg.predict(X_test_r)
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_r, reg_preds))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test_r, reg_preds))
    
    return rf_model_class, rf_model_reg