from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

def train_models(X, y_class, y_reg):
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    preds = log_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    lin_model = LinearRegression()
    lin_model.fit(X, y_reg)

    return log_model, lin_model