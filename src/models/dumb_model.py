import pandas as pd
from sklearn.dummy import DummyClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

BASELINE_FEATURES = [
    "Total Spend",
    "Usage Frequency",
    "Support Calls",
]

def train_dumb_model(X_train, y_train):
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")

    X = train[BASELINE_FEATURES]
    y = train["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1234,
        stratify=y,
    )

    model = train_dumb_model(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:, 1]

    print("Accuracy:")
    print(accuracy_score(y_val, y_pred_val))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred_val))
    print()
    print("Classification report:")
    print(classification_report(y_val, y_pred_val))
    print()
    print("AUC:")
    print(roc_auc_score(y_val, y_proba_val))

    # Predict on test set (always 0)
    X_test = test[BASELINE_FEATURES]
    test_pred = model.predict(X_test)

    os.makedirs("data/submissions", exist_ok=True)
    submission = pd.DataFrame({
        "CustomerID": test["CustomerID"],
        "Churn": test_pred,
    })

    submission.to_csv("data/submissions/dumb_baseline.csv", index=False)
    print("Wrote dumb_baseline to a csv")

# def train_dumb_model(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
#     """Train a model that always predicts the majority class (never fraud)."""
#     model = DummyClassifier(strategy="constant", constant=0)
#     model.fit(X_train, y_train)
#     return model
