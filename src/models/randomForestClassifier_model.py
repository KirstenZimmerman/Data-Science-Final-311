"""
Random Forest model for customer churn prediction.
This module defines a preprocessing + RandomForestClassifier pipeline,
provides a training function for use in main.py, and allows standalone
execution to evaluate performance and generate a Kaggle-ready submission.
"""

import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

CATEGORICAL_COLS = ["Contract Length", "Subscription Type", "Gender"]
NUMERIC_COLS = [
    "Total Spend",
    "Support Calls",
    "Usage Frequency",
    "Age",
    "Last Interaction",
    "Tenure",
    "Payment Delay",
]
RF_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

def build_rf_pipeline():
    """ Build and return a preprocessing + Random Forest pipeline. """

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        remainder="passthrough",
    )

    rf_model = RandomForestClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=1234,
    )

    pipeline = make_pipeline(preprocessor, rf_model)
    return pipeline

def train_random_forest_model(X_train, y_train):
    """ Train and return a Random Forest churn prediction model. """
    
    model = build_rf_pipeline()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")

    X = train[RF_FEATURES].copy()
    y = train["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1234,
        stratify=y,
    )

    rf_model = train_random_forest_model(X_train, y_train)

    val_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    y_val_pred = rf_model.predict(X_val)

    auc_rf = roc_auc_score(y_val, val_pred_proba)
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc_rf:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    X_test = test[RF_FEATURES].copy()
    test_pred_rf = rf_model.predict_proba(X_test)[:, 1]

    os.makedirs("data/submissions", exist_ok=True)
    submission_rf = pd.DataFrame({
        "CustomerID": test["CustomerID"],
        "Churn": test_pred_rf,
    })

    submission_rf.to_csv("data/submissions/random_forest.csv", index=False)
    print("Wrote random_forest to a csv")
