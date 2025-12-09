"""
This baseline does not use machine learning. It applies simple rules motivated by
EDA on four key features:
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

SMART_BASELINE_FEATURES = [
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Age",
]


def smart_baseline_predict(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized rule-based churn prediction on a DataFrame.
    """
    rule = (
        (df["Support Calls"] >= 5)
        | (df["Payment Delay"] >= 20)
        | (df["Total Spend"] < 500)
        | (df["Age"] >= 50)
    )
    return rule.astype(int)

if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")
   
    y_true = train["Churn"]
    y_pred = smart_baseline_predict(train)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)

    print("Smart Rule-Based Baseline")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("Confusion matrix:")
    print(conf)

    test_pred = smart_baseline_predict(test)

    os.makedirs("data/submissions", exist_ok=True)
    submission = pd.DataFrame({
        "CustomerID": test["CustomerID"],
        "Churn": test_pred,
    })
    submission.to_csv("data/submissions/smart_baseline.csv", index=False)
    print("Wrote smart_baseline.csv")