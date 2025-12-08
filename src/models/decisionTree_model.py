import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
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
BASELINE_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

def build_tree_pipeline():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        (StandardScaler(), NUMERIC_COLS),
        remainder="drop",
    )

    tree_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=240,
        random_state=1234,
    )

    pipeline = make_pipeline(preprocessor, tree_model)
    return pipeline

def train_decision_tree_model(X_train, y_train):
    model = build_tree_pipeline()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")

    if "Payment Delay" in train.columns:
        train["Payment Delay"] = pd.to_numeric(train["Payment Delay"], errors="coerce")
    if "Payment Delay" in test.columns:
        test["Payment Delay"] = pd.to_numeric(test["Payment Delay"], errors="coerce")

    X = train[BASELINE_FEATURES].copy()
    y = train["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1234,
        stratify=y,
    )

    tree_model = train_decision_tree_model(X_train, y_train)

    val_pred_proba = tree_model.predict_proba(X_val)[:, 1]
    val_pred_class = tree_model.predict(X_val)
    auc_tree = roc_auc_score(y_val, val_pred_proba)

    accuracy = accuracy_score(y_val, val_pred_class)
    f1 = f1_score(y_val, val_pred_class)
    conf_matrix = confusion_matrix(y_val, val_pred_class)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc_tree:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    X_test = test[BASELINE_FEATURES]
    test_pred_tree = tree_model.predict_proba(X_test)[:, 1]

    os.makedirs("data/submissions", exist_ok=True)

    submission_tree = pd.DataFrame({
        "CustomerID": test["CustomerID"],
        "Churn": test_pred_tree,
    })

    submission_tree.to_csv("data/submissions/tree_baseline.csv", index=False)
    print("Wrote tree_baseline to a csv")
