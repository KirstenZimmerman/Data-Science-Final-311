import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

CATEGORICAL_COLS = [
    "Contract Length",
    "Gender",
    "Subscription Type",
]

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

def build_knn_pipeline():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        (StandardScaler(), NUMERIC_COLS),
        remainder="drop",
    )

    knn_spec = KNeighborsClassifier(
        n_neighbors=21,
        weights="distance",
    )

    pipeline = make_pipeline(preprocessor, knn_spec)
    return pipeline

def train_knn_model(X_train, y_train):
    model = build_knn_pipeline()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")

    X = train[BASELINE_FEATURES].copy()
    y = train["Churn"]

    #train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1234,
        stratify=y,
    )

    #train
    knn_model = train_knn_model(X_train, y_train)

    #validate
    val_pred_proba = knn_model.predict_proba(X_val)[:, 1]
    auc_knn = roc_auc_score(y_val, val_pred_proba)

    print("KNN validation AUC:")
    print(auc_knn)

    #predict
    X_test = test[BASELINE_FEATURES]
    test_pred_knn = knn_model.predict_proba(X_test)[:, 1]

    #save
    os.makedirs("data/submissions", exist_ok=True)
    submission_knn = pd.DataFrame({
        "CustomerID": test["CustomerID"],
        "Churn": test_pred_knn,
    })

    submission_knn.to_csv("data/submissions/knn_baseline_model.csv", index=False)
    print("Wrote knn_baseline_model to a csv")


# def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
#     """Train and return a 3-NN classifier."""
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X_train, y_train)
#     return model
