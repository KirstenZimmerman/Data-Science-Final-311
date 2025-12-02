from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
    
    churn_distribution,
    numeric_histograms,
    categorical_churn,
    correlation_heatmap
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def main() -> None:

    print("---Loading data...")
    #raw_df = load_dataset("data/raw/card_transdata.csv")
    train_raw, test_raw = load_train_test(
        "data/raw/train.csv",
        "data/raw/test.csv",
    )

    # Print shape of the raw dataset
    print(f"Raw dataset shape: {train_raw.shape}")

    print("---Cleaning data...")
    train_clean, test_clean = clean_dataset(train_raw, test_raw)

    print(f"Cleaned dataset shape: {train_clean.shape}")

    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)

    print("---Creating EDA visuals...")
    plot_eda(train_clean)

    # EDA
    churn_distribution(train_clean)
    numeric_histograms(train_clean)
    categorical_churn(train_clean)
    correlation_heatmap(train_clean)

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_train_df)




#stopped here
    print("---Training models...")
    knn_model = train_knn_model(X_train, y_train)
    dumb_model = train_dumb_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_dumb = dumb_model.predict(X_val)

    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    val_prob_dumb = dumb_model.predict_proba(X_val)[:, 1]

    plot_confusion_matrices(y_val, y_val_pred_dumb, y_val_pred_knn)
    plot_performance_comparison(y_val, y_val_pred_dumb, y_val_pred_knn)

    auc_dumb = plot_roc_curve(y_val, val_prob_dumb, "Never Fraud")
    auc_knn = plot_roc_curve(y_val, val_prob_knn, "3-NN")

    best_model = knn_model if auc_knn >= auc_dumb else dumb_model
    best_label = "3-NN" if best_model is knn_model else "Never Fraud"

    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Done.")

#st here
if __name__ == "__main__":
    main()
