from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/raw/card_transdata.csv")
    
    # Print shape of the raw dataset
    print(f"Raw dataset shape: {raw_df.shape}")

    print("---Cleaning data...")
    clean_df = clean_dataset(raw_df)

    print(f"Cleaned dataset shape: {clean_df.shape}")

    print("---Creating EDA visuals...")
    plot_eda(clean_df)

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)

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


if __name__ == "__main__":
    main()
