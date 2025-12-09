import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_knn, model1_name: str, model2_name: str,) -> None:
    """ Plot confusion matrices for both models. """

    conf_baseline = confusion_matrix(y_test, y_pred_baseline)
    conf_knn = confusion_matrix(y_test, y_pred_knn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(conf_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title(model1_name)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(conf_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(model2_name)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from src.data.load_data import load_train_test
    from src.data.preprocess import clean_dataset
    from src.models.train_model import train_models

    train_raw, test_raw = load_train_test(
        "data/raw/train.csv",
        "data/raw/test.csv",
    )

    train_clean, test_clean = clean_dataset(train_raw, test_raw)

    y_test, y_pred_baseline, y_pred_knn = train_models(train_clean)
  
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_knn)