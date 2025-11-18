import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_knn) -> None:
    """Plot confusion matrices for both models."""
    conf_baseline = confusion_matrix(y_test, y_pred_baseline)
    conf_knn = confusion_matrix(y_test, y_pred_knn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(conf_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title('Never Fraud')
    sns.heatmap(conf_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('3-NN')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_test, y_pred_baseline, y_pred_knn) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    baseline_scores = [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline, zero_division=0),
        recall_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_baseline)
    ]
    knn_scores = [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_knn)
    ]
    df = pd.DataFrame({'Metric': metrics, 'k-NN': knn_scores, 'Never Fraud': baseline_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset
    from src.models.train_model import train_models

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    y_test, baseline, knn = train_models(clean)
    plot_confusion_matrices(y_test, baseline, knn)
    plot_performance_comparison(y_test, baseline, knn)
