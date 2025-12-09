import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y_true, y_scores, label: str, show: bool = False):
    """
    Add an ROC curve to the current figure.
    Returns the AUC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_value:.3f})")

    if show:
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Validation Set)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return auc_value
