# """Utility helpers for the project."""


# def main() -> None:
#     pass


# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(y_true, y_scores, label: str):
    """
    Plot an ROC curve and return the AUC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc_value

