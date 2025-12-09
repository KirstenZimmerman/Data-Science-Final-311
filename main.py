from src.data.load_data import load_train_test
from src.data.preprocess import clean_dataset

from src.visualization.eda import (
    churn_distribution,
    numeric_histograms,
    categorical_churn,
)

from src.utils.helper_functions import plot_roc_curve

from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.models.decisionTree_model import train_decision_tree_model
from src.models.randomForestClassifier_model import train_random_forest_model

from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def main() -> None:
    """ 
    Run the full customer churn prediction pipeline.
    This function loads and cleans the data, generates EDA visualizations,
    trains multiple models (baseline, KNN, decision tree, random forest),
    evaluates them using confusion matrices and ROC/AUC metrics, selects the
    best model based on validation performance, and writes a Kaggle-ready
    submission file.
    """

    print("Loading data")
    train_raw, test_raw = load_train_test(
        "data/raw/train.csv",
        "data/raw/test.csv",
    )

    # Print shape of the raw dataset
    print(f"Raw dataset shape: {train_raw.shape}")

    print("Cleaning data")
    train_clean, test_clean = clean_dataset(train_raw, test_raw)

    print(f"Cleaned dataset shape: {train_clean.shape}")

    os.makedirs("data/processed", exist_ok=True)
    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)

    print("Creating EDA visuals")
    #plot_eda(train_clean)

    # EDA
    churn_distribution(train_clean)
    numeric_histograms(train_clean)
    categorical_churn(train_clean)

    print("Splitting data")
    feature_cols = [
        "Contract Length",
        "Subscription Type",
        "Gender",
        "Total Spend",
        "Support Calls",
        "Usage Frequency",
        "Age",
        "Last Interaction",
        "Tenure",
        "Payment Delay",
    ]

    X = train_clean[feature_cols].copy()
    y = train_clean["Churn"]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1234,
        stratify=y,
    )

    print("Training models")
    knn_model = train_knn_model(X_train, y_train)
    dumb_model = train_dumb_model(X_train, y_train)
    tree_model = train_decision_tree_model(X_train, y_train)
    rf_model = train_random_forest_model(X_train, y_train)

    print("Evaluating on validation set")
    # class predictions
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_dumb = dumb_model.predict(X_val)
    y_val_pred_tree = tree_model.predict(X_val)
    y_val_pred_rf = rf_model.predict(X_val)

    #probabilities
    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    val_prob_dumb = dumb_model.predict_proba(X_val)[:, 1]
    val_prob_tree = tree_model.predict_proba(X_val)[:, 1]
    val_prob_rf = rf_model.predict_proba(X_val)[:, 1]

    plot_confusion_matrices(y_val, y_val_pred_dumb, y_val_pred_knn, model1_name="Predict No Churn", model2_name="k-NN",)
    plot_performance_comparison(y_val, y_val_pred_dumb, y_val_pred_knn, model1_name="Predict No Churn", model2_name="k-NN",)
    
    plot_confusion_matrices(y_val, y_val_pred_tree, y_val_pred_rf, model1_name="Decision Tree", model2_name="Random Forest")
    plot_performance_comparison(y_val, y_val_pred_tree, y_val_pred_rf, model1_name="Decision Tree", model2_name="Random Forest")

    auc_dumb = plot_roc_curve(y_val, val_prob_dumb, "Predict No Churn", show=False)
    auc_knn = plot_roc_curve(y_val, val_prob_knn, "k-NN", show=False)
    auc_tree = plot_roc_curve(y_val, val_prob_tree, "Decision Tree", show=False)
    auc_rf = plot_roc_curve(y_val, val_prob_rf, "Random Forest Classifier", show=True)

    # auc_dumb = roc_auc_score(y_val, val_prob_dumb, "Predict No Churn")
    # auc_knn = roc_auc_score(y_val, val_prob_knn, "k-NN")
    # auc_tree = roc_auc_score(y_val, val_prob_tree, "Decision Tree")
    # auc_rf = roc_auc_score(y_val, val_prob_rf, "Random Forest", show=True)

    print("Validation AUC - Predict No Churn:")
    print(auc_dumb)
    print("Validation AUC - KNN:")
    print(auc_knn)
    print("Validation AUC - Decision Tree:")
    print(auc_tree)
    print("Validation AUC - Random Forest:")
    print(auc_rf)

    # pick best model 
    models = [
        ("Predict No Churn", dumb_model, auc_dumb),
        ("k-NN", knn_model, auc_knn),
        ("Decision Tree", tree_model, auc_tree),
        ("Random Forest", rf_model, auc_rf),
    ]
    best_label, best_model, best_auc = max(models, key=lambda t: t[2])

    print("Best model on validation set:")
    print(best_label)
    print(best_auc)

    print("Generating test predictions with best model...")
    X_test = test_clean[feature_cols].copy()
    test_prob = best_model.predict_proba(X_test)[:, 1]

    os.makedirs("data/submissions", exist_ok=True)
    submission = pd.DataFrame(
        {
            "CustomerID": test_clean["CustomerID"],
            "Churn": test_prob,
        }
    )
    submission.to_csv("data/submissions/best_model_submission.csv", index=False)
    print("Wrote best_model_submission to a csv")
    print("Done.")

if __name__ == "__main__":
    main()
