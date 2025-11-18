from src.data.load_data import load_dataset
from src.features.build_features import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import train_models
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)


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

    print("---Training ML models...")
    y_test, baseline_pred, knn_pred = train_models(clean_df)

    print("---Creating performance visuals...")
    plot_confusion_matrices(y_test, baseline_pred, knn_pred)
    plot_performance_comparison(y_test, baseline_pred, knn_pred)
    print("Done.")


if __name__ == "__main__":
    main()
