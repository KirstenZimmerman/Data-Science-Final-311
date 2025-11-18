import os
import pandas as pd
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values."""
    return df.dropna().copy()


if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/card_transdata.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/card_transdata_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
