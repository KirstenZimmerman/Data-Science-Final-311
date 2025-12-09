"""
Data loading utilities for the customer churn project.
Provides helper functions to load raw training and test datasets
from CSV files with basic validation.
"""

import pandas as pd
from pathlib import Path

def load_dataset(file_path: str) -> pd.DataFrame:
    """ Load a CSV file into a pandas DataFrame. """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)

def load_train_test(train_path: str, test_path: str):
    """ Load training and test datasets. """
    
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    return train_df, test_df

if __name__ == "__main__":
    df = load_dataset("data/raw/train.csv")
    print(df.head())
