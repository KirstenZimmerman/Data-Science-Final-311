"""
Train/validation splitting utilities for customer churn data.
This module provides a helper function to split a cleaned dataset
into stratified training and validation subsets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .load_data import load_dataset


def split_train_validation(
    
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.2,
    seed: int = 123,
): 
    """ Split a dataset into stratified training and validation sets. """
    if abs(train_frac + val_frac - 1.0) > 1e-8:
        raise ValueError("Train + validation fractions must sum to 1.0")

    train_df, val_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=df["Churn"],   
    )

    return train_df, val_df


if __name__ == "__main__":
    cleaned_path = "data/processed/train_clean.csv"
    df = load_dataset(cleaned_path)

    train_split, val_split = split_train_validation(df)

    os.makedirs("data/processed", exist_ok=True)
    train_split.to_csv("data/processed/train_split.csv", index=False)
    val_split.to_csv("data/processed/val_split.csv", index=False)

    print("Train + validation splits saved to data/processed/")