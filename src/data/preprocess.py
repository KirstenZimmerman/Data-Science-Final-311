import os
import numpy as np
import pandas as pd
from .load_data import load_dataset


# def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
#     """Clean the dataset by dropping rows with missing values."""
#     return df.dropna().copy()


# if __name__ == "__main__":
#     # Load the raw dataset
#     raw = load_dataset("data/raw/card_transdata.csv")
#     # Clean the dataset
#     cleaned = clean_dataset(raw)
#     # Ensure the processed directory exists
#     os.makedirs("data/processed", exist_ok=True)
#     # Save the cleaned data
#     processed_path = "data/processed/card_transdata_clean.csv"
#     cleaned.to_csv(processed_path, index=False)
#     print(f"Cleaned data saved to {processed_path}")

def clean_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame):
   
    train = train_df.copy()
    test = test_df.copy()

    numeric_cols = [
        "Age", "Tenure", "Usage Frequency", "Support Calls",
        "Payment Delay", "Total Spend", "Last Interaction"
    ]

    categorical_cols = ["Contract Length", "Subscription Type", "Gender"]

    # Replace none-like values
    for col in numeric_cols + categorical_cols:
        if col in train.columns:
            train[col] = train[col].replace(["none", "None", ""], np.nan)
        if col in test.columns:
            test[col] = test[col].replace(["none", "None", ""], np.nan)

    # Convert numeric
    for col in numeric_cols:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors="coerce")

    date_cols = ["Last Due Date", "Last Payment Date"]

    for col in date_cols:
        if col in train.columns:
            train[col] = pd.to_datetime(
                train[col],
                format="%m-%d",
                errors="coerce"
            )

    if "Payment Delay" in train.columns and all(c in train.columns for c in date_cols):
        mask_train = (
            train["Payment Delay"].isna()
            & train["Last Due Date"].notna()
            & train["Last Payment Date"].notna()
        )
        train.loc[mask_train, "Payment Delay"] = (
            train.loc[mask_train, "Last Payment Date"]
            - train.loc[mask_train, "Last Due Date"]
        ).dt.days

    # Fallback: if any Payment Delay still missing (e.g., missing dates), fill with 0
    if "Payment Delay" in train.columns:
        train["Payment Delay"] = train["Payment Delay"].fillna(0)

    if "Tenure" in train.columns:
        median_ten = train["Tenure"].median()
        train["Tenure"] = train["Tenure"].fillna(median_ten)
        if "Tenure" in test.columns:
            test["Tenure"] = test["Tenure"].fillna(median_ten)

    if "Support Calls" in train.columns:
        train["Support Calls"] = train["Support Calls"].fillna(0)
        if "Support Calls" in test.columns:
            test["Support Calls"] = test["Support Calls"].fillna(0)

    if "Last Interaction" in train.columns:
        train["Last Interaction"] = train["Last Interaction"].fillna(0)
        if "Last Interaction" in test.columns:
            test["Last Interaction"] = test["Last Interaction"].fillna(0)

    # Categorical fill
    for col in categorical_cols:
        if col in train.columns:
            train[col] = train[col].fillna("Unknown")
        if col in test.columns:
            test[col] = test[col].fillna("Unknown")

    return train, test


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    train_raw, test_raw = load_train_test(
        "data/raw/train.csv",
        "data/raw/test.csv",
    )

    train_clean, test_clean = clean_dataset(train_raw, test_raw)

    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)

    print("Saved cleaned datasets!")
