import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values."""
    return df.dropna().copy()


if __name__ == "__main__":
    from src.data.load_data import load_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    cleaned = clean_dataset(raw)
    print(cleaned.head())
