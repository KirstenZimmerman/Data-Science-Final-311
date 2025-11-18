import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Create simple exploratory plots."""
    sns.countplot(x='fraud', data=df)
    plt.title('Fraud Class Distribution')
    plt.show()

    sns.countplot(x='used_pin_number', data=df)
    plt.title('Used PIN Distribution')
    plt.show()

    sns.countplot(x='repeat_retailer', data=df)
    plt.title('Repeat Retailer Distribution')
    plt.show()

    sns.histplot(df['ratio_to_median_purchase_price'], bins=30)
    plt.title('Ratio to Median Purchase Price Distribution')
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    plot_eda(clean)
