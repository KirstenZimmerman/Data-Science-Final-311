import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def churn_distribution(df: pd.DataFrame) -> None:
    """Show overall churn class distribution."""

    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Also print normalized counts for quick reference
    print("Churn proportion:")
    print(df["Churn"].value_counts(normalize=True))


def numeric_histograms(df: pd.DataFrame) -> None:
    """
    Show overall numeric distributions, plus special views for
    variables where 0 means 'no activity'.
    """

    numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns

    # Overall numeric histograms
    df[numeric_cols].hist(figsize=(15, 12), bins=30)
    plt.tight_layout()
    plt.show()

    # Non-zero distributions for variables where 0 = no activity
    for col in ["Support Calls", "Last Interaction"]:
        if col not in df.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.histplot(df[df[col] > 0][col], bins=30)
        plt.title(f"{col} (Non-zero Distribution)")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Highlight Total Spend by churn as a 1D profile
    if "Total Spend" in df.columns and "Churn" in df.columns:
        bins = pd.cut(df["Total Spend"], bins=30)

        grouped = (
            df.groupby([bins, "Churn"])
              .size()
              .unstack(fill_value=0)
        )

        x = [interval.mid for interval in grouped.index]

        plt.figure(figsize=(8, 5))
        if 0 in grouped.columns:
            plt.plot(x, grouped[0], color="green", label="No Churn (0)")
        if 1 in grouped.columns:
            plt.plot(x, grouped[1], color="red", label="Churn (1)")

        plt.xlabel("Total Spend")
        plt.ylabel("Count")
        plt.title("Total Spend by Churn")
        plt.legend()
        plt.tight_layout()
        plt.show()


def categorical_churn(df: pd.DataFrame) -> None:
    """
    For each categorical feature, show counts of churn vs no churn
    with color-coded bars.
    """

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(
            x=col,
            hue="Churn",
            data=df,
            palette={0: "green", 1: "red"},
        )
        plt.title(f"Churn by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title="Churn", labels=["No (0)", "Yes (1)"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numeric features (including Churn)."""

    numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_clean.csv")
    churn_distribution(df)
    numeric_histograms(df)
    categorical_churn(df)
    correlation_heatmap(df)