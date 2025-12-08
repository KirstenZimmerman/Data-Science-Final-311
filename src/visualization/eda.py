import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def churn_distribution(df: pd.DataFrame) -> None:
    """Plot overall churn class distribution and print normalized rates."""
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print("Churn rate (normalized):")
    print(df["Churn"].value_counts(normalize=True))
    print()


def numeric_histograms(df: pd.DataFrame) -> None:
    """
    Show numeric feature distributions and churn-related numeric views:
    - overall histograms for all numeric columns
    - non-zero distributions for Support Calls and Last Interaction
    - boxplots of each numeric feature by churn
    - line plots of selected numeric features by churn
    - special binned line plot for Total Spend by churn
    """
    # All numeric columns (including nullable Int64)
    numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns

    # 1) Overall distributions
    df[numeric_cols].hist(figsize=(15, 12), bins=30)
    plt.tight_layout()
    plt.show()

    # 2) Non-zero distributions for Support Calls and Last Interaction
    for col in ["Support Calls", "Last Interaction"]:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[df[col] > 0][col], bins=30)
            plt.title(f"{col} (Non-zero Distribution)")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

    # 3) Boxplots of each numeric feature by churn
    for col in numeric_cols:
        if col == "Churn":
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Churn", y=col, data=df)
        plt.title(f"{col} vs Churn")
        plt.xlabel("Churn (0 = No, 1 = Yes)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    # 4) Line plots of selected numeric features by churn
    other_cols = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Last Interaction",
    ]

    for col in other_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(6, 4))

        grouped = (
            df.groupby([col, "Churn"])
            .size()
            .reset_index(name="count")
        )

        for churn_value, color, label in [(0, "green", "No Churn"), (1, "red", "Churn")]:
            subset = grouped[grouped["Churn"] == churn_value]
            plt.plot(
                subset[col],
                subset["count"],
                color=color,
                label=label,
            )

        plt.title(f"{col} by Churn")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 5) Special: Total Spend by churn with binned x-axis
    if "Total Spend" in df.columns:
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
    For each categorical feature, show countplot with bars color-coded by churn.
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

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_clean.csv")
    churn_distribution(df)
    numeric_histograms(df)
    categorical_churn(df)
   
