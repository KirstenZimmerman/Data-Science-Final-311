import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def train_models(df: pd.DataFrame):
    """Train baseline and k-NN models and return predictions."""
    X = df.drop('fraud', axis=1)
    y = df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline model: never predict fraud
    y_pred_baseline = [0] * len(y_test)

    # k-NN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    return y_test, y_pred_baseline, y_pred_knn


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    y_test, baseline, knn = train_models(clean)
    print("Baseline sample:", baseline[:5])
    print("k-NN sample:", knn[:5])
