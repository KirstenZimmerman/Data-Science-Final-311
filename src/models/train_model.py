import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # For working with data
    import pandas as pd
    
    # For visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    
    # For machine learning modeling
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

    # Select features and target variable
    X = transaction_data_cleaned.drop('fraud', axis=1)
    y = transaction_data_cleaned['fraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred_knn = knn.predict(X_test)
    
    # Print just the first 100 predictions
    print(y_pred_knn[:100]) # Notice how some predictions are 1, fradulent!



if __name__ == '__main__':
    main()
