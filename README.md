# Customer Churn Prediction – MAT 311 Final Project

This repository contains my end-to-end machine learning workflow for predicting customer churn as part of the MAT 311: Introduction to Data Science final project. The goal is to prepare the dataset, explore key patterns, engineer features, build multiple predictive models, and generate a submission for a Kaggle competition evaluated using ROC AUC.

## Purpose

The project follows a production-style organizational structure modeled after industry best practices and the course guidelines. Each step—data loading, cleaning, feature engineering, modeling, evaluation, and prediction—is separated into modular scripts under src/ for clear reproducibility.

Jupyter notebooks serve only for EDA and experimentation, while main.py runs the complete pipeline.

This project can be used both to fulfill course requirements and as a well-organized portfolio example demonstrating end-to-end machine learning workflow skills

## Project layout

```
.
├── main.py                     # Runs full churn-prediction pipeline
├── requirements.txt            # Python environment dependencies

├── data/
│   ├── raw/                    # Original files 
│   ├── processed/              # Cleaned + imputed training and test data
│   └── submissions/            # Kaggle-ready prediction files

├── notebooks/
│   ├── eda_churn.ipynb         
│   └── model_experiments.ipynb # Baseline, KNN, and Decision Tree

└── src/
    ├── data/
    │   ├── load_data.py        # Loads raw data into pandas DataFrames
    │   ├── clean_data.py       # Handles missing values, type conversion, and feature creation
    │   └── split_data.py       # Train/validation split utilities

    ├── features/
    │   └── build_features.py   # Constructs feature lists, encoders, and transformers

    ├── models/
    │   ├── baseline.py         # Custom baseline churn classifier
    │   ├── knn_model.py        # KNN model pipeline
    │   ├── decision_tree.py    # Optimized DecisionTreeClassifier    
    │   └── random_forest.py    


    ├── utils/
    │   └── helpers.py         

    └── visualization/
        ├── eda.py              # plots for distributions, correlations, class balance
        └── performance.py      # Confusion matrices, ROC curves, and model comparison
```

# Models Implemented
1. Baseline Classifier (Custom Rule-Based Model) - A model using insights from EDA, incorporating features like:
    - Contract Length
    - Total Spend
    - Support Calls
    - Tenure
    - Age
    - Usage Frequency
    - Last Interaction
Used to establish a minimum performance benchmark.

2. K-Nearest Neighbors (KNN)
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    - Tuned across multiple neighbor values
    - Achieved strong validation ROC AUC

3. Decision Tree
    - OneHotEncoder + StandardScaler preprocessing
    - Tuned hyperparameters such as
        - max_depth
        - min_samples_leaf
    - Used for Kaggle submissions

4. Random Forest Classifier - An ensemble model used to improve performance over a single decision tree.
    - Tuned parameters such as:
        - n_estimators
        - max_depth
        - min_samples_leaf


# Notes
- "Payment Delay" was removed from modeling because it creates data leakage.
- Missing values were filled using a combination of median and zeros.
- All models are wrapped in full scikit-learn pipelines for reproducibility.