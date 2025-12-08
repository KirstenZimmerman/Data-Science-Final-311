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
    │   ├── load_data.py # Load raw train/test CSVs
    │   ├── preprocess.py # Data cleaning, imputation, type conversion
    │   └── split_data.py # train/validation split utilities

    ├── models/
    │   ├── dumb_model.py # Truly dumb baseline (predicts no churn)
    │   ├── knn_model.py # KNN pipeline (OneHot + scaling)
    │   ├── decisionTree_model.py # Tuned Decision Tree pipeline
    │   └── randomForestClassifier_model.py # Random Forest pipeline

    ├── utils/
    │   └── helper_functions.py # ROC curve plotting and shared helpers

    └── visualization/
        ├── eda.py # Exploratory data analysis plots
        └── performance.py # Confusion matrices & metric comparison
```

# Models Implemented
1. Dumb Baseline (Predict No Churn)
    - Implemented using `DummyClassifier(strategy="constant", constant=0)`
   - Always predicts “no churn”
   - Serves as a true lower-bound benchmark, matching the course baseline

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


## Running the project

Install the dependencies and run the pipeline.

The dependecies are: numpy, pandas, scikit-learn, matplotlib, and seaborn

You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n final_project --file requirements.txt
conda activate final_project
python main.py
```
Once you do this the dataset will be loaded in and prepocesd. First, the EDA will be displayed. After this, the feautures will be builts, my data split into training and validation sets, and my three models will be trained: my eda model, my knn model, and my decision tree model. After this, the rsults of the mdoels will be visualized including their confusion matrix, four meausring metrics for models, and their ROC-AUC curve. After this, the ROC-AUC curve and its confusion matrix of the best model will be shown.

The cleaned data will be written to `data/processed/` and all plots will be displayed interactively.