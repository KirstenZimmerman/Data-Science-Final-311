import pandas as pd
import numpy as np
import os


def main():
    # Load the Credit Card Fraud Detection dataset (assuming it's in the same directory as this notebook)
    file_path = "card_transdata.csv"
    transaction_data_raw = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset and general information
    transaction_data_raw.head()



if __name__ == '__main__':
    main()
