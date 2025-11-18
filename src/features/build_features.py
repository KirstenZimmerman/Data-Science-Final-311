import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # Drop rows with null values
    transaction_data_cleaned = transaction_data_raw.dropna().copy()
    print(transaction_data_cleaned.shape)



if __name__ == '__main__':
    main()
