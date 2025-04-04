import pandas as pd
import numpy as np

def load_and_clean_data(x_paths, y_paths, clean_func):
    """
    Load, clean, and sort datasets for training, validation, and testing.

    Parameters:
    - x_paths (list): List of file paths for the input feature datasets.
    - y_paths (list): List of file paths for the outcome datasets.
    - clean_func (function): Function to clean the data.

    Returns:
    - X_train, X_valid, X_test: Sorted and cleaned feature datasets.
    - y_train, y_valid, y_test: Sorted outcome datasets as NumPy arrays.
    """
    X_sets = [pd.read_parquet(path) for path in x_paths]
    Y_sets = [pd.read_parquet(path) for path in y_paths]

    if '' in X_sets[-1].columns:
        X_sets[-1] = X_sets[-1].drop(columns=[''])

    X_sets = [clean_func(X) for X in X_sets]
    X_train, X_valid, X_test = [X.sort_values(by=['recordid', 'time']) for X in X_sets]
    y_train, y_valid, y_test = [np.array(Y.sort_values(by="recordid").drop(columns=["recordid"])).ravel() for Y in Y_sets]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
