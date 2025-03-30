import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def forward_fill_then_median(X_set_a, medians=None):
    """
    Applies forward fill imputation and replaces remaining NaNs with medians for each column.
    Optionally, if medians are provided, uses them to fill missing values.

    Parameters:
    X_set_a (pd.DataFrame): The input dataset with missing values.
    medians (dict, optional): A dictionary with the medians for each column to fill missing values.

    Returns:
    pd.DataFrame: The dataset with missing values imputed.
    dict: The medians used for filling (if medians were provided, otherwise the medians are computed).
    """
    # Forward fill within each person's data
    X_filled = X_set_a.groupby('recordid').apply(lambda group: group.ffill()).reset_index(drop=True)

    # If medians are provided, use them to fill missing values, otherwise compute medians
    if medians is not None:
        X_filled = X_filled.fillna(medians)
    else:
        medians = X_filled.median()
        X_filled = X_filled.fillna(medians)

    return X_filled, medians