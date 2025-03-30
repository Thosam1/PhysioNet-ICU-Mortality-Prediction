import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def forward_fill(X_set_a, person_id_column):
    """
    Applies forward fill imputation within each person's data.
    This step fills missing values based on the previous known value.

    Parameters:
    X_set_a (pd.DataFrame): The input dataset with missing values.
    person_id_column (str): The name of the column that identifies individuals.

    Returns:
    pd.DataFrame: The dataset with missing values forward-filled.
    """
    # Forward fill within each person's data
    return X_set_a.groupby(person_id_column).apply(lambda group: group.ffill()).reset_index(drop=True)

def fill_with_median(X_set_a, medians):
    """
    Fills missing values with the provided medians for each column.

    Parameters:
    X_set_a (pd.DataFrame): The input dataset with missing values.
    medians (dict): A dictionary of column medians to use for filling missing values.

    Returns:
    pd.DataFrame: The dataset with missing values filled using medians.
    """
    return X_set_a.fillna(medians)

def get_column_medians(X_set_a):
    """
    Computes the medians for each column in the dataset.

    Parameters:
    X_set_a (pd.DataFrame): The input dataset.

    Returns:
    dict: A dictionary with the medians of each column.
    """
    return X_set_a.median()

def forward_fill_then_median(X_set_a, person_id_column, medians=None):
    """
    Applies forward fill imputation and replaces remaining NaNs with medians for each column.
    Optionally, if medians are provided, uses them to fill missing values.

    Parameters:
    X_set_a (pd.DataFrame): The input dataset with missing values.
    person_id_column (str): The name of the column that identifies individuals.
    medians (dict, optional): A dictionary with the medians for each column to fill missing values.

    Returns:
    pd.DataFrame: The dataset with missing values imputed.
    dict: The medians used for filling (if medians were provided, otherwise the medians are computed).
    """
    # Step 1: Forward fill within each person's data
    X_filled = forward_fill(X_set_a, person_id_column)

    # Step 2: If medians are provided, use them to fill missing values, otherwise compute medians
    if medians is not None:
        X_filled = fill_with_median(X_filled, medians)
    else:
        medians = get_column_medians(X_filled)
        X_filled = fill_with_median(X_filled, medians)

    return X_filled, medians