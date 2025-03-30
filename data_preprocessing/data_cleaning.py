import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Functions for data cleaning
# Inspired by https://github.com/alistairewj/challenge2012/blob/master/prepare-data.ipynb
# and by feature descriptions in https://physionet.org/content/challenge-2012/1.0.0/

def delete_value(df, c, value=0):
    """
    Replaces specified values in a given column of the dataframe with NaN.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    c (str): The column name where the value needs to be replaced.
    value (numeric, default=0): The value to be replaced with NaN.

    Returns:
    pandas.DataFrame: The dataframe with the specified value replaced with NaN.
    """
    if c in df.columns:
        df[c] = df[c].replace(value, np.nan)
    return df


def replace_value(df, c, value=np.nan, below=None, above=None):
    """
    Replaces values in a specified column based on conditions of the column's values.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    c (str): The column name to modify.
    value (numeric or function, default=np.nan): The value to replace with.
    below (numeric, optional): Replace values below this threshold.
    above (numeric, optional): Replace values above this threshold.

    Returns:
    pandas.DataFrame: The dataframe with updated values in the specified column.
    """
    # Identify rows where the condition holds (value is below or above specified thresholds)
    idx = (df[c].notna()) & ((below is not None and df[c] < below) | (above is not None and df[c] > above))

    # Apply the replacement (use a callable function or direct value replacement)
    df.loc[idx, c] = df.loc[idx, c].apply(value) if callable(value) else value
    return df


def clean_data(df):
    """
    Cleans the data by applying various rules for handling outliers and missing values.
    These rules are based on known constraints and ranges for medical parameters.

    Parameters:
    df (pandas.DataFrame): The dataframe to be cleaned.

    Returns:
    pandas.DataFrame: The cleaned dataframe with updated values for medical parameters.
    """
    # Replace -1 values in 'DiasABP' with NaN
    df = delete_value(df, 'DiasABP', -1)
    df = replace_value(df, 'DiasABP', value=np.nan, below=1)
    df = replace_value(df, 'DiasABP', value=np.nan, above=200)

    # Handle FiO2 (Fraction of Inspired Oxygen)
    df = replace_value(df, 'FiO2', value=np.nan, below=0, above=1)

    # Handle GCS (Glasgow Coma Scale)
    df = replace_value(df, 'GCS', value=np.nan, below=3, above=15)

    # Handle Hematocrit (HCT)
    df = replace_value(df, 'HCT', value=np.nan, below=0, above=100)

    # Handle Height (assumes units need conversion)
    df = replace_value(df, 'Height', value=np.nan, below=0)
    df = replace_value(df, 'Height', value=lambda x: x * 100, below=10)  # Convert meters to cm
    df = replace_value(df, 'Height', value=lambda x: x * 10, below=25)  # Convert meters to mm
    df = replace_value(df, 'Height', value=lambda x: x * 2.54, below=100)  # Convert inches to cm
    df = replace_value(df, 'Height', value=lambda x: x * 0.1, above=1000)  # Convert mm to cm
    df = replace_value(df, 'Height', value=lambda x: x * 0.3937, above=250)  # Convert cm to inches

    # Handle Heart Rate (HR)
    df = replace_value(df, 'HR', value=np.nan, below=1)
    df = replace_value(df, 'HR', value=np.nan, above=299)

    # Handle Mean Arterial Pressure (MAP)
    df = replace_value(df, 'MAP', value=np.nan, below=1)

    # Handle Blood Pressure readings: NIDiasABP, NIMAP, NISysABP
    df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
    df = replace_value(df, 'NIMAP', value=np.nan, below=1)
    df = replace_value(df, 'NISysABP', value=np.nan, below=1)

    # Handle PaCO2 (Partial Pressure of Carbon Dioxide in Arterial Blood)
    df = replace_value(df, 'PaCO2', value=np.nan, below=1)
    df = replace_value(df, 'PaCO2', value=lambda x: x * 10, below=10)

    # Handle PaO2 (Partial Pressure of Oxygen in Arterial Blood)
    df = replace_value(df, 'PaO2', value=np.nan, below=1)
    df = replace_value(df, 'PaO2', value=lambda x: x * 10, below=20)

    # Handle pH (Blood pH level)
    df = replace_value(df, 'pH', value=np.nan, below=6.5)
    df = replace_value(df, 'pH', value=np.nan, above=8.0)

    # Handle Respiratory Rate (RespRate)
    df = replace_value(df, 'RespRate', value=np.nan, below=1)

    # Handle Oxygen Saturation (SaO2)
    df = replace_value(df, 'SaO2', value=np.nan, below=1, above=100)

    # Handle Systolic Blood Pressure (SysABP)
    df = replace_value(df, 'SysABP', value=np.nan, below=1)

    # Handle Body Temperature (Temp) with unit conversions
    df = replace_value(df, 'Temp', value=lambda x: x * 9 / 5 + 32, below=10, above=1)  # Convert Celsius to Fahrenheit
    df = replace_value(df, 'Temp', value=lambda x: (x - 32) * 5 / 9, below=113,
                       above=95)  # Convert Fahrenheit to Celsius

    # Handle invalid temperature values
    df = replace_value(df, 'Temp', value=np.nan, below=25)
    df = replace_value(df, 'Temp', value=np.nan, above=45)

    # Handle White Blood Cell count (WBC)
    df = replace_value(df, 'WBC', value=np.nan, below=1)

    # Handle Weight
    df = replace_value(df, 'Weight', value=np.nan, below=35)
    df = replace_value(df, 'Weight', value=np.nan, above=299)

    return df