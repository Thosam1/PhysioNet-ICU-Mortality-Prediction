import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest, StackingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score

def parse_data(dataset):
    """
    Parses the data from the given dataset directory and saves it as a parquet file.

    Parameters:
    dataset (str): The path to the dataset directory.

    Returns:
    None
    """
    filename = dataset[5:] + '.parquet'

    # If the file already exists, delete it and create a new one
    if os.path.exists(filename):
        os.remove(filename)

    # Read and compile all files in the directory
    txt_all = []
    for f in os.listdir(dataset):
        with open(os.path.join(dataset, f), 'r') as fp:
            txt = fp.readlines()

        # Get recordid to add as a column
        recordid = txt[1].rstrip('\n').split(',')[-1]
        txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
        txt_all.extend(txt[1:])

    # Create DataFrame
    df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])
    df = df[df.parameter != 'RecordID']

    # Convert 'time' to hours and minutes and then to total minutes
    def time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    df['time_minutes'] = df['time'].apply(time_to_minutes)

    # Round up to the nearest hour
    df['time'] = ((df['time_minutes'] + 59) // 60).clip(upper=48)

    # Keep the most recent value by sorting and dropping duplicates
    df.sort_values(by=['recordid', 'time', 'time_minutes'], ascending=[True, True, False], inplace=True)
    df.drop_duplicates(subset=['recordid', 'time', 'parameter'], keep='first', inplace=True)

    # Pivot the DataFrame
    df_pivot = df.pivot_table(index=['recordid', 'time'], columns='parameter', values='value',
                              aggfunc='first').reset_index()

    # Ensure all patients have 49 rows (0 to 48 hours)
    all_hours = pd.DataFrame({'time': range(49)})

    patients = df['recordid'].unique()
    patient_hours = pd.MultiIndex.from_product([patients, all_hours['time']], names=['recordid', 'time'])

    # Reindex to include all patient-hour combinations
    final_df = df_pivot.set_index(['recordid', 'time']).reindex(patient_hours).reset_index()

    # Convert time column into timesteps
    final_df['time'] = final_df['time'].apply(lambda x: f'{x:02}:00')

    static_columns = ["Age", "Gender", "Height", "ICUType", "Weight"]

    # Remove -1 values and propagate the static values to all the rows
    for col in static_columns:
        final_df[col] = final_df[col].astype(float).replace(-1, np.nan)
        final_df[col] = final_df.groupby('recordid')[col].transform(lambda x: x.ffill().bfill())

    for col in final_df.columns:
        try:
            final_df[col] = pd.to_numeric(final_df[col], errors='raise')
            # If all values are integers, convert to int
            if final_df[col].apply(float.is_integer).all():
                final_df[col] = final_df[col].astype(int)
        except:
            pass  # Keep as is if conversion fails

    # Fill None missing values with NaN
    final_df = final_df.fillna(value=np.nan)

    final_df.to_parquet(filename, index=False)


def extract_labels(outcome):
    """
    Extracts the labels from the given outcome file and saves it as a parquet file.

    Parameters:
    outcome (str): The path to the outcome file.

    Returns:
    None
    """
    filename = outcome[5:-4].lower() + '.parquet'

    # If the file already exists, delete it and create a new one
    if os.path.exists(filename):
        os.remove(filename)

    with open(outcome, 'r') as fp:
        txt = fp.readlines()

    labels = []
    for line in txt[1:]:  # Skip header
        parts = line.strip().split(',')
        recordid = int(parts[0])
        in_hospital_death = int(parts[-1])
        labels.append((recordid, in_hospital_death))

    labels_df = pd.DataFrame(labels, columns=['recordid', 'in_hospital_death'])
    labels_df.to_parquet(filename, index=False)