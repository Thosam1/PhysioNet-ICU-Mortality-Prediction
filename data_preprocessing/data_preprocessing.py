from PIL.ImageChops import offset
from sklearn.preprocessing import StandardScaler
from data_preprocessing.data_imputation import forward_fill_then_median
import pandas as pd
from sklearn.preprocessing import RobustScaler

from data_preprocessing.data_imputation import forward_fill_then_median

def aggregate_patient_data_basic(X, method):
    """
    Aggregates the time-series data for each patient using the specified method.

    Parameters:
    X (pd.DataFrame): The input dataset containing time-series features.
    method (str): Aggregation method to use. Options:
                  - "mean": Computes the mean of each feature per patient.
                  - "max": Computes the max of each feature per patient.
                  - "last": Takes the last available value of each feature per patient.

    Returns:
    pd.DataFrame or tuple: Depending on the selected method, returns one or multiple aggregated DataFrames.
    """
    grouped = X.groupby('recordid')

    if method == "mean":
        return grouped.mean(numeric_only=True).reset_index()
    elif method == "max":
        return grouped.max(numeric_only=True).reset_index()
    elif method == "last":
        return grouped.last().reset_index()
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'max', 'last'.")


def aggregate_patient_data_advanced(X):
    """
    Aggregates the time-series data for each patient using multiple aggregation methods:
    - Dynamic features: first, last, lowest, highest, median.
    - Lab test features: first, last.

    Parameters:
    X (pd.DataFrame): The input dataset containing time-series features.

    Returns:
    pd.DataFrame: Aggregated dataset with one row per patient.
    """

    # Define feature groups
    # 11
    dynamic_feats = [
        'DiasABP', 'GCS', 'Glucose', 'HR', 'MAP',
        'NIDiasABP', 'NIMAP', 'NISysABP',
        'RespRate', 'SaO2', 'Temp'
    ]

    # 24
    lab_feats = [
        'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
        'Creatinine', 'FiO2', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na',
        'PaCO2', 'PaO2', 'pH', 'Platelets', 'SysABP', 'TroponinI',
        'TroponinT', 'WBC', 'Weight', 'Urine'
    ]

    static_feats = [
        'Age',
        'Gender',
        'Height',
        'MechVent',
    ]

    removed_feats = [
        'MechVent',
    ]

    # Define aggregation methods for dynamic features (first, last, lowest, highest, median)
    agg_methods = {
        "first": "first",
        "last": "last",
        "lowest": "min",
        "highest": "max",
        "median": "median"
    }

    # Start by grouping the data
    grouped = X.groupby('recordid')

    # Dictionary to store all aggregated features
    aggregated_features = {}

    # Aggregate dynamic features using different methods
    for feature in dynamic_feats:
        for agg_name, agg_func in agg_methods.items():
            agg_col_name = f"{feature}_{agg_name}"
            aggregated_features[agg_col_name] = grouped[feature].agg(agg_func)

    # Aggregate lab features (first and last)
    for feature in lab_feats:
        aggregated_features[f"{feature}_first"] = grouped[feature].first()
        aggregated_features[f"{feature}_last"] = grouped[feature].last()

    # Aggregate static features (only one value per patient)
    for feature in static_feats:
        aggregated_features[feature] = grouped[feature].last()

    # Convert the dictionary into a DataFrame
    aggregated_df = pd.DataFrame(aggregated_features)

    # Reset index to bring 'recordid' as a column (if needed)
    aggregated_df.reset_index(inplace=True)

    return aggregated_df

def preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test):
    """
    Preprocesses patient data by:
    - Dropping unnecessary columns ('time', 'ICUType')
    - Forward filling missing values within each person's data
    - Replacing remaining NaNs with column medians from the training set

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_valid (pd.DataFrame): Validation dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    tuple: Preprocessed X_train, X_valid, X_test datasets with missing values handled.
    """

    # Step 1: Drop unnecessary columns
    drop_cols = ["time", "ICUType"]
    X_train = X_train.drop(columns=drop_cols)
    X_valid = X_valid.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # Step 2: Handle missing values
    X_train, train_medians = forward_fill_then_median(X_train)
    X_valid, _ = forward_fill_then_median(X_valid, medians=train_medians)
    X_test, _ = forward_fill_then_median(X_test, medians=train_medians)

    return X_train, X_valid, X_test

def prepare_data_basic_for_ML_classifier(X_train, X_valid, X_test, method="mean"):
    """
    Prepares the data for machine learning classification by:
    - Dropping unnecessary columns
    - Forward filling missing values and replacing NaNs with column medians
    - Aggregating the data using the specified method ('mean', 'max', or 'last')
    - Scaling the data using StandardScaler

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_valid (pd.DataFrame): Validation dataset
    X_test (pd.DataFrame): Test dataset
    method (str): Aggregation method, one of {'mean', 'max', 'last'}. Default is 'mean'.

    Returns:
    tuple: Scaled datasets (X_train, X_valid, X_test), each aggregated using the specified method.
    """

    # Validate the method argument
    if method not in {"mean", "max", "last"}:
        raise ValueError("Invalid method. Choose from 'mean', 'max', or 'last'.")

    # Preprocess the data
    X_train, X_valid, X_test = preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test)

    # Aggregate the data based on the selected method
    X_train = aggregate_patient_data_basic(X_train, method=method)
    X_valid = aggregate_patient_data_basic(X_valid, method=method)
    X_test = aggregate_patient_data_basic(X_test, method=method)

    # Drop the patient ID column (do this on copies to avoid modifying the original DataFrames)
    X_train = X_train.copy().drop(columns=["recordid"])
    X_valid = X_valid.copy().drop(columns=["recordid"])
    X_test = X_test.copy().drop(columns=["recordid"])

    # TODO
    # Scale the aggregated data using RobustScaler
    scaler = RobustScaler()
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, scaler)

    return X_train, X_valid, X_test

def prepare_data_advanced_for_ML_classifier(X_train, X_valid, X_test):
    """
    Prepares the data for machine learning classification by:
    - Dropping unnecessary columns
    - Forward filling missing values and replacing NaNs with column medians
    - Aggregating the data using an advanced aggregation method
    - Scaling the data using RobustScaler

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_valid (pd.DataFrame): Validation dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    tuple: Scaled datasets (X_train, X_valid, X_test), each aggregated using an advanced method.
    """

    # Preprocess patient data
    X_train, X_valid, X_test = preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test)

    # Aggregate the data using the advanced method
    X_train = aggregate_patient_data_advanced(X_train)
    X_valid = aggregate_patient_data_advanced(X_valid)
    X_test = aggregate_patient_data_advanced(X_test)

    # Drop patient ID column (if still present)
    X_train = X_train.copy().drop(columns=["recordid"])
    X_valid = X_valid.copy().drop(columns=["recordid"])
    X_test = X_test.copy().drop(columns=["recordid"])

    # Scale data using RobustScaler
    scaler = RobustScaler()
    return scale_data(X_train, X_valid, X_test, scaler)

def scale_data(X_train, X_valid, X_test, scaler):
    """
    Scales the training, validation, and test datasets using the provided scaler.

    Parameters:
    X_train (pd.DataFrame): The training dataset
    X_valid (pd.DataFrame): The validation dataset
    X_test (pd.DataFrame): The test dataset
    scaler (sklearn.preprocessing.RobustScaler): The scaler to use for transformation

    Returns:
    tuple: Scaled datasets for training, validation, and test sets
    """
    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the validation and test data using the same scaler
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_valid_scaled, X_test_scaled

