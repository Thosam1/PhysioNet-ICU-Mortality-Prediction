from PIL.ImageChops import offset
from sklearn.preprocessing import StandardScaler
from data_preprocessing.data_imputation import forward_fill_then_median
import pandas as pd

def aggregate_patient_data_basic(X, person_id_column, method):
    """
    Aggregates the time-series data for each patient using the specified method.

    Parameters:
    X (pd.DataFrame): The input dataset containing time-series features.
    person_id_column (str): The name of the column that identifies individuals.
    method (str): Aggregation method to use. Options:
                  - "mean": Computes the mean of each feature per patient.
                  - "max": Computes the max of each feature per patient.
                  - "last": Takes the last available value of each feature per patient.

    Returns:
    pd.DataFrame or tuple: Depending on the selected method, returns one or multiple aggregated DataFrames.
    """
    grouped = X.groupby(person_id_column)

    if method == "mean":
        return grouped.mean(numeric_only=True).reset_index()
    elif method == "max":
        return grouped.max(numeric_only=True).reset_index()
    elif method == "last":
        return grouped.last().reset_index()
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'max', 'last'.")


def aggregate_patient_data_advanced(X, person_id_column):
    """
    Aggregates the time-series data for each patient using multiple aggregation methods:
    - Dynamic features: first, last, lowest, highest, median.
    - Lab test features: first, last.

    Parameters:
    X (pd.DataFrame): The input dataset containing time-series features.
    person_id_column (str): The name of the column that identifies individuals.

    Returns:
    pd.DataFrame: Aggregated dataset with one row per patient.
    """

    # Define feature groups
    dynamic_feats = [
        'DiasABP', 'GCS', 'Glucose', 'HR', 'MAP',
        'NIDiasABP', 'NIMAP', 'NISysABP',
        'RespRate', 'SaO2', 'Temp'
    ]

    lab_feats = [
        'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
        'Creatinine', 'FiO2', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na',
        'PaCO2', 'PaO2', 'pH', 'Platelets', 'SysABP', 'TroponinI',
        'TroponinT', 'WBC', 'Weight'
    ]

    # Define aggregation methods for dynamic features (first, last, lowest, highest, median)
    agg_methods = {
        "first": "first",
        "last": "last",
        "lowest": "min",
        "highest": "max",
        "median": "median"
    }

    # Start by grouping the data by person_id_column
    grouped = X.groupby(person_id_column)

    # Initialize an empty list to store the aggregated data for each patient
    aggregated_data = []

    # Aggregate dynamic features using the defined aggregation methods
    for patient_id, group in grouped:
        patient_data = {}

        print("Patient ID:", patient_id)
        # Aggregating dynamic features
        for feature in dynamic_feats:
            for agg_name, agg_func in agg_methods.items():
                agg_col_name = f"{feature}_{agg_name}"
                patient_data[agg_col_name] = group[feature].first()

                print(group[feature])

        # Aggregating lab features (first and last only)
        for feature in lab_feats:
            patient_data[f"{feature}_first"] = group[feature].first()
            patient_data[f"{feature}_last"] = group[feature].last().reset_index()

        # Add the patient's aggregated data to the list
        aggregated_data.append(patient_data)

    # Convert the list of aggregated data into a DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    return aggregated_df

def preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test, person_id_column):
    """
    Preprocesses patient data by:
    - Dropping unnecessary columns ('time', 'ICUType')
    - Forward filling missing values within each person's data
    - Replacing remaining NaNs with column medians from the training set

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_valid (pd.DataFrame): Validation dataset
    X_test (pd.DataFrame): Test dataset
    person_id_column (str): Column that identifies unique patients

    Returns:
    tuple: Preprocessed X_train, X_valid, X_test datasets with missing values handled.
    """

    # Step 1: Drop unnecessary columns
    drop_cols = ["time", "ICUType"]
    X_train = X_train.drop(columns=drop_cols)
    X_valid = X_valid.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # Step 2: Handle missing values
    X_train, train_medians = forward_fill_then_median(X_train, person_id_column)
    X_valid, _ = forward_fill_then_median(X_valid, person_id_column, medians=train_medians)
    X_test, _ = forward_fill_then_median(X_test, person_id_column, medians=train_medians)

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

    # Step 1 & 2: Preprocess the data
    X_train, X_valid, X_test = preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test, person_id_column="recordid")

    # Step 3: Aggregate the data based on the selected method
    X_train = aggregate_patient_data_basic(X_train, person_id_column="recordid", method=method)
    X_valid = aggregate_patient_data_basic(X_valid, person_id_column="recordid", method=method)
    X_test = aggregate_patient_data_basic(X_test, person_id_column="recordid", method=method)

    # Step 4: Drop the patient ID column (do this on copies to avoid modifying the original DataFrames)
    X_train = X_train.drop(columns=["recordid"]).copy()
    X_valid = X_valid.drop(columns=["recordid"]).copy()
    X_test = X_test.drop(columns=["recordid"]).copy()

    # Step 5: Scale the aggregated data using StandardScaler
    scaler = StandardScaler()
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, scaler)

    return X_train, X_valid, X_test

def prepare_data_advanced_for_ML_classifier(X_train, X_valid, X_test):
    """
    Prepares the data for machine learning classification by:
    - Dropping unnecessary columns
    - Forward filling missing values and replacing NaNs with column medians
    - Aggregating the data using an advanced aggregation method
    - Scaling the data using StandardScaler

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_valid (pd.DataFrame): Validation dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    tuple: Scaled datasets (X_train, X_valid, X_test), each aggregated using an advanced method.
    """

    # Step 1: Preprocess patient data
    X_train, X_valid, X_test = preprocess_patient_data_for_ML_classifier(X_train, X_valid, X_test,
                                                                         person_id_column="recordid")

    # Step 2: Aggregate the data using the advanced method
    X_train = aggregate_patient_data_advanced(X_train, person_id_column="recordid")
    X_valid = aggregate_patient_data_advanced(X_valid, person_id_column="recordid")
    X_test = aggregate_patient_data_advanced(X_test, person_id_column="recordid")

    # Step 3: Drop patient ID column (if still present)
    X_train = X_train.drop(columns=["recordid"], errors="ignore")
    X_valid = X_valid.drop(columns=["recordid"], errors="ignore")
    X_test = X_test.drop(columns=["recordid"], errors="ignore")

    # Step 4: Scale data using StandardScaler
    scaler = StandardScaler()
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

    return X_train_scaled, X_valid_scaled, X_test_scaled

