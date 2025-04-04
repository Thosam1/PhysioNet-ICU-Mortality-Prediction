import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distributions(df, static_columns):
    """
    Plots the distribution of all columns in the DataFrame.
    """
    exclude_cols = ['time', 'recordid']

    cols_to_plot = [col for col in df.columns if col not in exclude_cols]
    num_cols = len(cols_to_plot)

    plt.figure(figsize=(15, 5 * (num_cols // 3 + 1)))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot((num_cols // 3) + 1, 3, i)
        if col in static_columns:  # Static variables should be taken once per record
            if col in ['Gender', 'ICUType']:  # Categorical and static variables
                bins = len(df[col].dropna().unique())  # Unique categories
                sns.histplot(df.groupby('recordid')[col].first(), bins=bins, color='blue')
            else:  # Other static numerical variables
                sns.histplot(df.groupby('recordid')[col].first(), kde=True, color='blue')
        elif col in ['GCS', 'MechVent']:  # Categorical time-varying variables
            bins = len(df[col].dropna().unique())
            sns.histplot(df[col], bins=bins, color='blue')
        else:
            sns.histplot(df[col], kde=True, color='blue')
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()

def plot_distributions_by_gender(df, static_columns):
    """
    Plots the distribution of all columns in the DataFrame, grouped by Gender.
    """
    exclude_cols = ['time', 'recordid', 'Gender']
    cols_to_plot = [col for col in df.columns if col not in exclude_cols]
    num_cols = len(cols_to_plot)

    plt.figure(figsize=(15, 5 * (num_cols // 3 + 1)))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot((num_cols // 3) + 1, 3, i)
        if col in static_columns:
            sns.histplot(data=df.groupby('recordid')[[col, 'Gender']].first(),
                         x=col, hue='Gender', stat='probability', common_norm=False)
        else:
            sns.histplot(data=df, x=col, hue='Gender', stat='probability', common_norm=False)
        plt.title(f"Distribution of {col} by Gender")

    plt.tight_layout()
    plt.show()

def plot_distributions_by_age_bins(df, static_columns):
    """
    Plots the distribution of all columns in the DataFrame, grouped by Age Bins.
    """
    df['age_bins'] = pd.cut(df['Age'], bins=np.arange(0, 100, 10))
    df['age_bins'] = df['age_bins'].astype(str)

    exclude_cols = ['time', 'recordid', 'Age', 'age_bins']
    cols_to_plot = [col for col in df.columns if col not in exclude_cols]
    num_cols = len(cols_to_plot)

    plt.figure(figsize=(15, 5 * (num_cols // 3 + 1)))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(num_cols // 3 + 1, 3, i)
        if col in static_columns:
            sns.histplot(data=df.groupby('recordid')[[col, 'age_bins']].first(),
                         x=col, hue='age_bins', stat='probability', common_norm=False, element="step")
        else:
            sns.histplot(data=df, x=col, hue='age_bins', stat='probability', common_norm=False, element="step")
        plt.title(f"Distribution of {col} by Age Bins")

    plt.tight_layout()
    plt.show()

    df.drop(columns=["age_bins"], inplace=True)

def plot_cholesterol_by_age_bins(df, age_col="Age", cholesterol_col="Cholesterol"):
    """
    Plots a boxplot and mean pointplot of cholesterol levels grouped by age bins.

    Parameters:
    - df (pd.DataFrame): The dataset containing age and cholesterol values.
    - age_col (str): The column representing age.
    - cholesterol_col (str): The column representing cholesterol levels.
    """
    df = df.copy()  # Avoid modifying the original dataframe
    df['age_bins'] = pd.cut(df[age_col], bins=np.arange(0, 100, 10))
    df['age_bins'] = df['age_bins'].astype(str)

    plt.figure(figsize=(10, 6))

    # Boxplot for cholesterol levels by age bins
    sns.boxplot(data=df, x='age_bins', y=cholesterol_col, hue='age_bins', palette="coolwarm", showfliers=False)

    # Pointplot for the mean cholesterol levels by age bins
    sns.pointplot(data=df, x='age_bins', y=cholesterol_col, color='red', estimator=np.mean,
                  errorbar=None, markers='o', linestyles='')

    plt.title(f"{cholesterol_col} Levels by Age Bins")
    plt.xlabel("Age Bins")
    plt.ylabel(cholesterol_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df.drop(columns=["age_bins"], inplace=True)

def plot_distributions_by_icu_type(df, static_columns):
    """
    Plots distributions of numeric variables in the dataset, grouped by ICU type.
    """
    exclude_cols = ['time', 'recordid', 'ICUType']
    cols_to_plot = [col for col in df.columns if col not in exclude_cols]
    num_cols = len(cols_to_plot)

    plt.figure(figsize=(15, 5 * (num_cols // 3 + 1)))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(num_cols // 3 + 1, 3, i)
        if col in static_columns:
            sns.histplot(
                data=df.groupby('recordid')[[col, 'ICUType']].first(),
                x=col, hue='ICUType', stat='probability', common_norm=False,
                element="step", palette="coolwarm"
            )
        else:
            sns.histplot(
                data=df, x=col, hue='ICUType', stat='probability', common_norm=False,
                element="step", palette="coolwarm"
            )
        plt.title(f"Distribution of {col} by ICU Type")

    plt.tight_layout()
    plt.show()


def plot_age_distribution_by_icu_type(df):
    """
    Plots the age distribution by ICU type using a boxplot and pointplot.
    """
    plt.figure(figsize=(10, 6))

    # Boxplot for distribution
    sns.boxplot(data=df, x='ICUType', y='Age', hue='ICUType', palette='viridis', legend=False)

    # Pointplot for mean values
    sns.pointplot(data=df, x='ICUType', y='Age', color='red', estimator=np.mean,
                  errorbar=None, markers='o', linestyles='')

    plt.title('Age Distribution by ICU Type')
    plt.xlabel('ICU Type')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.show()



