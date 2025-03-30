import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

def train_model(model_type, X_train, y_train):
    """
    Train the model on the given training data based on the specified model type.

    Parameters:
    model_type (str): The type of model to train ('rf' for RandomForest, 'lr' for Logistic Regression)
    X_train (pd.DataFrame): Features for training
    y_train (pd.Series or np.array): Target labels for training

    Returns:
    model: The trained model
    """
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=42)
    else:
        raise ValueError("Model type not supported. Use 'rf' or 'lr'.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate the model on a given dataset and return AuROC and AuPRC.

    Parameters:
    model: The trained model
    X (pd.DataFrame): Features for evaluation
    y (pd.Series or np.array): True labels for evaluation

    Returns:
    dict: A dictionary containing AuROC and AuPRC
    """
    y_pred_prob = model.predict_proba(X)[:, 1]  # Probability estimates for positive class
    auroc = roc_auc_score(y, y_pred_prob)
    auprc = average_precision_score(y, y_pred_prob)
    return {'auroc': auroc, 'auprc': auprc}

def train_and_evaluate_model(model_type, X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Trains and evaluates a machine learning model using the specified model type.

    Parameters:
    model_type (str): The type of model to train ('rf' for RandomForest, 'lr' for Logistic Regression)
    X_train, X_valid, X_test (pd.DataFrame): Feature sets for training, validation, and testing
    y_train, y_valid, y_test (pd.Series or np.array): Target labels for training, validation, and testing

    Returns:
    dict: A dictionary containing performance metrics (AuROC and AuPRC) for validation and test sets
    """
    # Train the model
    model = train_model(model_type, X_train, y_train)

    # Evaluate on validation and test sets
    val_results = evaluate_model(model, X_valid, y_valid)
    test_results = evaluate_model(model, X_test, y_test)

    return {
        'validation': val_results,
        'test': test_results
    }

def train_and_evaluate_for_datasets(datasets, model_type, X_train_dict, X_valid_dict, X_test_dict, y_train, y_valid, y_test):
    """
    Train and evaluate models (Logistic Regression or Random Forest) for each dataset (mean, max, last_measured).

    Parameters:
    datasets (list): List of dataset names to iterate over ('mean', 'max', 'last_measured')
    model_type (str): The type of model to train ('rf' or 'lr')
    X_train_dict, X_valid_dict, X_test_dict (dict): Dictionaries containing feature datasets for each type
    y_train, y_valid, y_test (pd.Series or np.array): Target labels for training, validation, and testing

    Returns:
    dict: A dictionary containing performance results for each dataset
    """
    results = {}

    # Using tqdm for progress feedback
    for dataset in tqdm(datasets, desc=f"Training and Evaluating {model_type.upper()} Models", unit="dataset"):
        # Select the appropriate dataset for X
        X_train = X_train_dict[dataset]
        X_valid = X_valid_dict[dataset]
        X_test = X_test_dict[dataset]

        # Train the model
        model = train_model(model_type, X_train, y_train)

        # Evaluate the model on the validation set for tuning purposes
        valid_result = evaluate_model(model, X_valid, y_valid)

        # Evaluate the model on the test set for final evaluation
        test_result = evaluate_model(model, X_test, y_test)

        # Store the result
        results[dataset] = {
            # 'valid_auroc': valid_result['auroc'],
            'test_auroc': test_result['auroc'],
            # 'valid_auprc': valid_result['auprc'],
            'test_auprc': test_result['auprc']
        }

    return results

def plot_performance_comparison(results_rf, results_lr):
    """
    Plot the performance comparison for each dataset's validation and test set for both Random Forest and Logistic Regression.

    Parameters:
    results_rf (dict): A dictionary with performance metrics for Random Forest
    results_lr (dict): A dictionary with performance metrics for Logistic Regression
    """
    labels = list(results_rf.keys())

    # Prepare data for RandomForest and Logistic Regression
    # rf_valid_auroc = [results_rf[dataset]['valid_auroc'] for dataset in labels]
    rf_test_auroc = [results_rf[dataset]['test_auroc'] for dataset in labels]
    # lr_valid_auroc = [results_lr[dataset]['valid_auroc'] for dataset in labels]
    lr_test_auroc = [results_lr[dataset]['test_auroc'] for dataset in labels]

    # rf_valid_auprc = [results_rf[dataset]['valid_auprc'] for dataset in labels]
    rf_test_auprc = [results_rf[dataset]['test_auprc'] for dataset in labels]
    # lr_valid_auprc = [results_lr[dataset]['valid_auprc'] for dataset in labels]
    lr_test_auprc = [results_lr[dataset]['test_auprc'] for dataset in labels]

    x = range(len(labels))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot AuROC
    # axes[0].bar(x, rf_valid_auroc, width=0.3, label='RF Validation AuROC', align='center', color='blue', alpha=0.7)
    # axes[0].bar(x, lr_valid_auroc, width=0.3, label='LR Validation AuROC', align='edge', color='green', alpha=0.7)
    axes[0].bar(x, rf_test_auroc, width=0.3, label='RF Test AuROC', align='center', color='orange', alpha=0.7)
    axes[0].bar(x, lr_test_auroc, width=0.3, label='LR Test AuROC', align='edge', color='red', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('AuROC')
    axes[0].set_title('Model AuROC Comparison (RF vs LR)')
    axes[0].legend()

    # Plot AuPRC
    # axes[1].bar(x, rf_valid_auprc, width=0.3, label='RF Validation AuPRC', align='center', color='blue', alpha=0.7)
    # axes[1].bar(x, lr_valid_auprc, width=0.3, label='LR Validation AuPRC', align='edge', color='green', alpha=0.7)
    axes[1].bar(x, rf_test_auprc, width=0.3, label='RF Test AuPRC', align='center', color='orange', alpha=0.7)
    axes[1].bar(x, lr_test_auprc, width=0.3, label='LR Test AuPRC', align='edge', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('AuPRC')
    axes[1].set_title('Model AuPRC Comparison (RF vs LR)')
    axes[1].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


