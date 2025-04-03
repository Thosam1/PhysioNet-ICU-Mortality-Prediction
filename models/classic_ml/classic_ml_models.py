import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score

def hyperparameter_tuning(model_type, X_train, y_train, X_valid, y_valid):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - model_type (str): 'rf' for RandomForest, 'lr' for Logistic Regression
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series or np.array): Training labels
    - X_valid (pd.DataFrame): Validation features
    - y_valid (pd.Series or np.array): Validation labels

    Returns:
    - best_model: The model trained with the best hyperparameters
    - best_params: The best hyperparameters found
    - best_score: The best AUROC score on validation set
    """

    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'newton-cg']
        }
    else:
        raise ValueError("Unsupported model type. Use 'rf' or 'lr'.")

    # Use AUROC as the scoring metric
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Grid search
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters for {model_type.upper()}: {best_params}")
    print(f"Best validation AUROC: {best_score:.4f}")

    return best_model, best_params, best_score

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
    Train and evaluate models for each dataset using hyperparameter tuning.
    """
    results = {}

    for dataset in tqdm(datasets, desc=f"Training and Evaluating {model_type.upper()} Models", unit="dataset"):
        X_train = X_train_dict[dataset]
        X_valid = X_valid_dict[dataset]
        X_test = X_test_dict[dataset]

        # Train the model
        model = train_model(model_type, X_train, y_train)

        # Evaluate on the test set
        test_result = evaluate_model(model, X_test, y_test)

        results[dataset] = {
            'test_auroc': test_result['auroc'],
            'test_auprc': test_result['auprc'],
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


