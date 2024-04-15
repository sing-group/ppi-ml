import os
import math
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from slugify import slugify

from print import pr_orange
from datasets import compute_counts


def duplicate_labels(arr):
    result = []
    result.extend(arr)
    result.extend(arr)

    return np.array(result)


def results_csv_row_header(metrics, sep=';'):
    return 'dataset;combination;model;' + sep.join(metrics)


def exclude_std_keys(keys):
    return list(filter(lambda k: not k.endswith('_std'), keys))


def results_csv_row(metrics, dataset_name, dataset_combination, model, sep=';'):
    row = [dataset_name, dataset_combination, model]
    for metric in exclude_std_keys(metrics.keys()):
        mean = metrics[metric]
        row.append(f'{mean:.3f}')

    return sep.join(row)


def show_metrics(results, indent=1):
    for metric in exclude_std_keys(results.keys()):
        mean = results[metric]
        indent_str = '\t' * indent
        
        metric_std = f'{metric}_std'
        if metric_std in results:
            std = results[metric_std]
            print(f'{indent_str}{metric} = {mean:.3f} ±{std:.3f}')
        else:
            print(f'{indent_str}{metric} = {mean:.3f}')


def cat(path):
    with open(path, 'r', encoding = 'utf-8') as file:
        for line in file:
            print(line[:-1])


def write_or_append_file(filename, content, add_new_line=True):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, encoding='utf-8') as file:
        file.write(content)
        if add_new_line:
            file.write('\n')


def show_intermediate_results(dataset, model, results, log_file, prompt):
    print(prompt)
    show_metrics(results, indent=2)
    print('*' * 80)

    dataset_name = dataset.split('__')[0]
    dataset_combination = dataset.split('__')[1]
    write_or_append_file(log_file, results_csv_row(results, dataset_name, dataset_combination, model))

def do_nested_cv(
    X,
    y,
    model,
    param_grid,
    combinator,
    inner_splits=5,
    outer_splits=5,
    inner_refit_scoring='f1',
    n_jobs=-1,
    print_debug_messages=False
):
    """
    Perform nested cross-validation for model selection and evaluation, with special considerations 
    for protein-protein interaction data.

    This function facilitates nested cross-validation, allowing for both model selection via hyperparameter
    tuning within the inner loop and model evaluation in the outer loop. It is specifically tailored for 
    datasets involving protein-protein interactions, offering options to handle duplicated labels resulting 
    from data augmentation techniques like adding inverted interactions.

    Parameters
    - X (pandas.DataFrame): The feature matrix with protein interaction data.
    - y (array-like): The target vector indicating the outcome of each interaction in X.
    - model: The machine learning estimator object from scikit-learn or a compatible library.
    - param_grid (dict or list of dicts): Dictionary with parameters names (str) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, each corresponding to a different search space.
    - combinator: An object responsible for preprocessing the feature matrix X and perform embedding extraction 
    and data augmentation.
    - inner_splits (int, optional): Number of folds for the inner cross-validation loop, used for hyperparameter
    tuning. Defaults to 5.
    - outer_splits (int, optional): Number of folds for the outer cross-validation loop, used for model evaluation.
    Defaults to 5.
    - inner_refit_scoring (str, optional): Scoring metric for refitting the model on the entire training set within
    the inner cross-validation loop. Defaults to 'f1'.
    - n_jobs (int, optional): Number of jobs to run in parallel during the GridSearchCV phase. -1 means using all 
    processors. Defaults to -1.
    - print_debug_messages (bool, optional): Flag to enable printing of debug messages during execution. Useful for 
    tracking progress and debugging.

    Returns:
    - tuple: A tuple containing three lists:
        - pred_y_folds (list of numpy arrays): Predictions for each outer fold.
        - true_y_folds (list of numpy arrays): True target values for each outer fold.
        - test_indexes (list of array-like): Indices of the test sets for each outer fold.

    Notes:
    - This function is particularly suitable for datasets where it's crucial to maintain the integrity of biological
    entities (e.g., proteins) across folds, thereby preventing data leakage and ensuring that the model's performance
    is evaluated on entirely unseen entities.
    - The function utilizes a custom process for creating training and testing splits to accommodate the unique 
    requirements of protein-protein interaction data.
    - The `combinator` object plays a critical role in preprocessing the data, potentially handling tasks such as 
    embedding extraction and the management of duplicated labels due to data augmentation techniques.

    Example:
        >>> X = pd.DataFrame({'prot1': ['A', 'B', 'C', 'D'], 'prot2': ['E', 'F', 'G', 'H'], 'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
        >>> y = np.array([0, 1, 0, 1])
        >>> model = SVC()
        >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
        >>> combinator = SomeCombinator()
        >>> pred_y_folds, true_y_folds, test_indexes = do_nested_cv(X, y, model, param_grid, combinator)
    """
    inner_CV = StratifiedGroupKFold(n_splits=inner_splits)
    outer_CV = StratifiedKFold(n_splits=outer_splits)

    pred_y_folds = []
    true_y_folds = []
    test_indexes = []

    iteration = 1
    
    for train_index, test_index in outer_CV.split(X, y):
        print(f'\tStarting outer CV iteration {iteration} out of {outer_splits}')
        iteration = iteration + 1

        X_tr, X_tt = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_tt = y[train_index], y[test_index]

        if print_debug_messages:
            pr_orange(f'\t\tOuter CV train interactions: {X_tr.shape[0]} with cv = {outer_splits}')

        X_tr_emb, groups = combinator.unpack_embeddings(X_tr)

        if combinator.should_duplicate_labels():
            y_tr = duplicate_labels(y_tr)

        inner_cv_split_indices = list(inner_CV.split(X_tr_emb, y_tr, groups=groups))

        inner_cv_split_indices_filtered = []

        for train_idx, test_idx in inner_cv_split_indices:
            # 1. Fix testing folds
            if combinator.should_duplicate_labels():
                # When reversed interactions are added by a combinator, those interactions are
                # removed from the test folds to make those tests like the original data.
                filtered_test_idx = [idx for idx in test_idx if idx < X_tr.shape[0]]
            else:
                filtered_test_idx = test_idx
            
            inner_cv_split_indices_filtered.append((train_idx, filtered_test_idx))

        clf = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=inner_cv_split_indices_filtered, refit=inner_refit_scoring, n_jobs=n_jobs)
        clf.fit(X_tr_emb, y_tr)

        X_tt_emb, _ = combinator.unpack_embeddings_test(X_tt)
        pred = clf.predict(X_tt_emb)
        
        pred_y_folds.append(pred)
        true_y_folds.append(y_tt)
        test_indexes.append(test_index)

    return pred_y_folds, true_y_folds, test_indexes


def compute_metrics(
    pred_y_folds,
    true_y_folds,
    scoring,
    per_fold=True
):
    """
    Compute evaluation metrics for model predictions against true values, optionally on a per-fold basis.

    This function calculates various evaluation metrics for given predicted and true values,
    which are organized by cross-validation folds. It supports computing metrics for each fold
    individually and then aggregating them, or computing metrics across all data ignoring fold divisions.

    Args:
        pred_y_folds (list of lists): Predicted values for each fold in cross-validation.
        true_y_folds (list of lists): True values for each fold in cross-validation.
        scoring (dict): A dictionary where keys are metric names and values are callable functions that
                        calculate the metric given true and predicted values.
        per_fold (bool, optional): If True, metrics are computed for each fold and then aggregated. 
                                   If False, metrics are computed across all data. Defaults to True.

    Returns:
        dict: A dictionary of computed evaluation metrics. If `per_fold` is True, each metric will also
              include its standard deviation across folds with the key format `{metric_name}_std`.

    Notes:
        - The length of each fold in `pred_y_folds` and `true_y_folds` must match.
        - The scoring functions in the `scoring` dictionary should accept two arguments: the true values
          and the predicted values, in that order.

    Raises:
        ValueError: If there's a mismatch in the length of folds or other input inconsistencies.
        TypeError: If scoring functions are not callable or return invalid values.

    Examples:
        >>> from sklearn.metrics import accuracy_score, precision_score
        >>> pred_y_folds = [[0, 1, 0], [1, 0, 0]]
        >>> true_y_folds = [[1, 0, 0], [0, 1, 0]]
        >>> scoring = {'accuracy': accuracy_score, 'precision': precision_score}
        >>> compute_metrics(pred_y_folds, true_y_folds, scoring)
        {'accuracy': 0.5, 'accuracy_std': 0.05, 'precision': 0.5, 'precision_std': 0.05}
    """
    results = {}

    if per_fold:
        results_folds = {}
        for y_pred, y_true in zip(pred_y_folds, true_y_folds):
            for metric in scoring:
                if not metric in results_folds:
                    results_folds[metric] = []

                results_folds[metric].append(scoring[metric](y_true, y_pred))
        
        for metric, values in results_folds.items():
            results[metric] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)

    else:
        y_pred = [item for sublist in pred_y_folds for item in sublist]
        y_true = [item for sublist in true_y_folds for item in sublist]

        for metric in scoring:
            try:
                results[metric] = scoring[metric](y_true, y_pred)
            except ValueError:
                results[metric] = math.nan
    
    return results


def extract_proteins(interaction_index, X):
    prot1 = X['prot1'].values[interaction_index]
    prot2 = X['prot2'].values[interaction_index]

    return prot1, prot2


def log_folds(
    pred_y_folds, 
    true_y_folds,
    test_indexes,
    X_train,
    y_train,
    logs_dir,
    log_filename
):
    log_filename = slugify(log_filename.replace('__', 'PLACEHOLDER'), separator='_', lowercase=False)
    log_filename = log_filename.replace('PLACEHOLDER', '__')

    y_pred = [item for sublist in pred_y_folds for item in sublist]
    y_true = [item for sublist in true_y_folds for item in sublist]
    indexes = [item for sublist in test_indexes for item in sublist]

    counts_y_true_1, counts_y_true_0 = compute_counts(X_train, y_train)

    with open(f'{logs_dir}/{log_filename}', 'w', encoding='utf-8') as log_file:
        log_file.write('index,prot_a,prot_b,prot_a_total_int,prot_a_pos_int,prot_b_total_int,prot_b_pos_int,y_true,y_pred\n')
        for i, current_index in enumerate(indexes):
            prot1, prot2 = extract_proteins(current_index, X_train)

            prot_a_pos_int = counts_y_true_1.get(prot1)
            prot_b_pos_int = counts_y_true_1.get(prot2)
            prot_a_total_int = prot_a_pos_int + counts_y_true_0.get(prot1)
            prot_b_total_int = prot_b_pos_int + counts_y_true_0.get(prot2)

            log_file.write(f'{current_index},{prot1},{prot2},{prot_a_total_int},{prot_a_pos_int},{prot_b_total_int},{prot_b_pos_int},{y_true[i]},{y_pred[i]}\n')
