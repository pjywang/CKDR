
from .CKDR import CKDR
from .pgd import train_ckdr

import numpy as np
import itertools, pickle

from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.spatial.distance import pdist


def ckdr_cv(X, Y, type_Y, folds=5, 
            sigma_list=None, epsilon_list=None, dim_list=None, 
            verbose=True, refit=True, save=False, seed=0, 
            med=True, **kwargs):
    """
    Perform cross-validation for the compositional kernel dimension reduction (CKDR) method
    to select the best hyperparameters (sigma_Z, epsilon, target_dim).

    This function iterates through a grid of specified hyperparameters,
    evaluates each combination using k-fold cross-validation, and
    identifies the set of parameters that yields the best average performance.

    Parameters:
    ----------
    X : array-like
        Input compositional data
    Y : array-like
        Response vector. 
    type_Y : str
        Type of the response. Used to determine appropriate
        cross-validation strategy (e.g., StratifiedKFold for classification).
        Expected values: "binary", "multiclass", "gaussian", "continuous", or None.
    folds : int, default=5
        Number of cross-validation folds.
    sigma_list : list or None, default=None
        List of sigma_Z values (kernel width for Z) to try.
        If None, a default range is used.
    epsilon_list : list or None, default=None
        List of epsilon values (regularization for K_Y) to try.
        If None, a default value is used.
    dim_list : list or None, default=None
        List of target_dim values (dimensionality of the subspace) to try.
        If None, a default value is used.
    verbose : bool, default=True
        Whether to print progress messages.
    refit : bool, default=True
        Whether to refit the model on the entire dataset using the best
        found parameters after cross-validation.
    save : bool or str, default=False
        If False, results are not saved. If a string is provided,
        it is used as the filename (without extension) to save the
        pickled results dictionary in the "./training_results/" directory.
    seed : int or np.random.RandomState, default=0
        Seed for random number generation to ensure reproducibility.
    **kwargs : dict
        Additional keyword arguments passed to the `train_ckdr` function.
        These can include parameters like initial_lr, armijo_c1, etc.

    Returns:
    -------
    results : dict
        A dictionary containing:
        - "scores": A numpy array of scores for all parameter combinations and folds.
        - "parameters": A dictionary with the best "target_dim", "epsilon", and "sigma_Z".
        - "best_index": A tuple of indices (i, j, k) corresponding to the best parameters.
        - "CV_fitted_P" (if refit=True): The projection matrix P from the refitted model.
        - "CV_ckdr_class" (if refit=True): The CKDR model instance from the refit.
    """
    # Default hyperparameter grid search values
    if dim_list is None:
        dim_list = [3]
    if epsilon_list is None:
        epsilon_list = [1/X.shape[0]] # good value for gaussian kernel regression
    if sigma_list is None:
        # Default sigma_list based on a geometric space and the median of pairwise distances
        sigma_list = np.geomspace(1/5, 1, 10)
    
    base_sigma = np.median(pdist(X)) if med else 1.
    sigma_list = base_sigma * np.array(sigma_list)

    # Initialize RandomState for reproducibility
    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    # Initialize CKDR instance to process data
    ckdr_instance = CKDR(X, Y, type_Y=type_Y)
    X_processed, Y_processed = ckdr_instance.X, ckdr_instance.Y
    
    # Store initial Y labels for later use
    y_names = ckdr_instance.values if type_Y in ["binary", "multiclass"] else None
    
    # Ensure folds do not exceed the number of samples
    num_folds = min(folds, X_processed.shape[0])

    # Initialize array to store scores
    # Dimensions: (num_dim_values, num_epsilon_values, num_sigma_values, num_folds)
    cv_scores = np.zeros((len(dim_list), len(epsilon_list), len(sigma_list), num_folds))
    
    # Get cross-validation splitter
    cv_splitter = _get_cv_splitter(num_folds, type_Y=type_Y, random_state=RS)

    # Static fold split across all parameter combinations to ensure consistency 
    cv_fold_indices = _get_cv_indices(X_processed, Y_processed, cv_splitter, type_Y, Y)

    # Fix random initialization seed across all folds to reduce unnecessary variance
    init_seed = RS.randint(0, 2**31 - 1, dtype=np.int32)

    # Iterate over all hyperparameter combinations
    for i, current_dim in enumerate(dim_list):
        for j, current_epsilon in enumerate(epsilon_list):
            for k, current_sigma_Z in enumerate(sigma_list):
                if verbose:
                    print(f"\\nTesting parameters: target_dim = {current_dim}, epsilon = {current_epsilon}, sigma_Z = {current_sigma_Z:.4f}")

                # Perform k-fold cross-validation
                for l, (train_indices, test_indices) in enumerate(cv_fold_indices):
                    if verbose:
                        print(f"   Fold {l + 1}:")
                    
                    X_train, Y_train = X_processed[train_indices], Y_processed[train_indices]
                    X_test, Y_test = X_processed[test_indices], Y_processed[test_indices]

                    # Train the CKDR model for the current fold and parameters
                    P_matrix, _, trained_ckdr_model = train_ckdr(
                        X_train, Y_train, 
                        target_dim=current_dim, epsilon=current_epsilon, sigma=current_sigma_Z, type_Y=type_Y,
                        verbose=False, # verbose is off for inner loops
                        seed=init_seed,
                        **kwargs
                    )

                    # Process test data with the trained model and calculate loss
                    X_test_p, Y_test_p = trained_ckdr_model.test_processing(X_test, Y_test)
                    loss = trained_ckdr_model.predict_loss(P_matrix, current_sigma_Z, current_epsilon, X_test_p, Y_test_p, proj_result=False)
                    cv_scores[i, j, k, l] = loss / X_test_p.shape[0] # Normalize loss

    # Calculate mean scores across folds
    mean_cv_scores = np.mean(cv_scores, axis=3)

    # Find the indices of the best parameters
    best_param_indices = np.unravel_index(np.argmin(mean_cv_scores), mean_cv_scores.shape)
    i_best, j_best, k_best = best_param_indices[0], best_param_indices[1], best_param_indices[2]
    
    # Retrieve the best hyperparameters
    best_target_dim = dim_list[i_best]
    best_epsilon = epsilon_list[j_best]
    best_sigma_Z = sigma_list[k_best]

    if verbose:
        print(f"\\nSelected parameters after {num_folds}-fold Cross-validation:")
        print(f"  target_dim = {best_target_dim}, epsilon = {best_epsilon}, sigma_Z = {best_sigma_Z:.4f}")
        print(f"  Corresponding mean CV score: {mean_cv_scores[i_best, j_best, k_best]:.4f}")
    
    results_dict = {
        "scores": cv_scores,
        "parameters": {
            "target_dim": best_target_dim,
            "epsilon": best_epsilon,
            "sigma_Z": best_sigma_Z
        },
        "best_index": (i_best, j_best, k_best)
    }

    if refit:
        if verbose:
            print("\\nRefitting model with selected best parameters on the full dataset...")
        # Refit the model on the entire processed dataset
        P_refit, obj_refit, ckdr_refit_model = train_ckdr(
            X_processed, Y_processed, 
            target_dim=best_target_dim, 
            epsilon=best_epsilon, 
            sigma=best_sigma_Z, 
            type_Y=type_Y,
            verbose=verbose, # Use main verbose setting for refit
            seed=RS, # Use the main random state for the final fit
            **kwargs
        )
        if type_Y not in ["binary", "multiclass"]:
            # preserve the standard deviation of Y for continuous responses
            ckdr_refit_model.Y_std = ckdr_instance.Y_std
        else:  
            # Retrieve original Y labels into the output model
            ckdr_refit_model.values = y_names if y_names is not None else ckdr_instance.values

        results_dict["CV_fitted_P"] = P_refit
        results_dict["CV_obj_refit"] = obj_refit
        results_dict["CV_ckdr_class"] = ckdr_refit_model

        if save:
            filename = str(save) if isinstance(save, (str, int, float)) else "ckdr_cv_results"
            filepath = f"./training_results/{filename}.pickle"
            if verbose:
                print(f"Saving results to {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict, f)
            if verbose:
                print("Results saved.")

    return results_dict


def ckdr_cv_parallel(X, Y, type_Y, folds=5, 
                     sigma_list=None, epsilon_list=None, dim_list=None, 
                     verbose=True, refit=True, save=False, seed=0,
                     n_jobs=-2, med=True, **kwargs):
    """
    Parallelized cross-validation for CKDR using joblib.
    This version parallelizes the evaluation of each hyperparameter combination
    across all cross-validation folds.

    Parameters:
    ----------
    X : array-like
        Input features.
    Y : array-like
        Target variable.
    type_Y : str
        Type of the target variable (e.g., "binary", "multiclass", "gaussian").
    folds : int, default=5
        Number of cross-validation folds.
    sigma_list : list or None, default=None
        List of sigma_Z values to try.
    epsilon_list : list or None, default=None
        List of epsilon values to try.
    dim_list : list or None, default=None
        List of target_dim values to try.
    verbose : bool, default=True
        Whether to print progress messages.
    refit : bool, default=True
        Whether to refit the model on the entire dataset using the best parameters.
    save : bool or str, default=False
        If a string, used as filename to save pickled results.
    seed : int or np.random.RandomState, default=0
        Seed for random number generation.
    n_jobs : int, default=-2
        Number of CPU cores to use for parallel processing.
        -1 means using all processors.
        -2 means using all processors but one.
    **kwargs : dict
        Additional keyword arguments passed to `train_ckdr`.

    Returns:
    -------
    results_dict : dict
        Dictionary containing CV scores, best parameters, and optionally the refitted model.
    """
    # Default hyperparameter grid search values
    if dim_list is None:
        dim_list = [3]
    if epsilon_list is None:
        epsilon_list = [1/X.shape[0]] # good value for gaussian kernel regression
    if sigma_list is None:
        # Default sigma_list based on a geometric space and the median of pairwise distances
        sigma_list = np.geomspace(1/5, 1, 10)
    
    base_sigma = np.median(pdist(X)) if med else 1.
    sigma_list = base_sigma * np.array(sigma_list)

    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    # Initialize CKDR instance to process data
    ckdr_instance = CKDR(X, Y, type_Y=type_Y)
    X_processed, Y_processed = ckdr_instance.X, ckdr_instance.Y
    
    # Store initial Y labels for later use
    y_names = ckdr_instance.values if type_Y in ["binary", "multiclass"] else None

    # Static fold split across all parameter combinations
    actual_num_folds = min(folds, X_processed.shape[0])
    cv_splitter = _get_cv_splitter(actual_num_folds, type_Y=type_Y, random_state=RS)
    cv_fold_indices = _get_cv_indices(X_processed, Y_processed, cv_splitter, type_Y, Y)
    # list(cv_splitter.split(X_processed, Y_processed.numpy()))

    # Prepare list of all tasks (each task is one param_set + one_fold evaluation)
    tasks_to_run = []
    parameter_combinations = list(itertools.product(enumerate(dim_list), 
                                                  enumerate(epsilon_list), 
                                                  enumerate(sigma_list)))

    for (i, target_dim_val), (j, epsilon_val), (k, sigma_Z_val) in parameter_combinations:
        for l, (train_indices, test_indices) in enumerate(cv_fold_indices):
            tasks_to_run.append(
                ((i, target_dim_val), (j, epsilon_val), (k, sigma_Z_val), 
                 (l, train_indices, test_indices))
            )
    
    if verbose:
        num_param_sets = len(parameter_combinations)
        print(f"Running parallel cross-validation with {num_param_sets} parameter combinations "
              f"across {actual_num_folds} folds, totaling {len(tasks_to_run)} individual training tasks.")

    # Fix random initialization seed across all folds to reduce unnecessary variance
    init_seed = RS.randint(0, 2**31 - 1, dtype=np.int32)

    # Prepare arguments for each task (without duplicating large arrays)
    task_args = [(task, init_seed, X_processed, Y_processed, type_Y, kwargs) 
                 for task in tasks_to_run]

    parallel_job_verbosity = 5 if verbose else 0 
    all_task_results = Parallel(n_jobs=n_jobs, 
                                verbose=parallel_job_verbosity,
                                )(
        delayed(_evaluate_single_fold_task)(args) for args in task_args
    )
    
    # Reconstruct the scores array from parallel results
    cv_scores = np.zeros((len(dim_list), len(epsilon_list), len(sigma_list), actual_num_folds))
    if all_task_results: # Ensure results were returned
        for i_res, j_res, k_res, l_res, score_res_val in all_task_results:
            cv_scores[i_res, j_res, k_res, l_res] = score_res_val
    
    mean_cv_scores = np.mean(cv_scores, axis=3)
    
    # Find best parameters
    best_param_indices_flat = np.unravel_index(np.argmin(mean_cv_scores), mean_cv_scores.shape)
    i_best, j_best, k_best = best_param_indices_flat[0], best_param_indices_flat[1], best_param_indices_flat[2]
    
    best_target_dim = dim_list[i_best]
    best_epsilon = epsilon_list[j_best]
    best_sigma_Z = sigma_list[k_best]

    if verbose:
        print(f"\\nSelected parameters after {actual_num_folds}-fold Cross-validation (parallel):")
        print(f"  target_dim = {best_target_dim}, epsilon = {best_epsilon}, sigma_Z = {best_sigma_Z:.4f}")
        print(f"  Corresponding mean CV score: {mean_cv_scores[i_best, j_best, k_best]:.4f}")
    
    results_output_dict = {
        "scores": cv_scores,
        "parameters": {
            "target_dim": best_target_dim,
            "epsilon": best_epsilon,
            "sigma_Z": best_sigma_Z
        },
        "best_index": (i_best, j_best, k_best)
    }

    if refit:
        if verbose:
            print("\\nRefitting model with selected best parameters on the full dataset (parallel CV)...")
        P_refit, obj_refit, ckdr_refit_model = train_ckdr(
            X_processed, Y_processed, target_dim=best_target_dim, epsilon=best_epsilon, sigma=best_sigma_Z, type_Y=type_Y,
            verbose=verbose, 
            seed=RS, 
            **kwargs
        )
        if type_Y not in ["binary", "multiclass"]:
            # preserve the standard deviation of Y for continuous responses
            ckdr_refit_model.Y_std = ckdr_instance.Y_std
        else:  
            # Retrieve original Y labels into the output model
            ckdr_refit_model.values = y_names if y_names is not None else ckdr_instance.values

        results_output_dict["CV_fitted_P"] = P_refit
        results_output_dict["CV_obj_refit"] = obj_refit
        results_output_dict["CV_ckdr_class"] = ckdr_refit_model\

        if save:
            filename = str(save) if isinstance(save, (str, int, float)) else "ckdr_cv_parallel_results"
            filepath = f"./training_results/{filename}.pickle"
            if verbose:
                print(f"Saving results to {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(results_output_dict, f)
            if verbose:
                print("Results saved.")

    return results_output_dict


# Define the worker function for parallel execution
def _evaluate_single_fold_task(args):
    """
    Top-level worker for ckdr_cv_parallel.
    Receives all necessary data as arguments in a tuple.
    """
    (task_details, seed, X_processed, Y_processed, type_Y, kwargs_dict) = args
    # from .pgd import train_ckdr
    (param_idx_dim, current_dim), (param_idx_eps, current_epsilon), (param_idx_sigma, current_sigma_Z), \
    (fold_idx, train_indices, test_indices) = task_details

    X_train_fold, Y_train_fold = X_processed[train_indices], Y_processed[train_indices]
    X_test_fold, Y_test_fold = X_processed[test_indices], Y_processed[test_indices]
    P_matrix, _, trained_ckdr_model = train_ckdr(
        X_train_fold, Y_train_fold, target_dim=current_dim, epsilon=current_epsilon, sigma=current_sigma_Z, type_Y=type_Y,
        verbose=False, seed=seed, **kwargs_dict
    )
    X_test_p_fold, Y_test_p_fold = trained_ckdr_model.test_processing(X_test_fold, Y_test_fold)
    score_val = trained_ckdr_model.predict_loss(
        P_matrix, current_sigma_Z, current_epsilon, X_test_p_fold, Y_test_p_fold, proj_result=False
    ) / X_test_p_fold.shape[0]
    return param_idx_dim, param_idx_eps, param_idx_sigma, fold_idx, score_val



def _get_cv_indices(X_processed, Y_processed, cv_splitter, type_Y, Y):
    """
    Helper function to get a K-fold or StratifiedKFold cross-validation indices.

    Returns:
    -------
    cv_indices: list of tuples
        Each tuple contains (train_indices, test_indices) for each fold.
    """
    if type_Y != "multiclass":
        return list(cv_splitter.split(X_processed, Y_processed))
    else:
        # if multiclass, use original input Y to ensure multiclass stratification
        return list(cv_splitter.split(X_processed, Y))  # Use original Y for multiclass stratification


def _get_cv_splitter(num_folds, type_Y, random_state=None, shuffle=True):
    """
    Helper function to get a K-fold or StratifiedKFold cross-validation splitter.

    Parameters:
    ----------
    num_folds : int
        Number of folds.
    type_Y : str
        Type of the target variable. Determines if StratifiedKFold is used.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before splitting.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.

    Returns:
    -------
    cv_splitter : KFold or StratifiedKFold instance
    """
    valid_Y_types = [None, "binary", "multiclass", "gaussian", "continuous"]
    if type_Y not in valid_Y_types:
        raise ValueError(f"Invalid type_Y: {type_Y}. Must be one of {valid_Y_types}.")

    if type_Y in ["binary", "multiclass"]:
        # Use StratifiedKFold for classification tasks to preserve class proportions
        return StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
    else:
        # Use KFold for regression or other tasks
        return KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

