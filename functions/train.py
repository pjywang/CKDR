# Train functions using train/test split
import os

from .cross_val import ckdr_cv, ckdr_cv_parallel

import torch
import numpy as np
import pickle

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

MASTER_SEED = 20241225  # Default seed for reproducibility


def fit_train_test_split(X, Y, type_Y, test_size=0.2, folds=5,
            sigma_list=None, epsilon_list=None, dim_list=None,
            parallel=False, n_jobs=-2, refit_return=False,
            verbose=True, seed=None, stratify=False, **kwargs):
    """
    Perform test error computation from a random train/test split using 
    the CKDR method. Within training data, the best hyperparameters
    (sigma_Z, epsilon, target_dim) are selected through cross-validation
    using `ckdr_cv` function.
    
    Parameters:
    ----------
    X : array-like
        Input compositional data
    Y : array-like
        Response vector. 
        Expected to be 1-dimensional if type_Y = "binary" or "multiclass"
    type_Y : str
        Type of the response. Used to determine appropriate
        cross-validation strategy (e.g., StratifiedKFold for classification).
        Expected values: "binary", "multiclass", "gaussian", "continuous", or None.
    test_size : float, default=0.2
        Proportion of the test data
    folds : int, default=5
        Number of cross-validation folds for inner training.
    sigma_list : list or None, default=None
        List of sigma_Z values (kernel width for Z) to try.
        If None, a default range is used as a multiple of the median heuristic.
        np.geomspace(1/2, 2, 5) is recommended.
    epsilon_list : list or None, default=None
        List of epsilon values (regularization for K_Y) to try.
        If None, a default value is used.
    dim_list : list or None, default=None
        List of target_dim values (dimensionality of the subspace) to try.
        If None, a default value is used.
    verbose : bool, default=True
        Whether to print progress messages.
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
    
    """
    assert type_Y in ("binary", "multiclass", "continuous", None), \
        "type_Y must be 'binary', 'multiclass', 'continuous', or None for valied prediction assessment."

    # RandomState setting for reproducibility
    RS = np.random.RandomState(seed)

    # Stratified splitting if classification
    if type_Y in ("binary", "multiclass"):
        assert Y.ndim == 1, "For classification, Y must be 1-dimensional."
        y = Y
    elif stratify==True and type_Y == "continuous":
        assert np.unique(Y).size < 20, "For continuous response, stratification is only valid if there are fewer than 20 unique values."
        y = Y
    else:
        y = None # continuous response--no stratification

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=y, random_state=RS)

    # Cross-validation to find the best hyperparameters
    if not parallel:
        results = ckdr_cv(X_train, Y_train, type_Y, folds=folds, 
                        sigma_list=sigma_list, epsilon_list=epsilon_list, 
                        dim_list=dim_list, verbose=verbose, seed=RS, **kwargs)
    else:
        results = ckdr_cv_parallel(X_train, Y_train, type_Y, folds=folds, 
                        sigma_list=sigma_list, epsilon_list=epsilon_list, 
                        dim_list=dim_list, verbose=verbose, n_jobs=n_jobs, seed=RS, **kwargs)    

    sigma_Z_opt = results['parameters']['sigma_Z']
    epsilon_opt = results['parameters']['epsilon']
    train_model = results['CV_ckdr_class']
    fitted_P = results['CV_fitted_P']

    # Prediction performance measured on test data with the trained model
    X_test_p, Y_test_p = train_model.test_processing(X_test, Y_test)

    predictions = train_model.predict(fitted_P, sigma_Z_opt, epsilon_opt, X_test_p)
    # Detach and convert to numpy for convenience
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(Y_test_p):
        Y_test_p = Y_test_p.detach().cpu().numpy()

    pred_measure = None
    if type_Y == "binary":
        # Test accuracy for binary classification
        Y_test_p = Y_test_p.ravel() # flatten to 1D for comparison
        pred_measure = np.mean(predictions == Y_test_p)
    elif type_Y == "multiclass":
        # Test accuracy for one-hot encoded multiclass classification
        pred_measure = np.mean(np.all(predictions == Y_test_p, axis=1))
    elif type_Y in ("continuous", None):
        # Mean squared error at the actual response values
        std = train_model.Y_std.numpy()
        predictions = predictions * std
        Y_test_p = Y_test_p * std
        pred_measure = np.sum(np.mean((predictions - Y_test_p) ** 2, axis=0)) # if multiple outputs, sum over all

    if refit_return:
        # Return the trained model and prediction measure
        return pred_measure, results
    else:
        # Return only the prediction measure
        return pred_measure



def parallel_fit_train_test_split(X, Y, type_Y, reps=100, n_jobs=-2,
            test_size=0.2, folds=5,
            sigma_list=None, epsilon_list=None, dim_list=None, 
            verbose=False, 
            save=False, 
            seed=MASTER_SEED, 
            refit_return=True, # Turn off when finalized
            med=True,
            stratify=True,
            **kwargs):
    """
    Parallel execution of fit_train_test_split using joblib.Parallel.
    Random seed for train/test split is set as seed + rep index.
    """
    seeds = [seed + rep for rep in range(reps)]  # Independent seeds for each repetition


    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(fit_train_test_split)(
        X, Y, type_Y, test_size=test_size, folds=folds,
        sigma_list=sigma_list, epsilon_list=epsilon_list, dim_list=dim_list, 
        verbose=verbose, 
        seed=seeds[rep],  # consistent, independent seed control
        refit_return=refit_return, med=med, stratify=stratify, **kwargs
    ) for rep in range(reps))

    print(f"Results of {reps} repetitions of train/test split: {np.mean([res[0] for res in results]):4f} ± {np.std([res[0] for res in results]) / np.sqrt(reps):4f}")

    if save:
        if not os.path.exists("./results/realdata-prediction/"):
            os.makedirs("./results/realdata-prediction/")
        if isinstance(save, str):
            # Given filename if string
            filename = save 
        else:
            filename = f"CKDR_train_test_split_{type_Y}_{reps}reps"
        with open(f"./results/realdata-prediction/{filename}.pkl", "wb") as f:
            pickle.dump(results, f)

    return results