import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from joblib import Parallel, delayed


MASTER_SEED = 20241225


def codalasso_clf(counts, y, test_size=.2, cvfolds=5, lamseq=np.geomspace(0.001, 1, 30), reps=100, verbose=False, njobs=False, seed=MASTER_SEED):
    """
    Coda-lasso classifier with 100 repetitions
    Requires R environment (added to R_HOME) and the rpy2 package

    Parameters
    ----------
    counts : pd.DataFrame of microbiome counts, zero replaced
    y : pd.Series of binary labels. Must be Series for dtype reconcilation.
    test_size: float, the proportion of test dataset, default is 0.2
    cvfolds : int, number of cross-validation folds, default is 5
    lamseq : np.array of lambda sequence, default is np.geomspace(0.001, 1, 30)
    reps : int, number of repetitions, default is 100
    """
    # This may help ensure R environment is set up. Uncomment if needed.
    # _setup_r_environment() 

    result = []

    def process_repetition(j, seed):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        # Source R functions 
        r = ro.r 
        r.source("./reproducibility/other_methods/CoDA-Penalized-Regression/R/functions.R")
        r.source("./reproducibility/other_methods/CoDA-Penalized-Regression/R/functions_coda_penalized_regression.R")

        lamseq_r = ro.FloatVector(lamseq)

        # Consistent split
        RS = np.random.RandomState(seed)
        X_train, X_test, y_train, y_test = train_test_split(counts, y, test_size=test_size, stratify=y, random_state=RS)

        # Cross validation for lambda selection
        scores = np.zeros((len(lamseq), 5))
        cv = _get_fold(folds=cvfolds, random_state=RS, type_Y="binary")

        folds_indices = list(cv.split(X_train, y_train))

        for i in range(len(lamseq)):
            for l, (train_idx, test_idx) in enumerate(folds_indices):
                X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
                Y_train_cv, Y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

                # Convert to R object
                with (ro.default_converter + pandas2ri.converter).context():
                    x_train_cv = ro.conversion.get_conversion().py2rpy(X_train_cv)
                    x_test_cv = ro.conversion.get_conversion().py2rpy(X_test_cv)
                    y_train_cv = ro.conversion.get_conversion().py2rpy(Y_train_cv)
                    y_test_cv = ro.conversion.get_conversion().py2rpy(Y_test_cv)

                model = r['coda_logistic_lasso'](ro.FactorVector(y_train_cv), x_train_cv, lamseq_r[i])
                y_pred = r['predict_codalasso'](x_test_cv, model)
                
                # R operations
                scores[i, l] = r['mean'](r["=="](y_pred, r["-"](r["as.numeric"](ro.FactorVector(y_test_cv)), 1)))[0]

        # Optimal lambda from CV
        lamb = lamseq_r[int(np.argmax(scores.mean(1)))]
        if verbose:
            print("Rep {} CV selected lambda: {:3f}".format(j + 1, lamb))

        # Train model with the selected lambda
        with (ro.default_converter + pandas2ri.converter).context():
            X_train = ro.conversion.get_conversion().py2rpy(X_train)
            X_test = ro.conversion.get_conversion().py2rpy(X_test)
            y_train = ro.conversion.get_conversion().py2rpy(y_train)
            y_test = ro.conversion.get_conversion().py2rpy(y_test)
        model = r['coda_logistic_lasso'](ro.FactorVector(y_train), X_train, lamb)
        Y_pred = r['predict_codalasso'](X_test, model)

        test_acc = r['mean'](r["=="](Y_pred, r["-"](r["as.numeric"](ro.FactorVector(y_test)), 1)))[0]
        return test_acc

    # Random seed for each repetition
    seeds = np.arange(seed, seed + reps)

    if njobs:
        result = Parallel(n_jobs=njobs, verbose=5)(delayed(process_repetition)(j, seeds[j]) for j in range(reps))
    else:
        for j in range(reps):
            result.append(process_repetition(j, seeds[j]))

    result = np.array(result)
    print("Estimated accuracy with 100 reps:", result.mean(), "+/-", result.std())

    return result


def codalasso_reg(X, Y, test_size=0.2, C=None, label=None, 
                  lamseq=np.geomspace(0.001, 1, 30), cvfolds=5, reps=100, verbose=False, njobs=False, seed=MASTER_SEED):
    '''
    Log-contrast regression with the lasso penalty.
    Uses the c-lasso library: https://github.com/Leo-Simpson/c-lasso/tree/master
    You can install it with `pip install c-lasso`.

    Input X should be zero-replaced count or compositional data
    '''
    from classo import classo_problem
  
    def learn(Z, Y, C=None, label=None, lam="theoretical"):
        # Z: log-transformed compositional data
        # Default C: zero sum constraint
        problem = classo_problem(Z, Y, C, label=label)
        problem.formulation.concomitant = False
        problem.formulation.huber = False
        problem.formulation.intercept = True

        problem.model_selection.CV = False
        problem.model_selection.ALO = False
        problem.model_selection.StabSel = False

        problem.model_selection.LAMfixed = True
        problem.model_selection.LAMfixedparameters.lam = lam
        problem.solve()
        
        return problem


    def predict(model, X_new):
        beta = model.solution.LAMfixed.beta
        X_new = np.c_[np.ones(len(X_new)), X_new]
        return X_new @ beta

    def one_split(j, seed):
        RS = np.random.RandomState(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=RS,
                                                            stratify=Y
                                                            )
        # Cross validation for lambda selection
        scores = np.zeros((len(lamseq), 5))
        cv = _get_fold(folds=cvfolds, random_state=RS)
        folds_indices = list(cv.split(X_train, y_train))
        
        for i in range(len(lamseq)):
            for l, (train_idx, test_idx) in enumerate(folds_indices):
                X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
                Y_train_cv, Y_test_cv = y_train[train_idx], y_train[test_idx]

                model = learn(X_train_cv, Y_train_cv, C=C, label=label, lam=lamseq[i])
                scores[i, l] = np.mean((predict(model, X_test_cv) - Y_test_cv)**2)
        lamb = lamseq[np.argmin(scores.mean(1))]
        if verbose:
            print("Rep {} CV selected lambda: {:3f}".format(j + 1, lamb))

        model = learn(X_train, y_train, C=C, label=label, lam=lamb)
        res = np.mean((predict(model, X_test) - y_test)**2)
        return res, lamb

    # Log-transformation for the log-contrast regression
    X = np.log(X)

    # 100 repetitions
    result = []
    sel_lamb = []

    # Random seed for each repetition
    seeds = np.arange(seed, seed + reps)

    if njobs:
        result, sel_lamb = zip(*Parallel(n_jobs=njobs, verbose=5)(delayed(one_split)(j, seeds[j]) for j in range(reps)))
    else:
        for j in range(reps):
            res, lamb = one_split(j, seeds[j])
            result.append(res)
            sel_lamb.append(lamb)

    result = np.array(result)
    sel_lamb = np.array(sel_lamb)
    print("Estimated MSE with 100 reps:", result.mean(), "+/-", result.std())

    return result, sel_lamb


# SVM function
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
from scipy.spatial.distance import pdist

def clr_svm(X, Y, test_size=0.2, cvfolds=5, reps=100, verbose=False, seed=MASTER_SEED):
    """
    Input X must be zero-replaced compositional data
    Input response Y must be an integral binary array for stratification in GridsearchCV
    """
    X_clr = clr(X)
    result = np.zeros(reps)
    for i in range(reps):
        RS = np.random.RandomState(seed + i)

        X_train, X_test, y_train, y_test = train_test_split(X_clr, Y, test_size=test_size, stratify=Y, random_state=RS)

        # CV
        median_distance = np.median(pdist(X_train))
        sigma_list= np.array([i * median_distance for i in np.geomspace(1/2, 2., 5)])
        gamma_list = 1 / (2 * sigma_list ** 2)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_list,
                            'C': [1., 10.]}]

        # Fold specification for reproducibility
        cv = _get_fold(cvfolds, random_state=RS, type_Y="binary")
        clf = GridSearchCV(SVC(), tuned_parameters, cv=cv, n_jobs=-1, scoring=make_scorer(accuracy_score))
        clf.fit(X_train, y_train)

        # print("Best parameters set found on development set:")
        # print(clf.best_params_)
        best_gamma_index = np.argwhere(gamma_list == clf.best_params_['gamma'])[0][0]

        if verbose:
            print("Reps", i + 1)
            print("CV fitted sigma:", sigma_list[best_gamma_index], " //  Index", best_gamma_index, "\n")
        result[i] = accuracy_score(y_test, clf.predict(X_test))
    print("Mean accuracy:", np.mean(result), "Std accuracy:", np.std(result))
    return result


# Random Forest classification
from sklearn.ensemble import RandomForestClassifier
def clr_rf_clf(X, Y, test_size=0.2, n_estimators=100, reps=100, seed=MASTER_SEED):
    """
    Input X must be zero-replaced compositional data
    Input response Y must be an integral binary array for stratification in GridsearchCV
    """
    X_clr = clr(X)
    result = np.zeros(reps)
    
    for i in range(reps):
        RS = np.random.RandomState(seed + i)    
        X_train, X_test, y_train, y_test = train_test_split(X_clr, Y, test_size=test_size, stratify=Y, random_state=RS)
        RF = RandomForestClassifier(n_estimators=n_estimators, random_state=RS)
        RF.fit(X_train, y_train)
        acc = accuracy_score(y_test, RF.predict(X_test))
        result[i] = acc

    print("Mean accuracy:", np.mean(result), "Std accuracy:", np.std(result))
    return result



# KRR function
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
def clr_krr(X, Y, test_size=0.2, cvfolds=5, reps=100, seed=MASTER_SEED):
    """Input X must be zero-replaced compositional data"""
    X_clr = clr(X)  # Centered log-ratio transformation
    result = np.zeros(reps)
    for i in range(reps):
        RS = np.random.RandomState(seed + i)
        
        X_train, X_test, y_train, y_test = train_test_split(X_clr, Y, test_size=test_size, random_state=RS, 
                                                            stratify=Y
                                                            )

        # Scale the train response (consistent with the CKDR method)
        y_train_std = y_train.std()
        y_train = y_train / y_train_std  # Standardize the response variable

        # CV
        median_distance = np.median(pdist(X_train))
        sigma_list= np.array([i * median_distance for i in np.geomspace(1/2, 2., 5)])
        gamma_list = 1 / (2 * sigma_list ** 2)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_list,
                            'alpha': np.array([0.1, 1.])}]
        
        # Fold specification for reproducibility - using KFold for regression
        cv = _get_fold(cvfolds, random_state=RS, type_Y="continuous")
        reg = GridSearchCV(KernelRidge(), tuned_parameters, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
        # grid search
        reg.fit(X_train, y_train)
        
        result[i] = mean_squared_error(y_test, reg.predict(X_test) * y_train_std)  # Scale back the predictions
    print("Mean MSE:", np.mean(result), "Std MSE:", np.std(result))
    return result


# Random Forest regression
from sklearn.ensemble import RandomForestRegressor
def clr_rf_reg(X, Y, test_size=0.2, reps=100, seed=MASTER_SEED):
    """ Input X must be zero-replaced compositional data"""
    X_clr = clr(X)  # Centered log-ratio transformation
    result = np.zeros(reps)
    for i in range(reps):
        RS = np.random.RandomState(seed + i)
        X_train, X_test, y_train, y_test = train_test_split(X_clr, Y, test_size=test_size, random_state=RS,
                                                            stratify=Y
                                                            )
        RF = RandomForestRegressor(random_state=RS, n_jobs=-1)
        RF.fit(X_train, y_train)
        mse = mean_squared_error(y_test, RF.predict(X_test))
        result[i] = mse

    print("Mean MSE:", np.mean(result), "Std MSE:", np.std(result))
    return result



#############################################################
#### Helper functions
#############################################################

def clr(mat):
    """
    Performs centered log ratio transformation.
    Adapted from the `composition_stats` library.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components
       each composition (row) must add up to unity

    Returns
    -------
    numpy.ndarray
         clr transformed matrix
    """
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


def _get_fold(folds, random_state=None, type_Y="binary", shuffle=True):
    """Get a cross-validation fold object for hyperparameter tuning"""
    if type_Y == "binary" or type_Y == "multiclass":
        cv = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    return cv


