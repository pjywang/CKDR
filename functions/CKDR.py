import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .objfunc import Gaussian_gram_matrix, center


class CKDR(object):
    """
    Initializes parameters and necessary processing for the CKDR method.

    Also serves as a training data & results storage.

    For prediction on test data, use the Y_test_processing method to preprocess the test response data.

    Later, the high-level training function will first process the data types into torch.Tensor.
    """

    def __init__(
            self, X, Y, type_Y,
            dtype=torch.float32, device='cpu'
    ):
        """
        :param X: Input data (np.ndarray or pd.DataFrame or torch.Tensor)
        :param Y: Label of data (np.ndarray or pd.DataFrame or torch.Tensor)
        :param sigma_Z: manual kernel parameter for the target simplex. Default is the median heuristic of X (slight shrinkage is recommended)


        :param type_Y: None, "multiclass", "binary", "continuous", "gaussian". If None (== continuous), we use linear kernel for Y. If gaussian, we embed Y into Gaussian RKHS.
        
        :param dtype: torch dtype (default is float32)           
        :param device: torch device (default is cpu; can be 'cuda' if available and the sample size is large (>1000))
        """

        assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or isinstance(X, torch.Tensor), 'X must be np.ndarray, pd.DataFrame, or torch.Tensor'
        assert isinstance(Y, np.ndarray) or isinstance(Y, pd.DataFrame) or isinstance(Y, torch.Tensor), 'Y must be np.ndarray, pd.DataFrame, or torch.Tensor'
        assert X.ndim == 2
        assert Y.ndim in (1, 2), 'Y must be 1D or 2D array'
        assert type_Y in [None, "binary", "multiclass", "gaussian", "continuous"], "Invalid type_Y. Must be one of [None, 'binary', 'multiclass', 'gaussian', 'continuous']."

        # self.epsilon = epsilon

        X = np.array(X)
        self.n, self.d = X.shape
        assert np.isclose(X.sum(axis=1), 1).all(), 'X must be compositional'
        # assert (np.abs(X.sum(axis=1) - 1) < 1e-8).all(), 'X is not compositional'
        assert Y.shape[0] == self.n, 'Sample sizes of X and Y do not match'

        # X as a class element
        self.X = torch.tensor(X, dtype=dtype, device=device)
        self.type_Y = type_Y

        # Preprocessing for Y
        Y = np.array(Y)
        Y = Y.reshape(self.n, -1)  # Ensure Y is a 2D array
        self.K_Y = None  # Initialize Gram matrix of Y for Gaussian case
        if type_Y == "binary":
            # -1, 1 encoding
            self.values = sorted(set(Y.ravel()))
            assert len(self.values) == 2, "Input Y has more than 2 unique values for binary classification"
            Y_new = np.zeros((self.n, 1))
            Y_new[Y == self.values[0]] = -1
            Y_new[Y == self.values[1]] = 1
            Y = Y_new

        elif type_Y == "multiclass":
            self.values = sorted(set(Y.ravel()))
            # convert to one-hot encoding; skip this if already one-hot encoded
            if set(np.unique(Y)) != {0., 1.}:
                assert len(Y.ravel()) == self.n, "Input Y must be a 1-dim array with multiple classes or a 2-dim array with one-hot encoding"
                Y_new = np.zeros((self.n, len(self.values)))
                for i, value in enumerate(self.values):
                    Y_new[Y.ravel() == value, i] = 1
                Y = Y_new
        else:
            # Continuous response, no encoding needed
            # Scale Y to unit variance; no centering needed since we center everything in the method
            Y_std = np.std(Y, axis=0)
            Y = Y / Y_std 
            self.Y_std = torch.tensor(Y_std, dtype=dtype, device=device)
        
        if type_Y == "gaussian":
            # Gaussian kernel Gram matrix for Y (in case continuous response)
            self.sigma_Y = torch.tensor(np.median(pdist(Y)) / np.sqrt(2), dtype=dtype)
            self.K_Y = Gaussian_gram_matrix(torch.tensor(Y, dtype=dtype), self.sigma_Y)

        # Store uncentered Y
        self.Y = torch.tensor(Y, dtype=dtype, device=device)
        
        # Store dtype and device for test use
        self.dtype = dtype
        self.device = device

    ##################################################################
    # Prediction part using the trained matrix and the processed data
    ##################################################################
    def test_processing(self, X_test, Y_test):
        """
        Process test data (X_test, Y_test) for prediction in the same way as training data.
        Uses the prior data processing from CKDR class instance.

        This function mainly performs the following:
            1. Process X_test to torch.Tensor if not already.
            2. Process Y_test to the same processing as training Y.
        
        Parameters
        ----------
        Y_test: np.ndarray or pd.DataFrame or torch.Tensor
            Test response data to be processed.

        Returns
        -------
        X_test_processed, Y_test_processed: torch.Tensor
            Processed test data in the same dtype and device as the CKDR instance.
        """
        X_test = X_test.reshape(-1, self.d)  # Ensure X_test is 2D
        
        # Process X_test to torch.Tensor if not already
        if not isinstance(X_test, torch.Tensor):
            X_test_processed = torch.tensor(X_test, dtype=self.dtype, device=self.device)
        else:
            X_test_processed = X_test.to(dtype=self.dtype, device=self.device)

        Y_test = Y_test.reshape(X_test.shape[0], -1)  # Ensure Y is 2D
        if self.type_Y == "binary":
            Y_new = np.zeros((Y_test.shape[0], 1))
            Y_new[Y_test == self.values[0]] = -1
            Y_new[Y_test == self.values[1]] = 1
            
        elif self.type_Y == "multiclass":
            # Convert to one-hot encoding; skip this if already one-hot encoded
            if set(np.unique(Y_test)) != {0., 1.}:
                Y_new = np.zeros((Y_test.shape[0], len(self.values))) 
                for i, value in enumerate(self.values):
                    Y_new[Y_test == value, i] = 1
            else:
                Y_new = Y_test
        else:
            Y_new = Y_test / self.Y_std # Scale Y_test using the same standard deviation as training Y
        
        if not isinstance(Y_new, torch.Tensor):
            Y_test_processed = torch.tensor(Y_new, dtype=self.dtype, device=self.device)
        else:
            Y_test_processed = Y_new.to(dtype=self.dtype, device=self.device)
        
        return X_test_processed, Y_test_processed


    def predict_loss(self, P, sigma, epsilon, X_test, Y_test, proj_result=False):
        """
        Prediction error of vector-valued KRR with intercept prediction.
            *** The error is computed in the RKHS H_Y directly ***
        
        Parameters
        ----------
        P: np.ndarray
            (Fitted) Projection matrix of shape (d, target_dim)
        sigma: float
            Kernel parameter used in the training phase
        epsilon: float
            Ridge regularization parameter used in the training phase
        X_test: torch.Tensor
            Test data (predictors) of shape (n_test, d)
        Y_test: torch.Tensor
            Test data (response) of shape (n_test, n_dim_response)
            *** X_test and Y_test must be processed a priori using the test_processing method ***
        proj_result: bool, False
            If True, the input dimension of X_test is the target_dim of dimension reduction

        Returns
        -------
        A float number that indicates the sum of squared errors
        """
        # one sample case
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
       
        # H_Y part
        if self.K_Y is not None:
            K_Y = self.K_Y
            K_YYtest = Gaussian_gram_matrix(Y_test, sigma=self.sigma_Y, Z2=self.Y)
            Tr = torch.trace(Gaussian_gram_matrix(Y_test, self.sigma_Y))
        else:
            K_Y = torch.matmul(self.Y, self.Y.T)
            K_YYtest = torch.matmul(Y_test, self.Y.T)
            Tr = torch.sum(Y_test * Y_test) # torch.trace(torch.matmul(Y_test, Y_test.T))

        # H_Z part       
        PX = torch.matmul(self.X, P.T)
        if proj_result:
            PX_test = torch.tensor(X_test, dtype=self.dtype, device=self.device)
        else:
            PX_test = torch.matmul(X_test, P.T)
        K_PX = Gaussian_gram_matrix(PX, sigma)
        G_PX_inv = torch.linalg.inv(center(K_PX) + self.n * epsilon * torch.eye(self.n, dtype=self.dtype, device=self.device))

        K_ZZtest = Gaussian_gram_matrix(PX_test, sigma, PX) - torch.mean(K_PX, dim=0, keepdim=True)
        K_ZZ_centered = K_ZZtest - torch.mean(K_ZZtest, dim=1, keepdim=True)

        mat = 1 / self.n + torch.matmul(K_ZZ_centered, G_PX_inv)

        Tr -= 2* torch.sum(K_YYtest * mat)
        Tr += torch.trace(torch.matmul(torch.matmul(mat, K_Y), mat.T))
        Tr = torch.squeeze(Tr).numpy()

        return Tr
        
    def predict(self, P, sigma, epsilon, X_test, proj_result=False):
        """
        Centered kernel ridge regression prediction.
        If k_Y is not a linear kernel, the final prediction is decoded into 

                argmin_y || k_Y(y, ) - yhat ||^2_{H_Y}

        Parameters
        ----------
        X_test: torch.tensor
            Shape = (n_test, n_variables)

        proj_result: bool, False
            If True, the input dimension of X_test is the target_dim of dimension reduction
        
        Yields
        ------
        numpy.ndarray of length n_test == X_test.shape[0]
        """
        # one sample case
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        if proj_result:
            PX_test = torch.tensor(X_test, dtype=self.dtype, device=self.device)
        else:
            PX_test = torch.matmul(X_test, P.T)
        
        PX = torch.matmul(self.X, P.T)
        K_PX = Gaussian_gram_matrix(PX, sigma)
        G_PX_inv = torch.linalg.inv(center(K_PX) + self.n * epsilon * torch.eye(self.n, dtype=self.dtype, device=self.device))

        K_ZZ = Gaussian_gram_matrix(PX_test, sigma, PX)

        fit_coef = torch.matmul(G_PX_inv, self.Y - torch.mean(self.Y, dim=0, keepdim=True))
        offset = torch.mean(self.Y, dim=0) - torch.matmul(
                                                    torch.mean(K_PX, dim=0, keepdim=True),
                                                    fit_coef)
        yhat = torch.matmul(K_ZZ, fit_coef) + offset    
        
        if self.type_Y == "binary":  ## reflects -1, 1 encoding
            yhat = torch.sign(yhat)
            yhat = torch.squeeze(yhat).numpy()
        elif self.type_Y == "multiclass":   ## reflects one-hot encoding
            max_indices = torch.argmax(yhat, dim=1)
            yhat = torch.nn.functional.one_hot(max_indices, num_classes=yhat.shape[1]).float().numpy()
        else:
            # Continuous response (ndim=2 output)
            yhat = yhat.numpy()

        return yhat
