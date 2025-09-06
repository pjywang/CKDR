import torch
from typing import Optional


def T(P, X, Y, epsilon, sigma, gradient=False, G_Y=None):
    """
    The empirical objective T function.
    If gradient=True, also returns the gradient of T with respect to P.
    
    Parameters:
    P: Projection matrix (torch.Tensor of shape (dim, original_dim))
    X: Input data matrix (torch.Tensor of shape (n_samples, original_dim))
    Y: Response data matrix (torch.Tensor of shape (n_samples, n_outputs))
    epsilon: Ridge regularization parameter.
    sigma: Kernel width parameter for the Gaussian kernel.
    gradient: If True, compute and return the gradient of T with respect to P.
    G_Y: Optional precomputed centered Gram matrix for Y to save computation.

    Returns:
    If gradient=False: returns the objective value (torch.Tensor scalar).
    If gradient=True: returns a tuple (objective value, gradient matrix).

    Note:
    - This function uses PyTorch's JIT compilation for performance.
    """
    if gradient:
        return _T_obj_and_grad(P, X, Y, epsilon, sigma, G_Y)
    else:
        return _T_obj(P, X, Y, epsilon, sigma, G_Y)
            

# jit-compile helps performance slightly (~20% speedup)
@torch.jit.script
def _T_common(P: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, epsilon: float, sigma: float, G_Y: Optional[torch.Tensor] = None):
    """
    Common computations for the objective function and gradient.
    This function is JIT-scriptable.
    """
    n, d = X.shape

    # Dimension reduction calculation as P times X
    Z = torch.matmul(X, P.T)
    
    # Gaussian gram matrix calculation
    dist_matrix = torch.cdist(Z, Z, p=2.) ** 2
    K_PX = torch.exp(-dist_matrix / (2. * sigma ** 2))

    # Centering the kernel matrix
    mean_col = torch.mean(K_PX, dim=0, keepdim=True)
    mean_row = torch.mean(K_PX, dim=1, keepdim=True)
    mean_all = torch.mean(K_PX)
    G_PX = K_PX - mean_col - mean_row + mean_all

    G_PX_reg = G_PX + n * epsilon * torch.eye(n, device=P.device, dtype=P.dtype)

    # Centered Gram matrix for Y
    if G_Y is None:
        Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)
        g_y_final = torch.matmul(Y_centered, Y_centered.T)
    else:
        g_y_final = G_Y

    # Solve for product
    prod = torch.linalg.solve(G_PX_reg, g_y_final, left=False)

    # Calculate the trace objective function
    obj = torch.trace(prod)
    
    return obj, G_PX_reg, prod, K_PX, Z

@torch.jit.script
def _T_obj(P: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, epsilon: float, sigma: float, G_Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the objective function value.
    This function is JIT-scriptable.
    """
    obj, _, _, _, _ = _T_common(P, X, Y, epsilon, sigma, G_Y)
    return obj

@torch.jit.script
def _T_obj_and_grad(P: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, epsilon: float, sigma: float, G_Y: Optional[torch.Tensor] = None):
    """
    Computes the objective function value and its gradient.
    This function is JIT-scriptable.
    """
    obj, G_PX_reg, prod, K_PX, Z = _T_common(P, X, Y, epsilon, sigma, G_Y)
    
    # Gradient calculation
    Q = torch.linalg.solve(G_PX_reg, prod)
    S = Q * K_PX
    Laplacian_S = torch.diag(torch.sum(S, dim=1)) - S
    grad_T = 2 * torch.matmul(torch.matmul(Z.T, Laplacian_S), X) / (sigma ** 2)

    # Numerical adjustment to the gradient
    grad_T = grad_T - torch.mean(grad_T, dim=0, keepdim=True)

    return obj, grad_T


def center(K):
    """
    Center the matrix K to HKH where H = I - (1/n) * 1_n 1_n^T
    """
    mean_col = torch.mean(K, dim=0, keepdim=True)
    mean_row = torch.mean(K, dim=1, keepdim=True)
    mean_all = torch.mean(K)
    return K - mean_col - mean_row + mean_all


def Gaussian_gram_matrix(Z, sigma, Z2=None):
    """
    Calculate the Gaussian kernel Gram matrix for the given data Z (or between Z and Z2)
    """
    if Z2 is None:
        Z2 = Z
    if Z2.ndim < 2:
        # Reshape Z2 to be a 2D tensor if it is 1D
        Z2 = Z2.reshape(-1, 1)
    dist_matrix = torch.cdist(Z, Z2, p=2) ** 2  # Squared distance matrix
    K = torch.exp(-dist_matrix / (2 * sigma ** 2))
    return K

# Note: Kernel function can be replaced with any other kernel function as needed.
