import torch
from typing import Optional


def T(P, X, Y, epsilon, sigma, gradient=False, G_Y=None, Y_centered=None, I_n=None):
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
    G_Y: Optional precomputed centered Gram matrix for kernel-valued Y.
    Y_centered: Optional centered response matrix. If given, it is used directly
                without re-centering or forming the full Gram matrix.
    I_n: Optional cached n x n identity matrix.

    Returns:
    If gradient=False: returns the objective value (torch.Tensor scalar).
    If gradient=True: returns a tuple (objective value, gradient matrix).

    Note:
    - This function uses PyTorch's JIT compilation for performance.
    """
    if gradient:
        return _T_obj_and_grad(P, X, Y, epsilon, sigma, G_Y, Y_centered, I_n)
    else:
        return _T_obj(P, X, Y, epsilon, sigma, G_Y, Y_centered, I_n)
            

# jit-compile helps performance slightly (~20% speedup)
@torch.jit.script
def _T_kernel_common(
        P: torch.Tensor, X: torch.Tensor, epsilon: float, sigma: float,
        I_n: Optional[torch.Tensor] = None,
        ):
    """
    Common kernel computations for the objective function and gradient.
    """
    n, d = X.shape

    # Dimension reduction calculation as P times X
    Z = torch.matmul(X, P.T)
    
    # Gaussian gram matrix calculation
    Z_norm = torch.sum(Z * Z, dim=1, keepdim=True)
    dist_matrix = Z_norm + Z_norm.T - 2. * torch.matmul(Z, Z.T)
    dist_matrix = torch.clamp(dist_matrix, min=0.)
    # dist_matrix = torch.cdist(Z, Z, p=2) ** 2  # Squared distance matrix (slower)
    K_PX = torch.exp(-dist_matrix / (2. * sigma ** 2))

    # Centering the kernel matrix
    mean_col = torch.mean(K_PX, dim=0, keepdim=True)
    mean_row = torch.mean(K_PX, dim=1, keepdim=True)
    mean_all = torch.mean(K_PX)
    G_PX = K_PX - mean_col - mean_row + mean_all

    eye = torch.eye(n, device=P.device, dtype=P.dtype) if I_n is None else I_n
    G_PX_reg = G_PX + n * epsilon * eye

    return G_PX_reg, K_PX, Z

@torch.jit.script
def _T_obj(
        P: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
        epsilon: float, sigma: float,
        G_Y: Optional[torch.Tensor] = None,
        Y_centered: Optional[torch.Tensor] = None,
        I_n: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
    """
    Computes the objective function value.
    This function is JIT-scriptable.
    """
    G_PX_reg, _, _ = _T_kernel_common(P, X, epsilon, sigma, I_n)
    # G_PX_reg is SPD. Cholesky gives A = L L^T; cholesky_solve(B, L)
    # returns A^{-1} B without forming the inverse.
    chol = torch.linalg.cholesky(G_PX_reg)

    if Y_centered is not None:
        # KRR coefficient matrix A^{-1} Y_c.
        coef = torch.cholesky_solve(Y_centered, chol)
        return torch.sum(Y_centered * coef)

    if G_Y is None:
        Y_centered_from_Y = Y - torch.mean(Y, dim=0, keepdim=True)
        # KRR coefficient matrix A^{-1} Y_c.
        coef = torch.cholesky_solve(Y_centered_from_Y, chol)
        return torch.sum(Y_centered_from_Y * coef)

    # Trace objective tr(G_Y A^{-1}) = tr(A^{-1} G_Y).
    prod = torch.cholesky_solve(G_Y, chol)
    return torch.trace(prod)

@torch.jit.script
def _T_obj_and_grad(
        P: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
        epsilon: float, sigma: float,
        G_Y: Optional[torch.Tensor] = None,
        Y_centered: Optional[torch.Tensor] = None,
        I_n: Optional[torch.Tensor] = None,
        ):
    """
    Computes the objective function value and its gradient.
    This function is JIT-scriptable.
    """
    G_PX_reg, K_PX, Z = _T_kernel_common(P, X, epsilon, sigma, I_n)
    # G_PX_reg is SPD. Cholesky gives A = L L^T; cholesky_solve(B, L)
    # returns A^{-1} B without forming the inverse.
    chol = torch.linalg.cholesky(G_PX_reg)

    if Y_centered is not None:
        # KRR coefficient matrix A^{-1} Y_c.
        coef = torch.cholesky_solve(Y_centered, chol)
        obj = torch.sum(Y_centered * coef)
        Q = torch.matmul(coef, coef.T)
    elif G_Y is None:
        Y_centered_from_Y = Y - torch.mean(Y, dim=0, keepdim=True)
        # KRR coefficient matrix A^{-1} Y_c.
        coef = torch.cholesky_solve(Y_centered_from_Y, chol)
        obj = torch.sum(Y_centered_from_Y * coef)
        Q = torch.matmul(coef, coef.T)
    else:
        # prod = A^{-1} G_Y gives the trace; Q = A^{-1} G_Y A^{-1}.
        prod = torch.cholesky_solve(G_Y, chol)
        obj = torch.trace(prod)
        Q = torch.cholesky_solve(prod.T, chol)
    
    # Gradient calculation
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
    Z_norm = torch.sum(Z * Z, dim=1, keepdim=True)
    Z2_norm = torch.sum(Z2 * Z2, dim=1, keepdim=True).T
    dist_matrix = Z_norm + Z2_norm - 2 * torch.matmul(Z, Z2.T)
    dist_matrix = torch.clamp(dist_matrix, min=0.)
    # dist_matrix = torch.cdist(Z, Z2, p=2) ** 2  # Squared distance matrix (slower)
    K = torch.exp(-dist_matrix / (2 * sigma ** 2))
    return K

# Note: Kernel function can be replaced with any other kernel function as needed.
