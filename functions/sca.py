import math
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist

from .CKDR import CKDR
from .objfunc import T, center
from .pgd import project_CDR


def train_ckdr(
        X, Y, target_dim, type_Y, epsilon=None, sigma=None,
        armijo_c1=1e-4, armijo_beta=0.5, armijo_min_lr=1e-8,
        max_iter=2000, tol=1e-6, verbose=True, device='cpu', seed=None,
        P_init=None, inner_tol=1e-7, inner_max_iter=50,
        ):
    """
    Train CKDR (column-stochastic) matrix P with successive convex approximation (SCA).

    At each outer iteration this routine builds a convex quadratic surrogate
    around the current P, approximately minimizes that surrogate over the CDR
    constraint set, and accepts the step by Armijo backtracking on the original
    CKDR objective.

    Parameters
    ----------
    X, Y : array-like
        Compositional predictors and response.
    target_dim : int
        Number of rows of P, i.e. the target CKDR dimension.
    type_Y : {"binary", "multiclass", "continuous", None}
        Response type passed to the CKDR data processor.
    epsilon : float or None
        KRR regularization. If None, use 1 / n.
    sigma : float or None
        Gaussian kernel bandwidth on projected data Z = X @ P.T.
        If None, use the median pairwise distance in X.
    P_init : array-like or None
        Optional initial projection matrix with shape (target_dim, n_features).
    inner_tol, inner_max_iter
        Stopping controls for the inner accelerated projected-gradient solver.

    Returns
    -------
    P : torch.Tensor
        Fitted row-stochastic projection matrix with shape
        (target_dim, n_features).
    obj_history : np.ndarray
        Original CKDR objective values along accepted outer iterations.
    CKDR_data : CKDR
        Processed CKDR data object used by downstream prediction routines.
    """
    CKDR_data = CKDR(X, Y, type_Y=type_Y, device=device)
    X, Y = CKDR_data.X, CKDR_data.Y

    epsilon = float(1 / X.shape[0] if epsilon is None else epsilon)
    if sigma is None:
        sigma = float(np.median(pdist(X.detach().cpu().numpy())))
    else:
        sigma = float(sigma)

    response, use_low_rank_response = _response_representation(CKDR_data, Y)

    n, d = X.shape
    m = target_dim
    I_n = torch.eye(n, device=X.device, dtype=X.dtype)
    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed=seed)
    P_init = RS.dirichlet(np.ones(m) * m ** 2, d).T if P_init is None else P_init
    P = torch.tensor(P_init, device=device, dtype=X.dtype)
    P.requires_grad_(False)

    G_Y_arg = None if use_low_rank_response else response
    Y_c_arg = response if use_low_rank_response else None

    low_rank_state = None
    if use_low_rank_response:
        low_rank_state = _low_rank_state(P, X, response, epsilon, sigma, I_n)
        current_obj = low_rank_state[3].detach()
    else:
        current_obj = T(
            P, X, Y, epsilon=epsilon, sigma=sigma,
            G_Y=G_Y_arg, Y_centered=Y_c_arg, I_n=I_n,
        ).detach()
    obj_history = np.zeros(max_iter)

    if verbose:
        print(f"Starting SCA CKDR on device: {device}")
        print(f"  n_samples={n}, n_features={d}, target_dim={m}")
        print(f"  sigma={sigma:.4f}, epsilon={epsilon:.4f}, tol={tol:.1e}, inner_tol={inner_tol:.1e}")

    outer_lr = 1.
    for iteration in range(max_iter):
        # Build the local quadratic majorizer Q_t(P) of the CKDR objective.
        if use_low_rank_response:
            a, S_blocks, grad_base = _build_low_rank_surrogate_from_state(
                low_rank_state[0], low_rank_state[1], low_rank_state[2],
                X, epsilon,
            )
        else:
            a, S_blocks, grad_base = _build_surrogate(
                P, X, response, epsilon, sigma, use_low_rank_response, I_n,
            )
        P_tilde, inner_residual, inner_iters = _solve_surrogate_apg(
            P, X, a, S_blocks,
            beta=armijo_beta,
            tol=inner_tol,
            max_iter=inner_max_iter,
        )

        # The surrogate minimizer defines the candidate descent direction.
        direction = P_tilde - P
        direction_norm = torch.norm(direction).item()

        grad_dot = torch.sum(grad_base * direction).detach()
        if grad_dot >= 0:
            if verbose:
                print(f"Stopped at iteration {iteration}: surrogate direction is not descending.")
            break

        lr = min(outer_lr / armijo_beta, 1.)
        accepted = False
        candidate_obj = current_obj
        P_candidate = P
        candidate_state = low_rank_state
        # Backtrack on the true CKDR objective, not on the surrogate.
        while lr >= armijo_min_lr:
            P_candidate = P + lr * direction
            if use_low_rank_response:
                candidate_state = _low_rank_state(
                    P_candidate, X, response, epsilon, sigma, I_n,
                )
                candidate_obj = candidate_state[3].detach()
            else:
                candidate_obj = T(
                    P_candidate, X, Y, epsilon=epsilon, sigma=sigma,
                    G_Y=G_Y_arg, Y_centered=Y_c_arg, I_n=I_n,
                ).detach()
            if candidate_obj <= current_obj + armijo_c1 * lr * grad_dot:
                accepted = True
                break
            lr *= armijo_beta

        if not accepted:
            if verbose:
                print(f"Iteration {iteration}: Armijo search exhausted.")
            break

        # # Check actual step size: if Armijo shrunk the step to negligible,
        # # the iterate is effectively stationary even though ||D|| > tol.
        # if lr * direction_norm <= tol:
        #     if verbose:
        #         print(f"Converged at iteration {iteration}: actual step lr*||D|| below tolerance.")
        #     break

        P = P_candidate.detach()
        if use_low_rank_response:
            low_rank_state = candidate_state
        current_obj = candidate_obj
        outer_lr = lr
        obj_history[iteration] = current_obj.item()
        # obj_history.append(current_obj.item())

        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(
                f"Iter: {iteration:4d}, Obj: {current_obj.item():.6f}, "
                f"LR: {lr:.2e}, ||D||: {direction_norm:.2e}, "
                f"inner residual: {inner_residual:.2e}, inner iters: {inner_iters}"
            )

        # Outer objective stagnation: if the relative change over the last
        # 20 iterations is negligible, the algorithm has effectively converged.
        # This is needed because inner_tol >> tol can create a noise floor in
        # ||D|| that prevents the step-size criterion from ever firing.
        stagnation_window = 20
        if iteration > stagnation_window:
            avg_obj = obj_history[iteration-stagnation_window:iteration].mean()
            rel_change = abs(obj_history[iteration] - avg_obj) / max(abs(obj_history[iteration]), 1.)
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}: objective stagnation "
                          f"(relative change {rel_change:.2e} over last {stagnation_window} iters).")
                break
    else:
        if verbose:
            print(f"Finished after {max_iter} iterations (max_iter reached).")

    return P, obj_history, CKDR_data


def _response_representation(CKDR_data, Y):
    """Return the response representation used by the SCA surrogate.

    Categorical responses use the centered response Gram matrix K_Y. Continuous
    responses use centered Y directly, which enables a cheaper low-rank KRR
    computation without forming K_Y.
    """
    if CKDR_data.K_Y is not None:
        return center(CKDR_data.K_Y).to(device=Y.device, dtype=Y.dtype), False

    return Y - torch.mean(Y, dim=0, keepdim=True), True


@torch.jit.script
def _low_rank_state(
        P: torch.Tensor, X: torch.Tensor, response: torch.Tensor,
        epsilon: float, sigma: float, I_n: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the refitted low-rank KRR state at P for continuous responses.

    Returns Z, C, B = R Y_c, and the objective value tr(Y_c^T R Y_c).
    """
    n = X.shape[0]

    # z_i = P x_i. With row-stored data, this is X @ P^T.
    Z = torch.matmul(X, P.T)

    # Gaussian Gram matrix K_Z on the reduced data.
    Z_norm = torch.sum(Z * Z, dim=1, keepdim=True)
    dist_matrix = Z_norm + Z_norm.T - 2. * torch.matmul(Z, Z.T)
    dist_matrix = torch.clamp(dist_matrix, min=0.)
    K_Z = torch.exp(-dist_matrix / (2. * sigma ** 2))

    # Centered Gram matrix G_Z = H K_Z H.
    G_Z = K_Z - torch.mean(K_Z, dim=0, keepdim=True)
    G_Z = G_Z - torch.mean(K_Z, dim=1, keepdim=True) + torch.mean(K_Z)

    # R = (G_Z + n epsilon I_n)^(-1).
    reg = G_Z + n * epsilon * I_n

    # Row i stores the diagonal of C_i = sigma^{-2} diag(K_i1, ..., K_in).
    C = K_Z / (sigma ** 2)

    # reg is SPD. Cholesky gives reg = L L^T; cholesky_solve(B, L)
    # returns reg^{-1} B without forming the inverse.
    chol = torch.linalg.cholesky(reg)
    B = torch.cholesky_solve(response, chol)

    obj = torch.sum(response * B)
    return Z, C, B, obj


@torch.jit.script
def _build_low_rank_surrogate_from_state(
        Z: torch.Tensor, C: torch.Tensor, B: torch.Tensor,
        X: torch.Tensor, epsilon: float
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the quadratic surrogate from a cached low-rank KRR state.

    Returns a_i, S_i, and grad Q_t(P_t). These define the convex quadratic
    surrogate used by the inner APG solver.
    """
    n = X.shape[0]

    # The low-rank branch only needs W_B = W @ B. Expanding the
    # difference gives W_B[i,k,p] = sum_j C[i,j](Z[i,k]-Z[j,k])B[j,p].
    C_B = torch.matmul(C, B)
    Z_B = Z.unsqueeze(2) * B.unsqueeze(1)
    C_Z_B = torch.einsum('ij,jkp->ikp', C, Z_B)
    W_B = Z.unsqueeze(2) * C_B.unsqueeze(1) - C_Z_B

    a = n * torch.sum(W_B * B.unsqueeze(1), dim=2)
    S_blocks = torch.bmm(W_B, W_B.transpose(1, 2)) / epsilon

    # g_t = grad Q_t(P_t) = (2/n) sum_i a_i x_i^T.
    grad_base = (2. / n) * torch.matmul(a.T, X)
    return a, S_blocks, grad_base


@torch.jit.script
def _build_surrogate(
        P: torch.Tensor, X: torch.Tensor, response: torch.Tensor,
        epsilon: float, sigma: float, use_low_rank_response: bool,
        I_n: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the explicit quadratic surrogate for the current iterate.

    Returns a_i, S_i, and the base gradient g_t = grad Q_t(P_t).
    """
    n = X.shape[0]

    if use_low_rank_response:
        Z, C, B, _ = _low_rank_state(P, X, response, epsilon, sigma, I_n)
        return _build_low_rank_surrogate_from_state(Z, C, B, X, epsilon)

    # z_i^t = P_t x_i. With row-stored data, this is X @ P_t^T.
    Z = torch.matmul(X, P.T)

    # Gaussian Gram matrix K_Z^t on the reduced data.
    Z_norm = torch.sum(Z * Z, dim=1, keepdim=True)
    dist_matrix = Z_norm + Z_norm.T - 2. * torch.matmul(Z, Z.T)
    dist_matrix = torch.clamp(dist_matrix, min=0.)
    K_Z = torch.exp(-dist_matrix / (2. * sigma ** 2))

    # Centered Gram matrix G_Z^t = H K_Z^t H.
    G_Z = K_Z - torch.mean(K_Z, dim=0, keepdim=True)
    G_Z = G_Z - torch.mean(K_Z, dim=1, keepdim=True) + torch.mean(K_Z)

    # R_t = (G_Z^t + n epsilon I_n)^(-1).
    reg = G_Z + n * epsilon * I_n

    # Row i stores the diagonal of C_i^t = sigma^{-2} diag(K_i1, ..., K_in).
    C = K_Z / (sigma ** 2)

    # reg is SPD. Cholesky gives reg = L L^T; cholesky_solve(B, L)
    # returns reg^{-1} B without forming the inverse.
    chol = torch.linalg.cholesky(reg)

    # --- Vectorised surrogate construction ---
    # diff[i, j, k] = Z[i,k] - Z[j,k]: pairwise differences in the
    # m-dimensional reduced space.  Shape (n, n, m).
    diff = Z.unsqueeze(1) - Z.unsqueeze(0)

    # V_C_all[i, j, k] = C[i,j] * diff[i,j,k]
    #                   = sigma^{-2} K_ij^t (z_i - z_j)[k]
    # This is the batched analogue of the per-sample V_i^t C_i^t matrix.
    # C is (n, n); unsqueeze(2) broadcasts it over the m-dim. Shape (n, n, m).
    V_C_all = diff * C.unsqueeze(2)

    # W[i, k, j] = V_C_all[i, j, k]. Transposing the last two dims gives
    # a batch of (m, n) matrices suitable for matmul / bmm. Shape (n, m, n).
    W = V_C_all.transpose(1, 2)

    # R is reg^{-1}, used here only to form A_alpha = R G_Y R^T.
    R = torch.cholesky_solve(I_n, chol)
    A_alpha = torch.matmul(torch.matmul(R, response), R.T)

    # a_i for the codebase objective T. The note's profiled loss is
    # epsilon * T, so a_i is divided by epsilon relative to the note.
    a = n * torch.einsum('ijk, ji -> ik', V_C_all, A_alpha)

    # S_blocks[i] = (1/epsilon) W[i] @ A_alpha @ W[i]^T.
    WA = torch.matmul(W, A_alpha)
    S_blocks = torch.bmm(WA, V_C_all) / epsilon

    # g_t = grad Q_t(P_t) = (2/n) sum_i a_i x_i^T.
    grad_base = (2. / n) * torch.matmul(a.T, X)
    return a, S_blocks, grad_base


@torch.jit.script
def _surrogate_value(
        P: torch.Tensor, P_base: torch.Tensor, X: torch.Tensor,
        a: torch.Tensor, S_blocks: torch.Tensor
        ) -> torch.Tensor:
    """
    Evaluate Q_t(P) without the additive constant.

    The constant does not affect the inner APG updates or stopping rule.
    """

    n = X.shape[0]

    # delta_z[i] = (P - P_t) x_i.
    delta_z = torch.matmul(X, (P - P_base).T)

    # curvature_delta[i] = S_i delta_z[i].
    # PyTorch's bmm multiplies batches of matrices, so delta_z is reshaped
    # from (n, m) to (n, m, 1). After multiplication, squeeze removes the
    # trailing singleton dimension and returns the result to shape (n, m).
    curvature_delta = torch.bmm(S_blocks, delta_z.unsqueeze(2)).squeeze(2)

    # Q_t(P) without the additive constant.
    linear = 2. * torch.sum(a * delta_z)
    quadratic = torch.sum(delta_z * curvature_delta)
    return (linear + quadratic) / n


@torch.jit.script
def _surrogate_value_and_grad(
        P: torch.Tensor, P_base: torch.Tensor, X: torch.Tensor,
        a: torch.Tensor, S_blocks: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the surrogate objective and its gradient in a single pass,
    sharing the expensive delta_z and bmm computations.
    """
    n = X.shape[0]

    # delta_z[i] = (P - P_t) x_i.
    delta_z = torch.matmul(X, (P - P_base).T)

    # curvature_delta[i] = S_i delta_z[i].
    # PyTorch's bmm multiplies batches of matrices, so delta_z is reshaped
    # from (n, m) to (n, m, 1). After multiplication, squeeze removes the
    # trailing singleton dimension and returns the result to shape (n, m).
    curvature_delta = torch.bmm(S_blocks, delta_z.unsqueeze(2)).squeeze(2)

    # Q_t(P) without the additive constant.
    linear = 2. * torch.sum(a * delta_z)
    quadratic = torch.sum(delta_z * curvature_delta)
    value = (linear + quadratic) / n

    # grad Q_t(P) = (2/n) sum_i {a_i + S_i delta_z[i]} x_i^T.
    grad = (2. / n) * torch.matmul((a + curvature_delta).T, X)

    return value, grad


@torch.jit.script
def _solve_surrogate_apg(
        P_base: torch.Tensor, X: torch.Tensor, a: torch.Tensor,
        S_blocks: torch.Tensor, beta: float, tol: float, max_iter: int
        ) -> Tuple[torch.Tensor, float, int]:
    """
    Solves the convex surrogate using accelerated projected gradient (APG).
    """
    P_current = P_base.clone()
    Y_current = P_current.clone()
    theta = 1.
    lr = 1.
    residual = math.inf
    iterations_done: int = 0
    q_current = _surrogate_value(P_current, P_base, X, a, S_blocks)

    # Circular buffer for recent surrogate values (moving-average stopping).
    window: int = 5
    q_history = torch.zeros(window, device=P_base.device, dtype=P_base.dtype)
    q_history[0] = q_current.item()

    for iteration in range(max_iter):
        # APG gradient step at the extrapolated point.
        q_y, grad_y = _surrogate_value_and_grad(Y_current, P_base, X, a, S_blocks)
        trial_lr = lr if iteration == 0 else lr / beta

        # Backtrack until the quadratic upper bound condition is met.
        P_next = P_current
        q_next = q_current
        while trial_lr > 1e-12:
            P_next = project_CDR(Y_current - trial_lr * grad_y)
            step = P_next - Y_current
            q_next = _surrogate_value(P_next, P_base, X, a, S_blocks)
            upper = q_y + torch.sum(grad_y * step) + torch.sum(step * step) / (2. * trial_lr)
            if bool(q_next <= upper):
                break
            trial_lr *= beta

        if trial_lr <= 1e-12:
            break

        # Adaptive restart (O'Donoghue & Candès, 2015): if the surrogate
        # value at P_next is worse than at P_current, momentum overshot.
        # Reset theta to 1 and skip extrapolation for this step.
        if bool(q_next > q_current):
            theta = 1.
            Y_next = P_next.clone()
        else:
            # Nesterov acceleration.
            theta_next = (1. + math.sqrt(1. + 4. * theta ** 2)) / 2.
            Y_next = P_next + ((theta - 1.) / theta_next) * (P_next - P_current)
            theta = theta_next

        P_current = P_next.detach()
        Y_current = Y_next.detach()
        q_current = q_next
        lr = trial_lr
        iterations_done = iteration + 1

        # Store q value in circular buffer.
        q_val = q_current.item()
        q_history[(iteration + 1) % window] = q_val

        # Check convergence by moving-average stagnation, robust to APG's
        # non-monotone transients. The projected-gradient residual is computed
        # only once at return for logging.
        if iteration >= window:
            moving_avg = torch.mean(q_history).item()
            stagnation = moving_avg - q_val
            if stagnation >= 0. and stagnation <= tol:
                _, grad_final = _surrogate_value_and_grad(
                    P_current, P_base, X, a, S_blocks
                )
                projected = project_CDR(P_current - grad_final)
                residual = torch.norm(P_current - projected).item()
                return P_current, residual, iteration + 1

    _, grad_final = _surrogate_value_and_grad(
        P_current, P_base, X, a, S_blocks
    )
    projected = project_CDR(P_current - grad_final)
    residual = torch.norm(P_current - projected).item()
    return P_current, residual, iterations_done
