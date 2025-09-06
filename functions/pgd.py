import torch
import numpy as np
from .objfunc import T, center
from .CKDR import CKDR

from scipy.spatial.distance import pdist


# Projected gradient descent of the objective function T
def train_ckdr(
    X, Y, target_dim, type_Y, epsilon=None, sigma=None, 
    initial_lr=1., armijo_c1=1e-4, armijo_beta=0.5, armijo_min_lr=1e-8,
    max_iter=10000, tol=1e-3, verbose=True, device='cpu', seed=None,
    P_init=None, # for code checking (will be removed later)
    ):
    """
    Performs projected gradient descent to optimize the CKDR objective function T.

    This function iteratively updates a projection matrix P to minimize T,
    which measures the conditional dependency between X and Y given a
    low-dimensional projection of X (X @ P).

    Args:
        X (array-like or torch.Tensor): Input data of shape (n_samples, n_features).
        Y (array-like or torch.Tensor): Target data of shape (n_samples, n_targets) or (n_samples,).
        target_dim (int): The desired dimensionality of the projected subspace (m).
        epsilon (float): Regularization parameter for the kernel on Y (K_Y).
        sigma (float): Kernel width parameter for the kernel on Z (K_Z), where Z = X @ P.
        type_Y (str): Type of the target variable. Used by the CKDR class to handle Y appropriately
                      (e.g., for kernel computation if Y is categorical).
                      Expected values: "binary", "multiclass", "gaussian", "continuous", or None.
        initial_lr (float, optional): Initial learning rate for the Armijo line search. Defaults to 1.0.
        armijo_c1 (float, optional): Sufficient decrease parameter for the Armijo condition. Defaults to 1e-3.
        armijo_beta (float, optional): Backtracking factor to decrease learning rate in Armijo search. Defaults to 0.5.
        armijo_max_iter (int, optional): Maximum iterations for the Armijo line search. Defaults to 100.
        max_iter (int, optional): Maximum number of gradient descent iterations. Defaults to 3000.
        tol (float, optional): Tolerance for convergence. Iteration stops if the change in P or
                             the objective function value is below this threshold. Defaults to 1e-5.
        verbose (bool, optional): Whether to print progress messages during training. Defaults to True.
        device (str, optional): The torch device ('cpu' or 'cuda') to perform computations on.
                                If None, it's inferred from X. Defaults to 'cpu'.
        seed (int or np.random.RandomState, optional): Seed for random number generation, primarily for
                                                       initializing P. Defaults to None.

    Returns:
        tuple: A tuple containing:        
            P (torch.Tensor): The optimized projection matrix P of shape (n_features, target_dim).
            obj_history (list): A list of objective function values T(P) at each iteration.
            CKDR_data (CKDR): The CKDR instance containing the processed data and the final P.
    """

    # --- Data Preparation and Initialization ---
    # Initialize CKDR data handling object
    CKDR_data = CKDR(X, Y, type_Y=type_Y, device=device)
    X, Y = CKDR_data.X, CKDR_data.Y # Use processed tensors

    # Recommended default parameters:
    epsilon = 1 / X.shape[0] if epsilon is None else epsilon
    sigma = np.median(pdist(X.numpy())) if sigma is None else sigma

    # Pre-compute centered Gram matrix for Y if K_Y is available (e.g., for Gaussian Y)
    # Otherwise, compute centered Gram matrix for continuous Y (linear kernel).
    if CKDR_data.K_Y is not None:
        G_Y = center(CKDR_data.K_Y) 
    else:
        Y_mean_centered = Y - torch.mean(Y, dim=0, keepdim=True)
        G_Y = torch.matmul(Y_mean_centered, Y_mean_centered.T)

    n, d = X.shape # n_samples, n_features
    m = target_dim # m is target_dim
    
    # Initialize RandomState for reproducible P initialization
    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed=seed)
    
    # Initialize P: column-wise uniform on the simplex using Dirichlet distribution.
    # P has shape (m, d) = (target_dim, n_features)
    # P_init = RS.dirichlet(np.ones(m), d).T if P_init is None else P_init
    P_init = RS.dirichlet(np.ones(m) * m ** 2, d).T if P_init is None else P_init # concentrated initialization
    P = torch.tensor(P_init, device=device, dtype=torch.float32) 
    P.requires_grad_(False)  # P will be updated manually; autograd is not used for its updates directly.

    # Ensure sigma, epsilon are on the correct device and dtype
    sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
    epsilon = torch.tensor(epsilon, device=device, dtype=torch.float32)

    obj_history = [] # To store objective function values

    if verbose:
        print(f"Starting Projected Gradient Descent on device: {device}")
        print(f"  n_samples={n}, n_features={d}, target_dim={m}")
        print(f"  sigma={sigma:.4f}, epsilon={epsilon:.4f}, initial_lr={initial_lr:.2e}, tol={tol:.1e}")

    lr_k = initial_lr # Initial learning rate for the Armijo line search

    # --- Projected Gradient Descent Loop ---
    for iteration in range(max_iter):
        # Calculate objective value and gradient at current P
        # Pass precomputed G_Y to avoid re-computation inside T
        current_obj, gradient = T(P, X, Y, epsilon=epsilon, sigma=sigma, 
                                    gradient=True, G_Y=G_Y)
        
        # Detach from computation graph to prevent history accumulation for manual updates
        gradient = gradient.detach()  
        current_obj = current_obj.detach()
        
        # --- Armijo Line Search for step size ---
        # Warm start with the lr_k value from the previous iteration
        P_new = P # Fallback if Armijo fails
        new_obj_candidate = current_obj # Initialize candidate objective value

        while lr_k > armijo_min_lr:
            # Projected gradient descent step
            P_candidate_projected = project_CDR(P - lr_k * gradient)

            # Evaluate objective at the projected candidate point
            new_obj_candidate = T(P_candidate_projected, X, Y, 
                                    epsilon=epsilon, sigma=sigma, G_Y=G_Y).detach()

            # Armijo-Goldstein condition for projected gradient descent:
            # T(P_projected) <= T(P_current) + c1 * <grad_T(P_current), P_projected - P_current>
            armijo_check_term = torch.sum(gradient * (P_candidate_projected - P))

            if new_obj_candidate <= current_obj + armijo_c1 * armijo_check_term:
                # Check for possible larger step size (to ensure we don't miss a better step)
                larger_lr = lr_k / armijo_beta
                while lr_k < 10:
                    P_larger_step = project_CDR(P - (larger_lr * gradient))
                    new_obj_larger_step = T(P_larger_step, X, Y, 
                                            epsilon=epsilon, sigma=sigma, G_Y=G_Y).detach()
                    new_check_term = torch.sum(gradient * (P_larger_step - P))

                    if new_obj_larger_step <= current_obj + armijo_c1 * new_check_term:
                        # larger lr is accepted
                        lr_k = larger_lr
                        P_candidate_projected = P_larger_step
                        new_obj_candidate = new_obj_larger_step

                        # Move to next larger_lr
                        larger_lr = larger_lr / armijo_beta

                    else:
                        break

                P_new = P_candidate_projected # Accept the step
                break # Exit Armijo loop, found suitable lr_k
            else:
                lr_k *= armijo_beta # Reduce learning rate
        
        if lr_k < armijo_min_lr:    # If learning rate too small, stop Armijo search
            if verbose:
                print(f"Iteration {iteration}: Armijo search exhausted. Stationary point found")
            P_new = P # Revert to P if no step found
            break 

        P_prev = P
        P = P_new.detach() # Update P for the next iteration

        # Record objective value after the step (after Armijo; reuse the computed, escaped value)
        current_obj_after_step = new_obj_candidate   

        # Calculate the norm of the difference between new and previous P
        norm_diff_P = torch.norm(P - P_prev).item()

        if verbose:
            if iteration % 50 == 0 or iteration == max_iter -1:
                print(f"Iter: {iteration:4d}, Obj: {current_obj_after_step.item():.6f}, LR: {lr_k:.2e}, update amount ||P_new - P_old||/LR: {norm_diff_P/lr_k:.2e}")

        # Store the objective value
        obj_history.append(current_obj_after_step.item())


        # --- Convergence Checks ---
        # Check 1: Small change in P
        if norm_diff_P/lr_k < tol:
            if verbose: print(f"Converged at iteration {iteration+1}: Change in P below tolerance.")
            break
        # Check 2: Small relative or absolute change in objective function value
        if iteration > 10:
            moving_avg = np.mean(obj_history[-10:]) #, 0)[0] 
            obj_rel_change = abs(obj_history[-1] - moving_avg) / moving_avg  # obj_history[-1][0]
            if obj_rel_change < tol * 1e-4:
                if verbose: print(f"Converged at iteration {iteration+1}: Change in objective value below tolerance.")
                break
    else: 
        # Executed if the main loop finishes without break (max_iter reached)
        if verbose: print(f"Finished after {max_iter} iterations (max_iter reached).")
    
    return P, obj_history, CKDR_data


@torch.jit.script
def project_CDR(A: torch.Tensor) -> torch.Tensor:
    """
    Vectorized projection of each column of a matrix onto the unit simplex.
    Ensure that the input matrix A is of type torch.Tensor with dtype torch.float32 a priori.
    
    This function is based on the efficient algorithm described in papers by
    Duchi et al. (2008) and Martins and Astudillo (2016).
    
    """

    z = torch.tensor(1.0, device=A.device, dtype=torch.float32)

    # Sort each column of A in descending order (O(dm log m); not significant when m is small)
    mu, _ = torch.sort(A, dim=0, descending=True)
    
    # Compute the cumulative sum of the sorted values
    mu_cumsum = torch.cumsum(mu, dim=0)
    
    # Create indices [1, 2, ..., m] for the condition check
    indices = torch.arange(1, A.shape[0] + 1, dtype=torch.float32, device=A.device).unsqueeze(1)
    
    # The condition for finding rho is that mu_k > (cumsum_k - z) / k.
    condition = mu * indices > mu_cumsum - z
    rho_count = torch.sum(condition, dim=0)
    
    # The index for mu_cumsum is rho_count - 1 (due to 0-indexing).
    col_indices = torch.arange(A.shape[1], device=A.device)
    
    # Gather the cumulative sums at the rho_count-1 index for each column
    # Note: rho_count is the 1-based index from the paper, so we use rho_count - 1 for 0-based tensor indexing
    theta = (mu_cumsum[rho_count - 1, col_indices] - z) / rho_count
    
    # Project using the calculated theta for each column
    projected_A = torch.maximum(A - theta.unsqueeze(0), torch.tensor(0.0, device=A.device))
    
    return projected_A