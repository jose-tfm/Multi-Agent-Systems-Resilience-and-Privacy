import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize
import networkx as nx
# ------------------------------------------------------------

def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s


def build_Ap(N, A):
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        inds = {
            0: i,
            1: N + 3*i,
            2: N + 3*i+1,
            3: N + 3*i+2
        }
        # ——————————————————————
        # a)
       
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
     
        # ——————————————————————
        # b)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # c)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[3]] = 1
        Ap[inds[2], inds[3]] = 1
        Ap[inds[3], inds[0]] = 2
        '''
        # ——————————————————————
        # d)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        '''
        # ——————————————————————
        # e)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # f)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # g)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # h)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # i)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # j)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # k)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 2
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # l)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        '''
        # ——————————————————————
        # m)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
    return row_normalize(Ap)

# Reconstruct full P from flattened variables (to optimize) and row-normalize
def reconstruct_P(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> np.ndarray:
    p_full = np.zeros(mask_flat.size)
    p_full[mask_flat] = p_vars
    P = p_full.reshape(size, size)
    return row_normalize(P)

# Objective: second-largest eigenvalue magnitude of P
def objective_vars(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> float:
    P = reconstruct_P(p_vars, mask_flat, size)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

# Simulate P dynamics and return convergence metrics
def simulate_P_metrics(P: np.ndarray, x0: np.ndarray,
                       tol: float = 1e-4, max_steps: int = 250) -> dict:
    """
    Run the dynamics x_{k+1} = P x_k, with augmented x0.
    Returns:
      - lambda2: second-largest eigenvalue magnitude of P
      - conv_steps: iteration at which all entries are within tol of target
      - final_val: consensus value at that step (or None if not reached)
      - X: the trajectory matrix (size x (max_steps+1))
    """
    size = P.shape[0]
    N = x0.size

    # Stationary left eigenvector of P
    w, V = eig(P.T)
    idx = np.argmin(np.abs(w - 1))
    v0 = np.real(V[:, idx])
    v0 /= v0.sum()

    # Build augmented initial state: first N entries = x0, next entries = x0[node]/3
    target = x0.mean()
    x0_aug = np.zeros(size)
    x0_aug[:N] = x0
    for i in range(N, size):
        x0_aug[i] = x0[(i - N) % N] / 3
    x0_aug *= target / (v0 @ x0_aug)

    # Simulate
    X = np.zeros((size, max_steps + 1))
    X[:, 0] = x0_aug
    conv_step = None

    for k in range(max_steps):
        X[:, k + 1] = P @ X[:, k]
        if conv_step is None and np.max(np.abs(X[:, k + 1] - target)) < tol:
            conv_step = k + 1

    final_val = X[0, conv_step] if conv_step is not None else None
    lambda2 = np.sort(np.abs(eigvals(P)))[-2]

    return {
        'lambda2': lambda2,
        'conv_steps': conv_step,
        'final_val': final_val,
        'X': X
    }

# Local minimality check via small random perturbations on p_vars
def local_sanity_check_vars(p_star: np.ndarray, mask_flat: np.ndarray, size: int,
                            n_checks: int = 150, eps: float = 1e-3) -> bool:
    """
    Check if p_star is a numerical local minimizer for objective_vars.
    For n_checks times, perturb p_star in [-eps, +eps]^M,
    project back to [0.01, ∞), and verify no smaller objective arises.
    """
    base_val = objective_vars(p_star, mask_flat, size)
    M = p_star.size

    for _ in range(n_checks):
        delta = np.random.uniform(-eps, eps, size=M)
        p_test = p_star + delta
        p_test = np.clip(p_test, 0.01, None)

        trial_val = objective_vars(p_test, mask_flat, size)
        if trial_val < base_val - 1e-8:
            print(f"  ↓ Found descent: objective {base_val:.6f} → {trial_val:.6f}")
            return False

    print(f"  ✓ No descent found in {n_checks} random perturbations.")
    return True

# ------------------------------------------------------------
# Build initial A via a random connected graph

N = 50  # number of original nodes

# 1. Set a random seed for reproducibility
seed = 4
np.random.seed(seed)

# 2. Generate a random Erdos–Rényi graph until it is connected
p_edge = 0.8  # edge-creation probability
while True:
    G = nx.erdos_renyi_graph(N, p_edge, seed=seed)
    if nx.is_connected(G):
        break

# 3. Assign a random positive weight to each existing edge
for u, v in G.edges():
    G[u][v]['weight'] = np.random.rand()

# 4. Visualize the initial graph
pos = nx.spring_layout(G, seed=seed)
plt.figure(figsize=(5, 5))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
scaled_widths = [2.0 * w for w in edge_weights]
nx.draw_networkx_edges(G, pos, width=scaled_widths, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title(f"Initial Random Connected Graph (N={N})")
plt.axis('off')
plt.tight_layout()
plt.show()

# 5. Extract the weighted adjacency matrix and row-normalize to get A
raw_A = nx.to_numpy_array(G, nodelist=range(N), weight='weight', dtype=float)
A = row_normalize(raw_A)

# 6. Define an arbitrary initial state x0
x0 = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])

# 7. Build the original augmented matrix Ap0
Ap0 = build_Ap(N, A)
size = Ap0.shape[0]

# 8. Create mask_flat over the flattened Ap0
mask_flat = Ap0.flatten() > 0

# 9. Extract the initial free-variable vector vars0
vars0 = Ap0.flatten()[mask_flat]

# 10. Set box constraints (each var ≥ 0.01)
bounds = [(0.01, None)] * vars0.size

# ------------------------------------------------------------
# 11. Optimize over the free entries of P
res = minimize(
    lambda v: objective_vars(v, mask_flat, size),
    vars0,
    method='SLSQP',
    bounds=bounds,
    options={'ftol': 1e-9, 'maxiter': 500}
)

vars_opt = res.x
P_opt = reconstruct_P(vars_opt, mask_flat, size)

# 12. Compute metrics for original Ap0 and optimized P_opt
metrics_orig = simulate_P_metrics(Ap0, x0)
metrics_opt = simulate_P_metrics(P_opt, x0)

# 13. Print numerical results (handle None for final_val)
print("=== Original Ap0 Metrics ===")
print(f"Second-largest eigenvalue: {metrics_orig['lambda2']:.6f}")
if metrics_orig['conv_steps'] is not None:
    print(f"Converged in {metrics_orig['conv_steps']} steps (tol=1e-4)")
    print(f"Consensus value: {metrics_orig['final_val']:.6f}\n")
else:
    print("Did not converge within max_steps; no consensus value.\n")

print("=== Optimized P_opt Metrics ===")
print(f"Second-largest eigenvalue: {metrics_opt['lambda2']:.6f}")
if metrics_opt['conv_steps'] is not None:
    print(f"Converged in {metrics_opt['conv_steps']} steps (tol=1e-4)")
    print(f"Consensus value: {metrics_opt['final_val']:.6f}\n")
else:
    print("Did not converge within max_steps; no consensus value.\n")

np.set_printoptions(precision=3, suppress=True)
print("Original augmented matrix (Ap0):")
print(Ap0)

print("\nOptimized matrix (P_opt):")
print(P_opt)

# 14. Perform local minimality test on the optimized vars_opt
print("\nChecking whether optimized vars form a local minimum...")
is_local_min = local_sanity_check_vars(vars_opt, mask_flat, size,
                                       n_checks=200, eps=1e-3)
if is_local_min:
    print("Result: vars_opt appears to be a numerical local minimum.")
else:
    print("Result: A descent direction was found; not a local minimum.")

# 15. Plot convergence trajectories side by side
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# 15a. Original Ap0 convergence
X_orig = metrics_orig['X']
for i in range(size):
    axes[0].plot(X_orig[i], label=f'$x_{{{i+1}}}$')
axes[0].axhline(x0.mean(), ls='--', color='k')
axes[0].set_title('Original Ap0 Convergence Trajectory')
axes[0].set_xlabel('Iteration k')
axes[0].set_ylabel('State Value')
axes[0].legend(ncol=4, fontsize='small')
axes[0].grid()

# 15b. Optimized P_opt convergence
X_opt = metrics_opt['X']
for i in range(size):
    axes[1].plot(X_opt[i], label=f'$x_{{{i+1}}}$')
axes[1].axhline(x0.mean(), ls='--', color='k')
axes[1].set_title('Optimized P_opt Convergence Trajectory')
axes[1].set_xlabel('Iteration k')
axes[1].set_ylabel('State Value')
axes[1].legend(ncol=4, fontsize='small')
axes[1].grid()

plt.tight_layout()
plt.show()