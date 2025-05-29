import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize

# ------------------------------------------------------------
# Row-stochastic normalization for any square matrix
# Adds tiny epsilon to avoid division by zero
def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

# Existing function to build the augmented matrix P from A
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

# Objective: second-largest eigenvalue magnitude of P
# Only optimize nonzero entries of initial P

def reconstruct_P(p_vars, mask_flat, size):
    # Build full flattened P, respecting zeros
    p_full = np.zeros(mask_flat.size)
    p_full[mask_flat] = p_vars
    P = p_full.reshape(size, size)
    return row_normalize(P)

def objective_vars(p_vars, mask_flat, size):
    P = reconstruct_P(p_vars, mask_flat, size)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

# Simulate P dynamics and return metrics
def simulate_P_metrics(P: np.ndarray, x0: np.ndarray,
                       tol=1e-4, max_steps=30) -> dict:
    size = P.shape[0]
    N = x0.size
    # left stationary eigenvector
    w, V = eig(P.T)
    v0 = np.real(V[:, np.argmin(np.abs(w - 1))])
    v0 /= v0.sum()
    # build augmented initial state
    target = x0.mean()
    x0_aug = np.zeros(size)
    x0_aug[:N] = x0
    for i in range(N, size):
        x0_aug[i] = x0[(i - N) % N] / 3
    x0_aug *= target / (v0 @ x0_aug)
    # simulate
    X = np.zeros((size, max_steps+1))
    X[:, 0] = x0_aug
    conv_step = None
    for k in range(max_steps):
        X[:, k+1] = P @ X[:, k]
        if conv_step is None and np.max(np.abs(X[:, k+1] - target)) < tol:
            conv_step = k+1
    final_val = X[0, conv_step] if conv_step is not None else None
    second_eig = np.sort(np.abs(eigvals(P)))[-2]
    return {'lambda2': second_eig,
            'conv_steps': conv_step,
            'final_val': final_val,
            'X': X}

# --- Main ---


N = 3
A = np.array([
    [0,   0.3, 0.7],
    [0.3, 0,   0.7],
    [0.3, 0.7, 0  ]
], dtype=float)
x0 = np.array([0.5, 1/3, 0.2])

'''
N = 11
# Original adjacency A and initial state x0
A = np.array([
    [0.   , 0.75 , 0.20 , 0.03 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.75 , 0.20 , 0.03 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.02 , 0.   , 0.   , 0.75 , 0.20 , 0.03 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 ],
    [0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 ],
    [0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 ],
    [0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   ]
])
x0 = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
'''

# Build original P and mask its sparsity
Ap0 = build_Ap(N, A)
size = Ap0.shape[0]
mask_flat = Ap0.flatten() > 0
vars0 = Ap0.flatten()[mask_flat]

# Optimize only the nonzero entries
bounds = [(0.01, None)] * vars0.size
res = minimize(lambda v: objective_vars(v, mask_flat, size), vars0,
               bounds=bounds, method='SLSQP')
vars_opt = res.x
# Reconstruct optimized P
P_opt = reconstruct_P(vars_opt, mask_flat, size)

# Metrics on optimized P
metrics = simulate_P_metrics(P_opt, x0)

print(f"Optimized second eigenvalue magnitude: {metrics['lambda2']:.6f}")
print(f"Converged in {metrics['conv_steps']} steps")
print(f"Consensus value: {metrics['final_val']:.6f}")

# Configure NumPy print options for readability
np.set_printoptions(precision=3, suppress=True)

# Print original augmented matrix Ap
print("Original Augmented Matrix (Ap):")
print(Ap0)

# Print optimized matrix P_opt
print("\nOptimized Matrix (P_opt):")
print(P_opt)


# Plot convergence
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(size):
    ax.plot(metrics['X'][i], label=f'$x_{{{i+1}}}$')
ax.axhline(x0.mean(), ls='--', color='k')
ax.set_title('Optimized P Convergence')
ax.legend(ncol=4, fontsize='small')
ax.grid()
plt.tight_layout()
plt.show()
