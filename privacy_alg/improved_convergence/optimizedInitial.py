import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize

# ------------------------------------------------------------
# Row-stochastic normalization
def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

# Build augmented P with fixed gadget weights
def build_Ap(N: int, A: np.ndarray) -> np.ndarray:
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
        ''''
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

# Reconstruct A only where mask=True
def build_A_from_p(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N = mask.shape[0]
    A = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                A[i, j] = p[idx]
                idx += 1
    return row_normalize(A)

# Objective: second largest eigenvalue magnitude of P
def consensus_rate_p(p: np.ndarray) -> float:
    A_var = build_A_from_p(p, mask)
    vals = np.abs(eigvals(build_Ap(N, A_var)))
    vals.sort()
    return vals[-2]

# Simulate trajectories and compute convergence steps (fixed to avoid zeros after convergence)
def simulate_and_metrics(A_mat: np.ndarray, x0: np.ndarray, N: int,
                         tol=1e-4, max_steps=60):
    P_mat = build_Ap(N, A_mat)
    # second largest eigenvalue
    eigs = np.abs(eigvals(P_mat)); eigs.sort(); lambda2 = eigs[-2]
    # stationary left eigenvector
    wv, V = eig(P_mat.T)
    idx = np.argmin(np.abs(wv - 1)); v0 = np.real(V[:, idx]); v0 /= v0.sum()
    # initial augmented state
    target = x0.mean()
    x_p0 = np.zeros(4*N)
    for j in range(N):
        x_p0[N+3*j:N+3*j+3] = 4 * x0[j] / 3
    x_p0 *= target / (v0 @ x_p0)
    # simulate
    Xo = np.zeros((N, max_steps+1))
    Xa = np.zeros((4*N, max_steps+1))
    Xo[:,0], Xa[:,0] = x0, x_p0
    orig_steps = aug_steps = None
    for k in range(max_steps):
        Xo[:,k+1] = A_mat @ Xo[:,k]
        Xa[:,k+1] = P_mat @ Xa[:,k]
        if orig_steps is None and np.max(np.abs(Xo[:,k+1] - target)) < tol:
            orig_steps = k+1
        if aug_steps is None and np.max(np.abs(Xa[:,k+1] - target)) < tol:
            aug_steps = k+1
    # fill remainder with final consensus value
    if orig_steps is not None:
        Xo[:, orig_steps+1:] = Xo[:, orig_steps][:,None]
    if aug_steps is not None:
        Xa[:, aug_steps+1:] = Xa[:, aug_steps][:,None]
    return {
        'lambda2': lambda2,
        'orig_steps': orig_steps,
        'aug_steps': aug_steps,
        'Xo': Xo,
        'Xa': Xa
    }

# --- Main setup ---

N = 3
raw_A = np.array([
    [0,   0.3, 0.7],
    [0.3, 0,   0.7],
    [0.3, 0.7, 0  ]
], dtype=float)
x0 = np.array([0.5, 1/3, 0.2])

'''
N = 11
raw_A = np.array([
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
x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
'''
mask = raw_A > 0
p0 = np.array([raw_A[i,j] for i in range(N) for j in range(N) if mask[i,j]])
bounds = [(0.1, 10)] * p0.size

# normalize and metrics for initial A
A_initial = build_A_from_p(p0, mask)
metrics_init = simulate_and_metrics(A_initial, x0, N)

# optimize
res = minimize(consensus_rate_p, p0, method='SLSQP', bounds=bounds)
A_opt = build_A_from_p(res.x, mask)
metrics_opt = simulate_and_metrics(A_opt, x0, N)

# --- Print results ---
print("A_original:")
print(np.round(A_initial,3))
print("Ap_original:")
print(np.round(build_Ap(N, A_initial),3))
print(f"lambda2(P_original) = {metrics_init['lambda2']:.4f}")
print(f"Convergence A_original in {metrics_init['orig_steps']} steps")
print(f"Convergence Ap_original in {metrics_init['aug_steps']} steps")

print("\nA_optimized:")
print(np.round(A_opt,3))
print("Ap_optimized:")
print(np.round(build_Ap(N, A_opt),3))
print(f"lambda2(P_optimized) = {metrics_opt['lambda2']:.4f}")
print(f"Convergence A_optimized in {metrics_opt['orig_steps']} steps")
print(f"Convergence Ap_optimized in {metrics_opt['aug_steps']} steps")

# --- Plots 2x2 ---
fig, axes = plt.subplots(2,2, figsize=(12,10))
labels_o = [f'$x_{{{i+1}}}$' for i in range(N)]
labels_a = [f'$\\tilde x_{{{(i-N)//3+1},{(i-N)%3+1}}}$' for i in range(N,4*N)]

# Original A convergence
for i in range(N):
    axes[0,0].plot(metrics_init['Xo'][i], label=labels_o[i])
axes[0,0].set_title('Original A Convergence')
axes[0,0].axhline(x0.mean(), ls='--', color='k')

# Original Ap convergence
for i in range(4*N):
    axes[0,1].plot(metrics_init['Xa'][i], label=(labels_o+labels_a)[i])
axes[0,1].set_title('Original Ap Convergence')
axes[0,1].axhline(x0.mean(), ls='--', color='k')

# Optimized A convergence
for i in range(N):
    axes[1,0].plot(metrics_opt['Xo'][i], label=labels_o[i])
axes[1,0].set_title('Optimized A Convergence')
axes[1,0].axhline(x0.mean(), ls='--', color='k')

# Optimized Ap convergence
for i in range(4*N):
    axes[1,1].plot(metrics_opt['Xa'][i], label=(labels_o+labels_a)[i])
axes[1,1].set_title('Optimized Ap Convergence')
axes[1,1].axhline(x0.mean(), ls='--', color='k')

for ax in axes.flatten():
    ax.grid()
axes[0,0].legend(ncol=3, fontsize='small')

plt.tight_layout()
plt.show()
