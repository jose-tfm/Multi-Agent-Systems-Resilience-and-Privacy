import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# --- Helper: Row normalize rows of a matrix ---------------------------
def row_normalize(M):
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

# Build the fixed cycle adjacency for Algorithm 1 augmentation --------
def build_cycle_block(N):
    size = 4 * N
    C = np.zeros((size, size))
    for i in range(N):
        k1, k2, k3 = N+3*i, N+3*i+1, N+3*i+2
        C[i,   k1] = 1
        C[k1, k2] = 1
        C[k2, k3] = 1
        C[k3, i]  = 1
    return C

# --- Parameters --------------------------------------------------------
N = 11
A = np.array([
    [0,   1/3, 1/3, 0,   0,   0,   0,   0,   0,   0,   1/3],
    [1/3, 0,   1/3, 1/3, 0,   0,   0,   0,   0,   0,   0  ],
    [1/3, 1/3, 0,   1/3, 0,   0,   0,   0,   0,   0,   0  ],
    [0,   0,   1/3, 0,   1/3, 0,   0,   1/3, 0,   0,   0  ],
    [0,   1/3, 0,   1/3, 0,   1/3, 0,   0,   0,   0,   0  ],
    [0,   0,   0,   0,   1/3, 0,   1/3, 0,   0,   1/3, 0  ],
    [0,   0,   0,   0,   0,   1/2, 0,   1/2, 0,   0,   0  ],
    [0,   0,   0,   1/3, 0,   0,   1/3, 0,   1/3, 0,   0  ],
    [0,   0,   0,   0,   0,   1/2, 0,   1/2, 0,   0,   0  ],
    [0,   0,   0,   0,   0,   0,   1/3, 0,   0,   1/3, 1/3],
    [1/2, 0,   0,   0,   0,   0,   0,   0,   0,   1/2, 0  ]
])
x0 = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
steps = 500

# Normalize original A and build cycle block
A_norm = row_normalize(A)
C = build_cycle_block(N)

# --- 1) Build raw augmented Ap ----------------------------------------
Ap_raw = build_cycle_block(N) + np.block([
    [A_norm,            np.zeros((N,3*N))],
    [np.zeros((3*N,N)), np.zeros((3*N,3*N))]
])
Ap_raw = row_normalize(Ap_raw)

# --- 2) Introduce per-node row scalings s -----------------------------
s = cp.Variable(N, nonneg=True)
# build M rows via linear expressions and enforce row-sums
M_rows = []
constraints = []
for i in range(N):
    scaled = s[i] * Ap_raw[i,:]             # linear in s
    M_rows.append(scaled)
    # enforce that this row sums to 1
    constraints.append(cp.sum(scaled) == 1)
for j in range(N,4*N):
    M_rows.append(Ap_raw[j,:])

M = cp.vstack(M_rows)

# --- 3) SDP: maximize algebraic connectivity of full chain ------------
# optimization variable for spectral gap
gamma = cp.Variable()
# full-chain Laplacian
i4 = np.eye(4*N)
one4 = np.ones((4*N,4*N)) / (4*N)
L = cp.diag(cp.sum(M,axis=1)) - M
# add spectral-gap constraint
constraints += [L - gamma * (i4 - one4) >> 0]
# solve
prob = cp.Problem(cp.Maximize(gamma), constraints)
prob.solve(solver=cp.SCS, eps=1e-6, max_iters=20000, verbose=True)
print("SDP status:", prob.status, "γ_opt=", gamma.value)

# recover optimized augmented matrix
M_opt = M.value
Ap_opt = row_normalize(M_opt)
t = M.value
Ap_opt = row_normalize(t)

# --- Print raw vs. optimized matrices --------------------------------
print("Raw augmented Ap (first 11x11 block and cycles):")
print(np.round(Ap_raw,4))
print("Optimized augmented Ap_opt:")
print(np.round(Ap_opt,4))

# --- 4) Initialize xP0 via left-eigenvector -------------------------- --------------------------
def init_xP0(Ap_chain):
    w, V = np.linalg.eig(Ap_chain.T)
    idx   = np.argmin(np.abs(w - 1))
    v0    = np.real(V[:,idx]); v0 /= v0.sum()
    xP0   = np.zeros(4*N)
    alpha, beta, gamma = np.full(N,1.4), np.full(N,1.0), np.full(N,1.0)
    for j in range(N):
        s = alpha[j] + beta[j] + gamma[j]
        coeff = 4 * x0[j] / s
        xP0[j] = 0
        xP0[N+3*j:N+3*j+3] = coeff * np.array([alpha[j], beta[j], gamma[j]])
    xP0 *= (x0.mean() / (v0 @ xP0))
    return xP0

xP0_raw = init_xP0(Ap_raw)
xP0_opt = init_xP0(Ap_opt)

# --- 5) Simulate raw vs. optimized chains -----------------------------
X_raw = np.zeros((4*N,steps+1)); X_raw[:,0] = xP0_raw
X_opt = np.zeros((4*N,steps+1)); X_opt[:,0] = xP0_opt
for k in range(steps):
    X_raw[:,k+1] = Ap_raw @ X_raw[:,k]
    X_opt[:,k+1] = Ap_opt @ X_opt[:,k]

# --- 6) Plot only primary states -------------------------------------
fig, axes = plt.subplots(2,1,figsize=(8,10))
for i in range(N): axes[0].plot(X_raw[i], alpha=0.6)
axes[0].set_title('Raw Augmented Chain')
for i in range(N): axes[1].plot(X_opt[i], alpha=0.6)
axes[1].set_title('Optimized Augmented Chain')
plt.tight_layout(); plt.show()

# --- 7) Print convergence steps --------------------------------------
def conv_steps(X):
    tol = 1e-4
    for k in range(1, X.shape[1]):
        if X[:N,k].max() - X[:N,k].min() < tol:
            return k
    return None
print('raw steps =', conv_steps(X_raw))
print('opt steps =', conv_steps(X_opt))
