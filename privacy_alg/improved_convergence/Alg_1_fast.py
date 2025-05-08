import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import cvxpy as cp

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=3, suppress=True)

# --- Functions ---------------------------------------------------------
def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def algebraic_connectivity(P):
    # Laplacian L = D − P
    D = np.diag(P.sum(axis=1))
    L = D - P
    # compute real eigenvalues, sort
    vals = np.linalg.eigvalsh(L)
    # λ1 == 0, so λ2 is at index 1
    return np.round(vals[1], 6)

# Algorithm 1 augmentation
def build_Ap_alg1(N, A):
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        k1 = N + 3*i
        k2 = k1 + 1
        k3 = k1 + 2
        # simple cycle: i -> k1 -> k2 -> k3 -> i
        Ap[i,   k1] = 1
        Ap[k1, k2] = 1
        Ap[k2, k3] = 1
        Ap[k3, i]  = 1
    return row_normalize(Ap)

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
x0 = np.array([0.1, 0.3, 0.6, 0.43, 0.85, 0.9, 0.45, 0.11, 0.06, 0.51, 0.13])
steps = 500

# Row-stochastic normalization of original
A_norm = row_normalize(A)

# --- 1) Build raw augmented A^P ---------------------------------------
Ap = build_Ap_alg1(N, A_norm)
print("Original A_norm:", np.round(A_norm,4))
print("Raw Augmented Ap:", np.round(Ap,4))

# --- 2) Compute left eigenvector v0 for raw Ap ------------------------
w, V = np.linalg.eig(Ap.T)
idx = np.argmin(np.abs(w - 1))
v0 = np.real(V[:, idx]); v0 /= v0.sum()

# Observability check (optional)
# Build observer matrix to check primary-state observability
C = np.hstack([np.eye(N), np.zeros((N, 3*N))])
P_list = [C @ np.linalg.matrix_power(Ap, k) for k in range(4*N)]
P_O = np.vstack(P_list)
P_sym = sp.Matrix(P_O)
# unit vectors for sample nodes in 4N-dimensional space
e10 = sp.Matrix([1 if i==9 else 0 for i in range(4*N)])
e12 = sp.Matrix([1 if i==11 else 0 for i in range(4*N)])
print("e10 observable?", e10 in P_sym.columnspace())
print("e12 observable?", e12 in P_sym.columnspace())

# --- 3) Inject initial x_p0 via alpha, beta, gamma -------------------- via alpha, beta, gamma --------------------
alpha = np.full(N, 1.4)
beta  = np.full(N, 1.0)
gamma = np.full(N, 1.0)
x_p0 = np.zeros(4*N)
for j in range(N):
    a,b,g = alpha[j], beta[j], gamma[j]
    s = a+b+g
    coeff = 4*x0[j]/s
    x_p0[j] = x0[j]  # primary slot initialized to x0  # primary slot zero
    x_p0[N+3*j+0] = coeff*a
    x_p0[N+3*j+1] = coeff*b
    x_p0[N+3*j+2] = coeff*g
# rescale so v0^T x_p0 = mean(x0)
target = x0.mean(); current = v0 @ x_p0
x_p0 *= (target / current)

print("Raw Ap:\n", Ap)
print("Left eigenvector v0:\n", v0)
print("Initial x_p0:\n", x_p0)

# --- 4) Simulate raw consensus ---------------------------------------
X_orig = np.zeros((N, steps+1)); X_orig[:,0] = x0
X_raw  = np.zeros((4*N, steps+1)); X_raw[:,0]  = x_p0
for k in range(steps):
    X_orig[:,k+1] = A_norm @ X_orig[:,k]
    X_raw[:,k+1]  = Ap     @ X_raw[:,k]

# --- 5) SDP optimize primary weights on A ----------------------------
eps, delta = 1e-6, 1e-6
α_f, β_f    = 0.5, 1.5

W     = cp.Variable((N, N), nonneg=True)
γ     = cp.Variable()
L     = cp.diag(cp.sum(W, axis=1)) - W
P     = np.eye(N) - np.ones((N, N)) / N
mask  = (A_norm > 0).astype(float)

constraints = [
    cp.multiply(mask, W) == W,                # sparsity pattern
    W >= eps * mask,                          # strictly positive on edges
    cp.sum(W, axis=1) == 1,                   # row-stochastic
    L - γ*P >> delta*np.eye(N),               # LMI for λ₂ ≥ γ > 0
    W[mask.astype(bool)] >= α_f * A_norm[mask.astype(bool)],
    W[mask.astype(bool)] <= β_f * A_norm[mask.astype(bool)],
]

prob = cp.Problem(cp.Maximize(γ), constraints)

# --- solve with fallback ----------------------------------------------
try:
    prob.solve(
        solver=cp.CVXOPT,
        feastol=1e-8,
        reltol=1e-8,
        abstol=1e-8,
        verbose=True,
        warm_start=True
    )
    print("Solved with CVXOPT.")
except Exception as e:
    print("CVXOPT failed, falling back to SCS:", e)
    prob.solve(
        solver=cp.SCS,
        eps=1e-5,
        max_iters=200000,
        verbose=True
    )

print("Status:", prob.status)
print("γ* =", γ.value)
W_opt = W.value
print("W_opt =\n", np.round(W_opt, 4))

# --- 6) Build optimized Ap and re-initialize xP0 ---------------------
Ap_opt = build_Ap_alg1(N, W_opt)
print("Optimized Augmented Ap_opt:", np.round(Ap_opt,4))

# recompute left-eigenvector for optimized chain
w_opt, V_opt = np.linalg.eig(Ap_opt.T)
idx_opt     = np.argmin(np.abs(w_opt - 1))
v0_opt      = np.real(V_opt[:, idx_opt]); v0_opt /= v0_opt.sum()

# rebuild xP0 for optimized chain using alpha,beta,gamma and rescale
xP0_opt = np.zeros(4*N)
for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    s = a + b + g
    coeff = 4 * x0[j] / s
    xP0_opt[j]       = 0
    xP0_opt[N+3*j+0] = coeff * a
    xP0_opt[N+3*j+1] = coeff * b
    xP0_opt[N+3*j+2] = coeff * g
# rescale so v0_opt^T xP0_opt = mean(x0)
xP0_opt *= (target / (v0_opt @ xP0_opt))
xP0_opt = np.zeros(4*N)
for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    s = a + b + g
    coeff = 4 * x0[j] / s
    xP0_opt[j]       = 0
    xP0_opt[N+3*j+0] = coeff * a
    xP0_opt[N+3*j+1] = coeff * b
    xP0_opt[N+3*j+2] = coeff * g
# rescale so v0_opt^T xP0_opt = mean(x0)
xP0_opt *= (target / (v0_opt @ xP0_opt))

# simulate optimized consensus
X_opt = np.zeros((4*N, steps+1)); X_opt[:,0] = xP0_opt
for k in range(steps):
    X_opt[:,k+1] = Ap_opt @ X_opt[:,k]

# original connectivity
λ2_A = algebraic_connectivity(A_norm)
# optimized connectivity
λ2_W = algebraic_connectivity(W_opt)

print(f"λ₂(L_A) = {λ2_A}")
print(f"λ₂(L_W) = {λ2_W}")

# --- 7) Plot comparisons -------------------------------------------- --------------------------------------------
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,12))
# original
for i in range(N): ax1.plot(X_orig[i],'--',alpha=0.6)
ax1.axhline(target,ls=':',color='k'); ax1.set_title('Orig consensus')
# raw
for i in range(N): ax2.plot(X_raw[i],alpha=0.6)
ax2.axhline(target,ls=':',color='k'); ax2.set_title('Raw Ap consensus')
# optimized
for i in range(N): ax3.plot(X_opt[i],alpha=0.6)
ax3.axhline(target,ls=':',color='k'); ax3.set_title('Optimized Ap consensus')
plt.tight_layout(); plt.show()

# --- 8) Convergence steps and final values ---------------------------
# compute consensus values (mean over primaries at last step)
cons_o = X_orig[:N, -1].mean()
cons_r = X_raw[:N, -1].mean()
cons_p = X_opt[:N, -1].mean()

# convergence detection: ignore k=0
tol = 1e-4
conv = lambda X: next((i for i in range(1, X.shape[1]) if X[:N,i].max()-X[:N,i].min()<tol), None)
step_o = conv(X_orig)
step_r = conv(X_raw)
step_p = conv(X_opt)

# print results
print(f"Original converged in {step_o} steps to {cons_o:.6f}")
print(f"Raw Augmented converged in {step_r} steps to {cons_r:.6f}")
print(f"Optimized Augmented converged in {step_p} steps to {cons_p:.6f}")
