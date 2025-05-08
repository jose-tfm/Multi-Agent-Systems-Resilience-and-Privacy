import numpy as np
import matplotlib.pyplot as plt

# --- utilities ----------------------------------------------------------

def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap_bidir_raw(N, A):
    """Algorithm 3 exactly — no normalization at the end."""
    size = 5 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A / 2.0
    for i in range(N):
        base = N + 4*i
        i1, i2, i3, i4 = base, base+1, base+2, base+3
        Ap[i,   i1] = 1/12
        Ap[i,   i2] =  1/8
        Ap[i,   i3] =  1/4
        Ap[i,   i4] = 1/24
        Ap[i1, i]   = 1/11
        Ap[i2, i]   =   1/2
        Ap[i3, i]   =   3/4
        Ap[i4, i]   = 1/16
        Ap[i2, i1] =  1/2
        Ap[i3, i1] =  1/4
        Ap[i4, i1] = 15/16
        Ap[i1, i2] =  3/22
        Ap[i1, i3] =   1/11
        Ap[i1, i4] = 15/22
    return Ap

def distributed_solve_alg4(N, Ap, x0, ratios=None):
    """Compute s, left eigenvector vL, and initial xP0."""
    size = 5 * N
    s = np.zeros(size); s[0] = 1.0
    visited, rem = {0}, set(range(1, N))
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i,j] > 0 and Ap[j,i] > 0:
                    s[j] = s[i] * Ap[i,j] / Ap[j,i]
                    visited.add(j)
                    rem.remove(j)
                    break
            else:
                continue
            break
    for i in range(N):
        si, base = s[i], N + 4*i
        s[base+0] = 11/12 * si
        s[base+1] =  1/4   * si
        s[base+2] =  1/3   * si
        s[base+3] =  2/3   * si
    Z = s.sum(); vL = s / Z

    xP0 = np.zeros(size)
    if ratios is None:
        ratios = np.ones(4)
    for i in range(N):
        xi, base = x0[i], N + 4*i
        alphas = ratios / ratios.sum() * (5 * xi)
        for k in range(4):
            xP0[base+k] = (Z / s[base+k]) * alphas[k]
    # force primaries back to x0
    xP0[:N] = x0
    # scale so ⟨vL, xP0⟩ = x0.mean()
    xP0 *= (x0.mean() / (vL @ xP0))
    return s, vL, xP0

# --- parameters --------------------------------------------------------

N     = 11
A     = np.array([
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
x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
steps = 500

# --- 1) Consensus on G(A) --------------------------------------------------
A_norm = row_normalize(A)
Xo = np.zeros((N,steps+1)); Xo[:,0] = x0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]

# --- 2) Augment original A ------------------------------------------------
Ap0, _, xP0_0 = distributed_solve_alg4(N, build_Ap_bidir_raw(N,A), x0)
Xa0 = np.zeros((5*N,steps+1)); Xa0[:,0] = xP0_0
for k in range(steps):
    Xa0[:,k+1] = build_Ap_bidir_raw(N,A) @ Xa0[:,k]

# --- 3) Metropolis weights on A ------------------------------------------
Adj = (A>0).astype(float)
deg = Adj.sum(axis=1)

W_met = np.zeros_like(A)
for i in range(N):
    for j in range(N):
        if Adj[i,j]:
            W_met[i,j] = 1.0 / (1 + max(deg[i], deg[j]))
# zero any stray off-diagonals, then fix diagonal
W_met *= Adj
off = W_met.copy(); np.fill_diagonal(off,0)
np.fill_diagonal(W_met, 1 - off.sum(axis=1))

# print matrices for inspection
np.set_printoptions(precision=5, suppress=True, linewidth=120)
print("Original A:\n", A)
print("\nMetropolis W_met (off-diagonals only where A>0):\n", W_met * (1-np.eye(N)))
print("\nW_met diagonal (self-weights):\n", np.diag(W_met))
print("\nRow sums of W_met:", W_met.sum(axis=1))

# --- 4) Augment Metropolis-weighted A -------------------------------------
ApM, _, xP0_M = distributed_solve_alg4(N, build_Ap_bidir_raw(N,W_met), x0)
XaM = np.zeros((5*N,steps+1)); XaM[:,0] = xP0_M
for k in range(steps):
    XaM[:,k+1] = build_Ap_bidir_raw(N,W_met) @ XaM[:,k]

# --- 5) Plot all three ----------------------------------------------------
fig, ax = plt.subplots(3,1,figsize=(8,12))

ax[0].set_title('(a) Consensus on $G(A)$')
for i in range(N):
    ax[0].plot(Xo[i],'--',alpha=0.6)
ax[0].grid(':')

ax[1].set_title('(b) Augmented on $G(A)$')
for i in range(5*N):
    ax[1].plot(Xa0[i],alpha=0.6)
ax[1].grid(':')

ax[2].set_title('(c) Augmented on Metropolis–weighted $A_{met}$')
for i in range(5*N):
    ax[2].plot(XaM[i],alpha=0.6)
ax[2].grid(':')

plt.tight_layout()
plt.show()

# --- 6) Convergence info --------------------------------------------------
tol = 1e-3
def conv_steps(X):
    d = np.max(X[:N,:],axis=0) - np.min(X[:N,:],axis=0)
    idx = np.where(d<tol)[0]
    return idx[0] if idx.size else None

print("(a) →",  np.round(Xo[:,-1].mean(),6), "in", conv_steps(Xo), "steps")
print("(b) →",  np.round(Xa0[:N,-1].mean(),6), "in", conv_steps(Xa0), "steps")
print("(c) →",  np.round(XaM[:N,-1].mean(),6), "in", conv_steps(XaM), "steps")
