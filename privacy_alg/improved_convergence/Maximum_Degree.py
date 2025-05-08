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
        Ap[i1, i] = 1/11
        Ap[i2, i] =   1/2
        Ap[i3, i] =   3/4
        Ap[i4, i] = 1/16
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
        s[base+1] =  1/4 * si
        s[base+2] =  1/3 * si
        s[base+3] =  2/3 * si
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
    [0,1/3,1/3,0,0,0,0,0,0,0,1/3],
    [1/3,0,1/3,1/3,0,0,0,0,0,0,0],
    [1/3,1/3,0,1/3,0,0,0,0,0,0,0],
    [0,0,1/3,0,1/3,0,0,1/3,0,0,0],
    [0,1/3,0,1/3,0,1/3,0,0,0,0,0],
    [0,0,0,0,1/3,0,1/3,0,0,1/3,0],
    [0,0,0,0,0,1/2,0,1/2,0,0,0],
    [0,0,0,1/3,0,0,1/3,0,1/3,0,0],
    [0,0,0,0,0,1/2,0,1/2,0,0,0],
    [0,0,0,0,0,0,1/3,0,0,1/3,1/3],
    [1/2,0,0,0,0,0,0,0,0,1/2,0]
])
x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
steps = 500

# 1) Consensus on original A
A_norm = row_normalize(A)
Xo     = np.zeros((N,steps+1)); Xo[:,0] = x0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]

# 2) Augmented on original A
Ap_orig    = build_Ap_bidir_raw(N, A)
_,_,xP0_o   = distributed_solve_alg4(N, Ap_orig, x0)
Xa_orig    = np.zeros((5*N,steps+1)); Xa_orig[:,0] = xP0_o
for k in range(steps):
    Xa_orig[:,k+1] = Ap_orig @ Xa_orig[:,k]

# 3) Maximum-degree weights on A
Adj = (A>0).astype(float)
deg = Adj.sum(axis=1)
Delta = int(deg.max())
alpha = 1.0 / Delta

W_max = np.zeros_like(A)
for i in range(N):
    for j in range(N):
        if Adj[i,j]:
            W_max[i,j] = alpha
# fix diagonal
off = W_max.copy(); np.fill_diagonal(off,0.0)
np.fill_diagonal(W_max,1.0 - off.sum(axis=1))

# 4) Augmented on W_max
Ap_opt    = build_Ap_bidir_raw(N, W_max)
_,_,xP0_t = distributed_solve_alg4(N, Ap_opt, x0)
Xa_opt    = np.zeros((5*N,steps+1)); Xa_opt[:,0] = xP0_t
for k in range(steps):
    Xa_opt[:,k+1] = Ap_opt @ Xa_opt[:,k]

# 5) Plot
fig,axes = plt.subplots(3,1,figsize=(8,12))
axes[0].set_title('(a) Consensus on original A')
for i in range(N):
    axes[0].plot(Xo[i],'--',alpha=0.6)
axes[0].grid(':')

axes[1].set_title('(b) Augmented on original A')
for i in range(5*N):
    axes[1].plot(Xa_orig[i],alpha=0.6)
axes[1].grid(':')

axes[2].set_title('(c) Augmented on max-degree A_max')
for i in range(5*N):
    axes[2].plot(Xa_opt[i],alpha=0.6)
axes[2].grid(':')

plt.tight_layout()
plt.show()

# 6) Convergence info
tol = 1e-3
def conv_steps(X):
    diam = np.max(X[:N,:],axis=0) - np.min(X[:N,:],axis=0)
    idx  = np.where(diam<tol)[0]
    return idx[0] if idx.size else None

sa = conv_steps(Xo)
sb = conv_steps(Xa_orig)
sc = conv_steps(Xa_opt)

print(f"(a) → final ≈ {Xo[:,-1].mean():.6f} in {sa} steps")
print(f"(b) → final ≈ {Xa_orig[:N,-1].mean():.6f} in {sb} steps")
print(f"(c) → final ≈ {Xa_opt[:N,-1].mean():.6f} in {sc} steps")
