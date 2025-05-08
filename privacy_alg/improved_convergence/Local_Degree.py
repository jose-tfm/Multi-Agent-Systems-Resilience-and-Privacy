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
        Ap[i,   i1] = 1/12;  Ap[i,   i2] = 1/8
        Ap[i,   i3] = 1/4;   Ap[i,   i4] = 1/24
        Ap[i1, i] = 1/11;    Ap[i2, i] = 1/2
        Ap[i3, i] = 3/4;     Ap[i4, i] = 1/16
        Ap[i2, i1] = 1/2;    Ap[i3, i1] = 1/4;    Ap[i4, i1] = 15/16
        Ap[i1, i2] = 3/22;   Ap[i1, i3] = 1/11;   Ap[i1, i4] = 15/22
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
    xP0[:N] = x0
    xP0 *= (x0.mean() / (vL @ xP0))
    return s, vL, xP0

# --- parameters --------------------------------------------------------

N     = 11
A = np.array([
 [0.0000, 0.5608, 0.2794, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1598],
 [0.5527, 0.0000, 0.1783, 0.0580, 0.2110, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
 [0.2665, 0.1726, 0.0000, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
 [0.0000, 0.0485, 0.4842, 0.0000, 0.1730, 0.0000, 0.0000, 0.2942, 0.0000, 0.0000, 0.0000],
 [0.0000, 0.3306, 0.0000, 0.3243, 0.0000, 0.3452, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
 [0.0000, 0.0000, 0.0000, 0.0000, 0.2715, 0.0000, 0.2711, 0.0000, 0.2855, 0.1718, 0.0000],
 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2792, 0.0000, 0.4726, 0.0000, 0.2482, 0.0000],
 [0.0000, 0.0000, 0.0000, 0.3380, 0.0000, 0.0000, 0.3576, 0.0000, 0.3044, 0.0000, 0.0000],
 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4223, 0.0000, 0.5777, 0.0000, 0.0000, 0.0000],
 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1343, 0.1885, 0.0000, 0.0000, 0.3460, 0.3312],
 [0.3142, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6858, 0.0000]
])

x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
steps = 500

# 1) Consensus on original A
A_norm = row_normalize(A)
Xo     = np.zeros((N, steps+1))
Xo[:,0] = x0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]

# 2) Augmented on original A
Ap_orig      = build_Ap_bidir_raw(N, A)
_, _, xP0_o  = distributed_solve_alg4(N, Ap_orig, x0)
Xa_orig      = np.zeros((5*N, steps+1))
Xa_orig[:,0] = xP0_o
for k in range(steps):
    Xa_orig[:,k+1] = Ap_orig @ Xa_orig[:,k]

# 3) Uniform weights (row-normalized binary adjacency)
Adj      = (A > 0).astype(float)
W_unif   = row_normalize(Adj)
print("W_unif =\n", np.round(W_unif,4))

# 4) Augmented on uniform W_unif
Ap_unif      = build_Ap_bidir_raw(N, W_unif)
_, _, xP0_u  = distributed_solve_alg4(N, Ap_unif, x0)
Xa_unif      = np.zeros((5*N, steps+1))
Xa_unif[:,0] = xP0_u
for k in range(steps):
    Xa_unif[:,k+1] = Ap_unif @ Xa_unif[:,k]

# 5) Plot
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].set_title('(a) Consensus on original A')
for i in range(N):
    axes[0].plot(Xo[i], '--', alpha=0.6)
axes[0].grid(':')

axes[1].set_title('(b) Augmented on original A')
for i in range(5*N):
    axes[1].plot(Xa_orig[i], alpha=0.6)
axes[1].grid(':')

axes[2].set_title('(c) Augmented on uniform W_unif')
for i in range(5*N):
    axes[2].plot(Xa_unif[i], alpha=0.6)
axes[2].grid(':')

plt.tight_layout()
plt.show()

# 6) Convergence info
tol = 1e-5
def conv_steps(X):
    diam = np.max(X[:N,:], axis=0) - np.min(X[:N,:], axis=0)
    idx  = np.where(diam < tol)[0]
    return idx[0] if idx.size else None

print(f"(a) → {Xo[:,-1].mean():.6f} in {conv_steps(Xo)} steps")
print(f"(b) → {Xa_orig[:N,-1].mean():.6f} in {conv_steps(Xa_orig)} steps")
print(f"(c) → {Xa_unif[:N,-1].mean():.6f} in {conv_steps(Xa_unif)} steps")
