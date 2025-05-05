import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# ——— force full‐array printing ———————————————————————————————
np.set_printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True)

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
        Ap[i1, i]   = 1/11;  Ap[i2, i]   = 1/2
        Ap[i3, i]   = 3/4;   Ap[i4, i]   = 1/16
        Ap[i2, i1]  = 1/2;   Ap[i3, i1]  = 1/4
        Ap[i4, i1]  = 15/16; Ap[i1, i2]  = 3/22
        Ap[i1, i3]  = 1/11;  Ap[i1, i4]  = 15/22
    return Ap

def distributed_solve_alg4(N, Ap, x0, ratios=None):
    """
    Compute s, left eigenvector vL, and an *unscaled* augmented xP0
    for a row-stochastic Ap.
    """
    size = 5 * N
    # 1) solve for s on primaries
    s = np.zeros(size); s[0] = 1.0
    visited, rem = {0}, set(range(1, N))
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i,j] > 0 and Ap[j,i] > 0:
                    s[j] = s[i] * Ap[i,j] / Ap[j,i]
                    visited.add(j); rem.remove(j)
                    break
            else:
                continue
            break

    # 2) extend to auxiliaries
    for i in range(N):
        si, b = s[i], N + 4*i
        s[b+0] = 11/12*si; s[b+1] = 1/4*si
        s[b+2] = 1/3*si;  s[b+3] = 2/3*si

    Z   = s.sum()
    vL  = s / Z           # left Perron vector of Ap

    # 3) build unscaled xP0
    xP0 = np.zeros(size)
    if ratios is None:
        ratios = np.ones(4)
    for i in range(N):
        xi, b = x0[i], N + 4*i
        alpha = ratios/ratios.sum() * (5*xi)
        for k in range(4):
            xP0[b+k] = (Z / s[b+k]) * alpha[k]

    # force primaries exactly to x0
    xP0[:N] = x0

    return s, vL, xP0

# --- main workflow ------------------------------------------------------

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
x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
steps = 1000

# --- 1) Original consensus on G(A) ------------------------------------
A_norm = row_normalize(A)
Xo = np.zeros((N, steps+1)); Xo[:,0] = x0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]

# --- 2) Vanilla consensus on G(A^P) -----------------------------------
Ap_raw = build_Ap_bidir_raw(N, A)
Ap     = row_normalize(Ap_raw)

print("===== Vanilla augmented matrix Ap (row-normalized) =====")
print(Ap)

s, vL, xP0_unscaled = distributed_solve_alg4(N, Ap, x0)

# scale for vanilla: ensure vLᵀ xP0 = x0.mean()
xP0_v = xP0_unscaled.copy()
xP0_v[:N] = x0
xP0_v *= (x0.mean() / (vL @ xP0_v))

Xa_v = np.zeros((5*N, steps+1)); Xa_v[:,0] = xP0_v
for k in range(steps):
    Xa_v[:,k+1] = Ap @ Xa_v[:,k]

# --- 3) SDP‐optimize G(A^P) with double‐stochastic Wp ------------------
maskP  = (Ap_raw > 0).astype(float) + np.eye(5*N)
Wp     = cp.Variable((5*N,5*N), nonneg=True)
gammaP = cp.Variable()
Lp     = cp.diag(cp.sum(Wp,axis=1)) - Wp
Pp     = np.eye(5*N) - np.ones((5*N,5*N))/(5*N)

constraints = [
    cp.multiply(maskP, Wp) == Wp,   # respect topology + self‐loops
    cp.sum(Wp, axis=1) == 1,        # row‐stochastic
    cp.sum(Wp, axis=0) == 1,        # column‐stochastic
    Lp - gammaP * Pp >> 0           # maximize connectivity
]
probP = cp.Problem(cp.Maximize(gammaP), constraints)
probP.solve(solver=cp.SCS, verbose=True)
Wp_opt = Wp.value

print("\n===== SDP-optimized augmented weight matrix Wp_opt =====")
print(Wp_opt)
print(f"\nAugmented SDP opt γ* = {gammaP.value:.4f}")

# --- 4) Consensus on optimized ----------------------------------------
# scale for optimized: ensure arithmetic mean of xP0 = x0.mean()
xP0_o = xP0_unscaled.copy()
xP0_o[:N] = x0
xP0_o *= ((5*N) * x0.mean()) / xP0_o.sum()

Xa_o = np.zeros((5*N, steps+1)); Xa_o[:,0] = xP0_o
for k in range(steps):
    Xa_o[:,k+1] = Wp_opt @ Xa_o[:,k]

# --- 5) Plot comparisons ----------------------------------------------
fig, axes = plt.subplots(3,1,figsize=(8,12))
axes[0].set_title('(a) $G(A)$')
axes[1].set_title('(b) vanilla $G(A^P)$')
axes[2].set_title('(c) SDP‐optimized $G(A^P)$')

for i in range(N):
    axes[0].plot(Xo[i], '--', alpha=0.6)
for i in range(5*N):
    axes[1].plot(Xa_v[i], alpha=0.6)
    axes[2].plot(Xa_o[i], linewidth=1.5)

for ax in axes:
    ax.axhline(x0.mean(), ls=':', color='black')
    ax.grid(linestyle=':')

plt.tight_layout()
plt.show()

# --- 6) Convergence times on primaries only ----------------------------
tol = 1e-5

def conv_steps_primary(X, target):
    sup = np.max(np.abs(X[:N, :] - target), axis=0)
    idx = np.where(sup < tol)[0]
    return idx[0] if idx.size else None

for label, X in [
    ('(a)', Xo),
    ('(b)', Xa_v),
    ('(c)', Xa_o),
]:
    step = conv_steps_primary(X, x0.mean())
    if step is None:
        print(f"{label} (primaries) did not converge within {steps} steps.")
    else:
        print(f"{label} (primaries) converged in {step} steps to {x0.mean():.6f}")
