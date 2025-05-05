import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cvxpy as cp
# Este foi o Ultimo que estive mudar
# ——— force full‐array printing ———————————————————————————————
np.set_printoptions(threshold=np.inf, linewidth=200, precision=7, suppress=True)


def build_Ap_bidir_raw(N, A):
    """Algorithm 3 exactly — no normalization at the end."""
    size = 5 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A / 2.0
    for i in range(N):
        base = N + 4 * i
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
    s = np.zeros(size)
    s[0] = 1.0
    visited, rem = {0}, set(range(1, N))
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i, j] > 0 and Ap[j, i] > 0:
                    s[j] = s[i] * Ap[i, j] / Ap[j, i]
                    visited.add(j)
                    rem.remove(j)
                    break
            else:
                continue
            break
    for i in range(N):
        si, base = s[i], N + 4 * i
        s[base+0] = 11/12 * si
        s[base+1] = 1/4   * si
        s[base+2] = 1/3   * si
        s[base+3] = 2/3   * si
    Z = s.sum()
    vL = s / Z
    xP0 = np.zeros(size)
    if ratios is None:
        ratios = np.ones(4)
    for i in range(N):
        xi, base = x0[i], N + 4 * i
        alphas = ratios / ratios.sum() * (5 * xi)
        for k in range(4):
            xP0[base + k] = (Z / s[base + k]) * alphas[k]
    xP0 *= (x0.mean() / (vL @ xP0))
    return s, vL, xP0

# --- parameters --------------------------------------------------------

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

# --- 1) Consensus on original and augmented networks ------------------


print('\nFull original A (size {}×{}):'.format(*A.shape))
print(A)
Xo = np.zeros((N, steps+1))
Xo[:, 0] = x0
for k in range(steps):
    Xo[:, k+1] = A @ Xo[:, k]

# build Ap (raw) and also print its normalized version
Ap = build_Ap_bidir_raw(N, A)
print('\nFull original Ap (size {}×{}):'.format(*Ap.shape))
print(Ap)


_, _, xP0 = distributed_solve_alg4(N, Ap, x0)
Xa = np.zeros((5*N, steps+1))
Xa[:, 0] = xP0
for k in range(steps):
    Xa[:, k+1] = Ap @ Xa[:, k]

# --- 2) SDP-based weight optimization --------------------------------
mask = (A > 0).astype(float)
eps  = 1e-6

W = cp.Variable((N,N), nonneg=True)
γ = cp.Variable()
L = cp.diag(cp.sum(W,axis=1)) - W
P = np.eye(N) - np.ones((N,N))/N

constraints = [
    cp.multiply(mask, W) == W,   # zero out non-edges
    W >= eps*mask,               # enforce W[i,j] >= eps on every original edge
    cp.sum(W, axis=1) == 1,      # row‐stochastic
    L - γ*P >> 0                 # maximize algebraic connectivity
]

prob = cp.Problem(cp.Maximize(γ), constraints)
prob.solve(solver=cp.SCS)

W_opt = W.value
print("Optimized A:\n", W_opt)



# build Ap from W_opt and print its normalized version
Ap_opt = build_Ap_bidir_raw(N, W_opt)
print('\nAp_opt from A_opt (size {}×{}):'.format(*Ap_opt.shape))
print(Ap_opt)

# --- 3) Consensus with optimized weights ------------------------------
X_sdp = np.zeros((N, steps+1))
X_sdp[:, 0] = x0
for k in range(steps):
    X_sdp[:, k+1] = W_opt @ X_sdp[:, k]

# --- 4) Plot comparisons ----------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# (a) Original
for i in range(N):
    axes[0].plot(Xo[i], linestyle='--', alpha=0.6)
axes[0].axhline(x0.mean(), ls=':', color='black', label=f"True avg={x0.mean():.2f}")
axes[0].set_title('(a) Consensus on $G(A)$')
axes[0].set_ylabel('Value')
axes[0].grid(linestyle=':')

# (b) Augmented
for i in range(N*5):
    axes[1].plot(Xa[i], alpha=0.6)
axes[1].axhline(x0.mean(), ls='--', color='black', label=f"True avg={x0.mean():.2f}")
axes[1].set_title('(b) Consensus on $G(A^P)$')
axes[1].set_ylabel('Value')
axes[1].grid(linestyle=':')

# (c) SDP-optimized + initial & augmented states
for i in range(N*5):
    axes[2].plot(Xa[i], alpha=0.3, linestyle='--', color='gray')
for i in range(N):
    axes[2].plot(X_sdp[i], linestyle='-', linewidth=1.5)
axes[2].axhline(x0.mean(), ls=':', color='black', label=f"True avg={x0.mean():.2f}")
axes[2].set_title('(c) Consensus with SDP-optimized weights')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('Step k')
axes[2].grid(linestyle=':')

plt.tight_layout()
plt.show()

# --- 5) Compute convergence steps and values ---------------------------
tol = 1e-5

# compute the actual consensus values from the last iteration
cons_a = Xo[:, -1].mean()
cons_b = Xa[:, -1].mean()
cons_c = X_sdp[:, -1].mean()

def conv_steps(X, target):
    sup = np.max(np.abs(X - target), axis=0)
    idx = np.where(sup < tol)[0]
    return idx[0] if idx.size > 0 else None

step_a = conv_steps(Xo, cons_a)
step_b = conv_steps(Xa, cons_b)
step_c = conv_steps(X_sdp, cons_c)

# print results safely
for label, step, cons in [('(a)', step_a, cons_a),
                          ('(b)', step_b, cons_b),
                          ('(c)', step_c, cons_c)]:
    if step is None:
        print(f"{label} did not converge within {steps} steps.")
    else:
        print(f"{label} converged in {step} steps to {cons:.6f}")
