import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

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
    s = np.zeros(size); s[0] = 1.0
    visited, rem = {0}, set(range(1, N))
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i, j] > 0 and Ap[j, i] > 0:
                    s[j] = s[i] * Ap[i, j] / Ap[j, i]
                    visited.add(j); rem.remove(j); break
            else: continue
            break
    for i in range(N):
        si, base = s[i], N + 4 * i
        s[base+0] = 11/12 * si; s[base+1] = 1/4 * si
        s[base+2] = 1/3 * si;    s[base+3] = 2/3 * si
    Z = s.sum(); vL = s / Z
    xP0 = np.zeros(size)
    if ratios is None: ratios = np.ones(4)
    for i in range(N):
        xi, base = x0[i], N + 4 * i
        alphas = ratios / ratios.sum() * (5 * xi)
        for k in range(4):
            xP0[base+k] = (Z / s[base+k]) * alphas[k]
    # force primaries to x0
    xP0[:N] = x0
    xP0 *= (x0.mean() / (vL @ xP0))
    return s, vL, xP0

# --- main workflow ------------------------------------------------------
N = 11
A = np.array([
    [0, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 1/3],
    [1/3, 0, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0],
    [1/3, 1/3, 0, 1/3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1/3, 0, 1/3, 0, 0, 1/3, 0, 0, 0],
    [0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1/3, 0, 1/3, 0, 0, 1/3, 0],
    [0, 0, 0, 0, 0, 1/2, 0, 1/2, 0, 0, 0],
    [0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0, 0],
    [0, 0, 0, 0, 0, 1/2, 0, 1/2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1/3, 0, 0, 1/3, 1/3],
    [1/2, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 0]
])

x0 = np.array([0.1, 0.3, 0.6, 0.43, 0.85, 0.9, 0.45, 0.11, 0.06, 0.51, 0.13])
steps = 500


# 1) Consensus on original and augmented networks ---------------------
A_norm = row_normalize(A)
Xo = np.zeros((N, steps+1)); Xo[:,0] = x0
for k in range(steps): Xo[:,k+1] = A_norm @ Xo[:,k]

Ap = build_Ap_bidir_raw(N, A)
_,_,xP0 = distributed_solve_alg4(N, Ap, x0)
Xa = np.zeros((5*N, steps+1)); Xa[:,0] = xP0
for k in range(steps): Xa[:,k+1] = Ap @ Xa[:,k]

# 2) SDP-based weight optimization -------------------------------------
mask = (A > 0).astype(float)
W = cp.Variable((N, N), nonneg=True)
gamma = cp.Variable()
L = cp.diag(cp.sum(W, axis=1)) - W
P = np.eye(N) - np.ones((N, N)) / N
prob = cp.Problem(cp.Maximize(gamma), [
    cp.multiply(mask, W) == W,
    cp.sum(W, axis=1) == 1,
    L - gamma * P >> 0
])
prob.solve(solver=cp.SCS)
A_opt = W.value
print('A_opt', A_opt)

# 3) Consensus with optimized weights -------------------------------
X_sdp = np.zeros((N, steps+1)); X_sdp[:,0] = x0
for k in range(steps): X_sdp[:,k+1] = A_opt @ X_sdp[:,k]

# 4) Plot comparisons -------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
for i in range(N): axes[0].plot(Xo[i],'--',alpha=0.6)
axes[0].set_title('(a) Consensus on G(A)'); axes[0].grid(':')
for i in range(5*N): axes[1].plot(Xa[i],alpha=0.6)
axes[1].set_title('(b) Consensus on G(A^P)'); axes[1].grid(':')
for i in range(5*N): axes[2].plot(Xa[i],alpha=0.3,linestyle='--',color='gray')
for i in range(N): axes[2].plot(X_sdp[i],linewidth=1.5)
axes[2].set_title('(c) Consensus with optimized weights'); axes[2].grid(':')
plt.tight_layout(); plt.show()

# 5) Compute convergence to each scenario's final mean ----------------
cons_a = Xo[:N,-1].mean(); cons_b = Xa[:N,-1].mean(); cons_c = X_sdp[:N,-1].mean()
time_tol = 1e-5

def conv_info(X, target):
    err = np.max(np.abs(X[:N,:] - target), axis=0)
    idx = np.where(err < time_tol)[0]
    return idx[0] if idx.size else None

sa = conv_info(Xo, cons_a); sb = conv_info(Xa, cons_b); sc = conv_info(X_sdp, cons_c)

for label, step, cons in [('(a)', sa, cons_a), ('(b)', sb, cons_b), ('(c)', sc, cons_c)]:
    if step is None:
        print(f"{label} did not converge to ~{cons:.6f} within {steps} steps.")
    else:
        print(f"{label} converged to ~{cons:.6f} in {step} steps.")
