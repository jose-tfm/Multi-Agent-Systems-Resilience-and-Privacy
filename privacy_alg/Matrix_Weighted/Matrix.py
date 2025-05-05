import numpy as np
import matplotlib.pyplot as plt

# ─── Make everything deterministic ──────────────────────────────────────────
np.random.seed(42)

# ─── Parameters ─────────────────────────────────────────────────────────────
n, d, d_prime = 5, 3, 3
D = d + d_prime
sigma = 2.0
max_iter = 120
T = D - 1
neighbors = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3, 4],
    3: [1, 2, 4],
    4: [2, 3]
}

# ─── Initial real states x(0) ──────────────────────────────────────────────
x_real0 = np.array([
    [ 1,  2,  3],
    [ 4,  5,  6],
    [ 7,  8,  9],
    [10, 11, 12],
    [13, 14, 15]
], dtype=float)
avg_real = x_real0.mean(axis=0)

# ─── Build orthogonal set V as per Appendix A.1 (37)–(42) ─────────────────
V = []

# v1 = [1,1,1,0,0,0]^T
v1 = np.concatenate([np.ones(d), np.zeros(d_prime)])
V.append(v1)

# v2 … v(D-1) for i=2…5 (1-based)
for i in range(2, D):
    v = np.zeros(D)
    inv = 1.0 / i
    # first coordinate +1/i, next i-1 coords = -1/i
    v[0]    = +inv
    v[1:i]  = -inv
    # one 1 at position i-1 (0-based index = i-1 → but Appendix uses 1-based)
    v[i-1] += 1.0  # this adds to the existing ±1/i entry
    V.append(v)

# v6 = [1/6, -1/6, ..., -1/6]
invD = 1.0 / D
vD = np.full(D, -invD)
vD[0] = +invD
V.append(vD)

# (Optionally you can normalize, but not strictly necessary for orthogonality)

# ─── Build Φ for k=0 ────────────────────────────────────────────────────────
# same as eq(37): Φ = (v1 v1^T)/(v1^T v1)
Phi = np.outer(v1, v1) / (v1 @ v1)

# ─── Static one-step update (Fig.7) ────────────────────────────────────────
# Raw 3-D projector for Fig.7(a–c)
v_raw = np.ones(d); v_raw /= np.linalg.norm(v_raw)
Phi_raw = np.outer(v_raw, v_raw)

x1_raw = x_real0.copy()
for i in range(n):
    diff = sum((Phi_raw @ x_real0[j] - Phi_raw @ x_real0[i]) for j in neighbors[i])
    x1_raw[i] += sigma * diff

# Lifted 6-D single-step (Fig.7(d–f))
alpha = 0.0
xv0 = np.full((n, d_prime), alpha)
xt0 = np.hstack((xv0, x_real0))
xt1 = xt0.copy()
for i in range(n):
    for j in neighbors[i]:
        A0 = Phi  # Pij = I
        delta = (A0 @ xt0[j] - A0 @ xt0[i])
        xt1[i] += sigma * delta

x0_lift = xt0[:, d_prime:]
x1_lift = xt1[:, d_prime:]

# ─── Full dynamic run for Fig.5 & Fig.6 ────────────────────────────────────
xt = xt0.copy()
traj = np.zeros((max_iter+1, n, D))
traj[0] = xt

# static k=0 with projector onto V[0]
Phi0 = np.outer(V[0], V[0])
for i in range(n):
    diff = sum((Phi0 @ xt[j] - Phi0 @ xt[i]) for j in neighbors[i])
    xt[i] += sigma * diff
traj[1] = xt.copy()

# pre-generate gamma/zeta sequences
gamma_seq = np.random.rand(max_iter+1) * (1/(4*(n-1)*sigma))
zeta_seq  = np.random.rand(max_iter+1) * (1/(4*(n-1)*sigma))

# record y_{3→2}
y32 = np.zeros((max_iter+1, d))

for k in range(1, max_iter+1):
    rho = k % T or T
    vr, vlast = V[rho-1], V[-1]

    # record the message from 3→2
    g, z = gamma_seq[k], zeta_seq[k]
    A23 = g*np.outer(vr, vr) + z*np.outer(vlast, vlast)
    y32[k] = (A23 @ xt[2])[d_prime:]

    # full consensus update
    new = xt.copy()
    for i in range(n):
        acc = np.zeros(D)
        for j in neighbors[i]:
            g, z = gamma_seq[k], zeta_seq[k]
            Aij = g*np.outer(vr, vr) + z*np.outer(vlast, vlast)
            acc += Aij @ (xt[j] - xt[i])
        new[i] += sigma * acc
    xt = new
    traj[k] = xt

real_traj = traj[:, :, d_prime:]  # (k, i, ℓ)

# ─── Plot Fig.5: evolution of x_{iℓ}(k) ─────────────────────────────────────
for ℓ in range(d):
    plt.figure()
    for i in range(n):
        plt.plot(real_traj[:, i, ℓ], label=f'Agent {i+1}')
    plt.plot(
        [avg_real[ℓ]]*(max_iter+1),
        color='k', linestyle='--', label='Avg(x(0))'
    )
    plt.title(f'Fig. 5({chr(97+ℓ)})')
    plt.xlabel('k')
    plt.ylabel(rf'$x_{{i{ℓ+1}}}(k)$')
    plt.legend()
    plt.grid()
    plt.show()

# ─── Plot Fig.6: x_3 vs y_{3→2} ────────────────────────────────────────────
for ℓ in range(d):
    plt.figure()
    plt.plot(real_traj[:, 2, ℓ], label=rf'$[x_3(k)]_{{{ℓ+1}}}$')
    plt.plot(y32[:, ℓ],         label=rf'$[y_{{3\to2}}(k)]_{{{ℓ+1}}}$')
    plt.plot(
        [avg_real[ℓ]]*(max_iter+1),
        color='k', linestyle='--', label='Avg(x(0))'
    )
    plt.title(f'Fig. 6({chr(97+ℓ)})')
    plt.xlabel('k')
    plt.legend()
    plt.grid()
    plt.show()

# ─── Plot Fig.7: one-step static (raw vs lifted) ───────────────────────────
fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey='row')
kvals = [0, 1]

# raw
for ℓ in range(d):
    ax = axs[0, ℓ]
    for i in range(n):
        ax.plot(kvals, [x_real0[i, ℓ], x1_raw[i, ℓ]], 'o-')
    ax.hlines(
        avg_real[ℓ], 0, 1,
        colors='k', linestyles='--'
    )
    ax.set_title(f'({chr(97+ℓ)}) raw')
    ax.set_xticks(kvals)
    if ℓ == 0:
        ax.set_ylabel(r'$x_{i\ell}(k)$')
    ax.grid(True)

# lifted
for ℓ in range(d):
    ax = axs[1, ℓ]
    for i in range(n):
        ax.plot(kvals, [x0_lift[i, ℓ], x1_lift[i, ℓ]], 'o-')
    ax.hlines(
        avg_real[ℓ], 0, 1,
        colors='k', linestyles='--'
    )
    ax.set_title(f'({chr(100+ℓ)}) lifted')
    ax.set_xticks(kvals)
    if ℓ == 0:
        ax.set_ylabel(r'$\tilde x_{i\ell}(k)$')
    ax.grid(True)

plt.tight_layout()
plt.show()
