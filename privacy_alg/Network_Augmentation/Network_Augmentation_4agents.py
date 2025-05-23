import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# --- main visualization ------------------------------------------------

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

Ap = build_Ap_bidir_raw(N, A)
print('Esta é a matrix', Ap)
s, vL, xP0 = distributed_solve_alg4(N, Ap, x0)
print('vL', vL, 'xP0', xP0)

steps = 60
A_norm = row_normalize(A)
Xo = np.zeros((N, steps+1))
Xa = np.zeros((5*N, steps+1))
Xo[:,0], Xa[:,0] = x0, xP0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]
    Xa[:,k+1] = Ap      @ Xa[:,k]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot (a)
colors = cm.get_cmap('tab10', N)
for i in range(N):
    ax1.plot(range(steps+1), Xo[i], label=fr'$x_{{{i+1}}}[k]$', color=colors(i))
ax1.axhline(x0.mean(), ls='--', color='black', label='Consensus={:.2f}'.format(x0.mean()))
ax1.set_title(r"(a) Consensus on $G(A)$")
ax1.set_xlabel("Step k")
ax1.set_ylabel("Value")
ax1.grid()
ax1.legend(fontsize='small')

# Plot (b)
for i in range(N):
    ax2.plot(range(steps+1), Xa[i], label=fr'$\tilde{{x}}_{{{i+1},1}}[k]$')
    for j in range(4):
        ax2.plot(range(steps+1), Xa[N + 4*i + j], label=fr'$\tilde{{x}}_{{{i+1},{j+2}}}[k]$')
ax2.axhline(x0.mean(), ls='--', color='black', label='Consensus={:.2f}'.format(x0.mean()))
ax2.set_title(r"(b) Consensus on $G(A^P)$")
ax2.set_xlabel("Step k")
ax2.set_ylabel("Value")
ax2.grid()
ax2.legend(ncol=5, fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
