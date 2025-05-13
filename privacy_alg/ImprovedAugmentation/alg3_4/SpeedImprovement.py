import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eigvals
from scipy.optimize import minimize

# --- utilities ----------------------------------------------------------

def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)


def build_Ap_param_raw(N, A, w):
    """
    Build raw lifted matrix A^P with variable lift weights w = [w1,w2,w3,w4],
    replacing the hardcoded 1/12,1/8,1/4,1/24 in Algorithm 3.
    No row normalization here; this is the raw A^P.
    """
    size = 5 * N
    Ap = np.zeros((size, size))
    # top-left block (untouched)
    Ap[:N, :N] = A / 2.0

    # base and back ratios from Alg3
    base_ratios = np.array([1/12, 1/8, 1/4, 1/24])
    back_ratios = np.array([1/11, 1/2, 3/4, 1/16])
    # between-lifts ratio matrix R
    R = np.zeros((4,4))
    R[0,1], R[0,2], R[0,3] = 3/22, 1/11, 15/22
    R[1,0], R[2,0], R[3,0] = 1/2,   1/4,   15/16

    for i in range(N):
        base = N + 4*i
        # i -> lifts using w
        Ap[i, base:base+4] = w
        # lifts -> i scaled by back/base ratios
        for k in range(4):
            Ap[base+k, i] = w[k] * (back_ratios[k] / base_ratios[k])
        # between lifts
        for j in range(4):
            for k in range(4):
                if R[j,k] > 0:
                    Ap[base+j, base+k] = w[j] * (R[j,k] / base_ratios[j])
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


def spectral_gap_obj(w, N, A):
    """Build P(w)=row_normalize(A^P_raw(w)), return SLEM."""
    Ap_raw = build_Ap_param_raw(N, A, w)
    P = row_normalize(Ap_raw)
    vals = np.sort(np.abs(eigvals(P)))
    return vals[-2].real

# --- main script -------------------------------------------------------
if __name__ == '__main__':
    # parameters
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

    # optimize lift weights
    w0 = np.ones(4)*0.5/4
    cons = [
        {'type':'eq',   'fun': lambda w: np.sum(w) - 0.5},
        {'type':'ineq','fun': lambda w: w}
    ]
    res = minimize(spectral_gap_obj, w0, args=(N,A),
                   constraints=cons, method='SLSQP', options={'ftol':1e-8, 'disp':True})
    w_opt = res.x
    print('Optimal w:', w_opt, 'Min SLEM:', res.fun)

    # rebuild P and initial xP0 via alg4
    Ap_raw_opt = build_Ap_param_raw(N, A, w_opt)
    P_opt = row_normalize(Ap_raw_opt)
    s, vL, xP0 = distributed_solve_alg4(N, Ap_raw_opt, x0)

    # consensus simulations
    steps = 60
    A_norm = row_normalize(A)
    Xo = np.zeros((N, steps+1)); Xa = np.zeros((5*N, steps+1))
    Xo[:,0] = x0; Xa[:,0] = xP0
    for k in range(steps):
        Xo[:,k+1] = A_norm @ Xo[:,k]
        Xa[:,k+1] = P_opt @ Xa[:,k]

    # plotting
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
    colors = cm.get_cmap('tab10',N)
    for i in range(N): ax1.plot(Xo[i],color=colors(i),label=f'x{i+1}[k]')
    ax1.axhline(x0.mean(),ls='--',color='black',label=f'Consensus={x0.mean():.2f}')
    ax1.set(title='(a) Consensus on G(A)',xlabel='k',ylabel='Value'); ax1.grid(); ax1.legend(fontsize='small')
    for i in range(N):
        ax2.plot(Xa[i],label=f'~x{i+1},1[k]')
        for j in range(4): ax2.plot(Xa[N+4*i+j])
    ax2.axhline(x0.mean(),ls='--',color='black')
    ax2.set(title='(b) Consensus on G(A^P)',xlabel='k',ylabel='Value'); ax2.grid()
    plt.tight_layout(); plt.show()
