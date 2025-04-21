import numpy as np
import matplotlib.pyplot as plt

def row_normalize(M):
    """Make each row of M sum to 1 (row‑stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap_bidir(N, A):
    """
    Algorithm 3:
    Build the 5N×5N privacy‐augmented matrix A^P for a
    bidirectional, time‐reversible dynamics A (row‐stochastic).
    """
    size = 5 * N
    Ap = np.zeros((size, size))

    # Step 4: copy A/2 into the top‐left N×N block
    Ap[:N, :N] = A / 2.0

    # Step 5: for i_alg = 1..N (paper’s 1-based index)
    for i_alg in range(1, N+1):
        i0 = i_alg - 1            # Python’s 0-based index for the original state
        # compute the four privacy‐slot indices (1-based → 0-based)
        j1 = (N + 4*i_alg - 3) - 1
        j2 = (N + 4*i_alg - 2) - 1
        j3 = (N + 4*i_alg - 1) - 1
        j4 = (N + 4*i_alg    ) - 1

        # original → privacy
        Ap[i0, j1] = 1/12
        Ap[i0, j2] =  1/8
        Ap[i0, j3] =  1/4
        Ap[i0, j4] = 1/24

        # privacy → original
        Ap[j1, i0] = 1/11
        Ap[j2, i0] =   1/2
        Ap[j3, i0] =   3/4
        Ap[j4, i0] = 1/16

        # inter‑privacy “downward” links from j1
        Ap[j2, j1] =  1/2
        Ap[j3, j1] =  1/4
        Ap[j4, j1] = 15/16

        # inter‑privacy “rightward” links from j1
        Ap[j1, j2] =  3/22
        Ap[j1, j3] =   1/11
        Ap[j1, j4] = 15/22

    # final row‐stochastic normalization
    return row_normalize(Ap)


def distributed_solve_alg4(N, Ap, x0):
    """
    Algorithm 4:
    - Compute the reversible weights s (first N entries) by a
      breadth‐first reach via edges with both Ap[i,j] and Ap[j,i] > 0.
    - Assign the privacy‐slot weights (lines 10–13).
    - Normalize to get the left‐eigenvector vL.
    - Distribute each x0[i] into its 4 privacy slots (lines 15–20).
    - Global rescale so vL^T xP0 = average(x0).
    """
    size = 5 * N
    s    = np.zeros(size)
    s[0] = 1.0

    # 4–9) grow s for the original N nodes via reversibility
    visited   = {0}
    remaining = set(range(1, N))
    while remaining:
        for j in list(remaining):
            for i in visited:
                if Ap[i,j] > 0 and Ap[j,i] > 0:
                    s[j] = s[i] * Ap[i,j] / Ap[j,i]
                    visited.add(j)
                    remaining.remove(j)
                    break
            else:
                continue
            break

    # 10–13) assign weights to each agent’s 4 privacy slots
    for i0 in range(N):
        si   = s[i0]
        base = N + 4*i0
        s[ base + 0 ] = (11/12) * si
        s[ base + 1 ] =   (1/4) * si
        s[ base + 2 ] =   (1/3) * si
        s[ base + 3 ] =   (2/3) * si

    # normalize → left‐eigenvector vL (sum to 1)
    Z  = s.sum()
    vL = s / Z

    # 15–20) distribute each x0[i] into its privacy slots
    xP0 = np.zeros(size)
    for i0 in range(N):
        xi     = x0[i0]
        # pick α₁,…,α₄ > 0 with (1/5) Σ α_j = xi  => Σ α_j = 5·xi
        # simplest: α_j = (5·xi)/4 each
        alphas = np.full(4, 5*xi/4)
        base   = N + 4*i0

        # original slot = 0
        xP0[i0] = 0.0
        # fill the four privacy slots
        for k in range(4):
            idx      = base + k
            xP0[idx] = (Z / s[idx]) * alphas[k]

    # final global rescale so vLᵀ xP0 = average(x0)
    avg     = x0.mean()
    factor  = avg / (vL @ xP0)
    xP0    *= factor

    return vL, xP0


if __name__ == "__main__":
    # --- Example 1: 3‑cycle ---
    N  = 3
    A  = np.array([
        [0,   0.5, 0.5],
        [0.5, 0,   0.5],
        [0.5, 0.5, 0  ]
    ], dtype=float)

    # initial states [1/2, 1/3, 1/5]
    x0    = np.array([0.5, 1/3, 0.2])
    A_norm = row_normalize(A)

    # build & solve
    Ap      = build_Ap_bidir(N, A)
    vL, xP0 = distributed_solve_alg4(N, Ap, x0)

    # simulate both systems
    steps = 30
    Xo = np.zeros((N,      steps+1))
    Xa = np.zeros((5*N,    steps+1))
    Xo[:,0] = x0
    Xa[:,0] = xP0

    for k in range(steps):
        Xo[:,k+1] = A_norm @ Xo[:,k]
        Xa[:,k+1] = Ap     @ Xa[:,k]

    # true consensus = average(x0)
    consensus = x0.mean()

    # --- plotting ---
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,10))

    # (a) original
    for i in range(N):
        ax1.plot(Xo[i], label=f'$x_{{{i+1}}}[k]$')
    ax1.axhline(consensus, color='k', lw=1, label=f'Consensus={consensus:.2f}')
    ax1.set_title('(a) Consensus on the 3‑cycle $G(A)$')
    ax1.set_xlabel('Step $k$');  ax1.set_ylabel('Value')
    ax1.legend(loc='upper right');  ax1.grid(True)

    # (b) augmented
    for i in range(5*N):
        if i < N:
            lbl = f'$x_{{{i+1}}}[k]$'
        else:
            agent = (i - N)//4 + 1
            slot  = (i - N)%4 + 1
            lbl   = f'$\\tilde x_{{{agent},{slot}}}[k]$'
        ax2.plot(Xa[i], label=lbl)
    ax2.axhline(consensus, color='k', lw=1, label=f'Consensus={consensus:.2f}')
    ax2.set_title('(b) Consensus on the augmented $G(A^P)$')
    ax2.set_xlabel('Step $k$');  ax2.set_ylabel('Value')
    ax2.legend(ncol=4, loc='upper right', fontsize='small')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
