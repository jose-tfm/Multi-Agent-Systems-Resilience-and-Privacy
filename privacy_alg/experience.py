import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Motif definitions (0=agent, 1–4=its four slots) -------------------

motifs = {
    'a': [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)],
    'b': [(0,1),(0,2),(0,3)],
    'c': [(0,1),(1,2),(2,0),(0,3),(0,4)],
    'd': [(0,1),(0,2),(1,2),(0,3),(3,4)],
    'e': [(0,1),(1,2),(2,3),(3,0),(0,4)],
    'f': [(0,1),(1,2),(2,3),(3,1),(0,4)],
    'g': [(0,1),(0,3),(1,2),(3,4)],
    'h': [(0,2),(2,3),(3,4),(2,4),(0,1)],
    'i': [(0,1),(1,2),(2,3),(3,0),(0,4)],
    'j': [(0,2),(2,1),(1,3),(3,2),(0,4)],
    'k': [(0,3),(3,1),(1,2),(2,3),(0,4)],
    'l': [(0,1),(0,2),(0,3),(1,3),(2,4)],
    'm': [(0,1),(1,2),(2,0),(0,3),(1,4),(2,4)],
    'n': [(0,1),(1,2),(2,3),(3,1),(3,4)],
    'o': [(0,1),(1,2),(2,3),(3,4),(4,1)],
    'p': [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)],
}

# --- Utilities -----------------------------------------------------------

def row_normalize(M):
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap_from_motif(N, A, motif_key, w_slot_slot=1/8):
    """Build the 5N×5N A^P for motif_key, auto‐computing w_agent_slot."""
    motif_edges = motifs[motif_key]
    size = 5 * N
    Ap = np.zeros((size, size))

    # 1) copy A/2
    Ap[:N, :N] = A / 2.0

    # 2) count spokes and set w_agent_slot so agent‐row sums to 1
    spokes = sum(1 for (u,v) in motif_edges if u==0 or v==0)
    w_agent_slot = 0.5 / spokes

    # 3) embed the motif per agent
    for i in range(N):
        idx = {0: i,
               1: N+4*i,
               2: N+4*i+1,
               3: N+4*i+2,
               4: N+4*i+3}
        # place edges
        for u,v in motif_edges:
            gu,gv = idx[u], idx[v]
            w = w_agent_slot if (u==0 or v==0) else w_slot_slot
            Ap[gu,gv] = w
            Ap[gv,gu] = w
        # renormalize each privacy‐slot row
        for slot in (1,2,3,4):
            u = idx[slot]
            off = Ap[u].sum() - Ap[u,u]
            Ap[u,u] = 1.0 - off

    # 4) ensure every row sums to 1
    return row_normalize(Ap)

def distributed_solve_alg4(N, Ap, x0):
    """
    Solve for s (detailed‐balance), vL, and xP0 so that:
      - s_i A^P_{i,j} = s_j A^P_{j,i}
      - vL = s/sum(s)
      - vL^T xP0 = mean(x0)
    """
    size = 5 * N
    s = np.zeros(size)
    s[0] = 1.0
    visited, rem = {0}, set(range(1, N))

    # propagate s on original N nodes
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i,j]>0 and Ap[j,i]>0:
                    s[j] = s[i] * Ap[i,j] / Ap[j,i]
                    visited.add(j)
                    rem.remove(j)
                    break
            else:
                continue
            break

    # detailed‐balance on each agent's slots
    for i in range(N):
        base = N + 4*i
        for k in range(4):
            s[base+k] = s[i] * Ap[i,base+k] / Ap[base+k,i]

    # stationary dist
    Z = s.sum()
    vL = s / Z

    # build xP0: true states + slots proportional to s
    xP0 = np.zeros(size)
    xP0[:N] = x0
    for i in range(N):
        xi = x0[i]
        base = N + 4*i
        total = s[base:base+4].sum()
        for k in range(4):
            xP0[base+k] = xi * (s[base+k] / total)

    # global rescale so vL^T xP0 = mean(x0)
    xP0 *= ( x0.mean() / (vL @ xP0) )

    return s, vL, xP0

# --- Main ---------------------------------------------------------------

if __name__ == "__main__":
    # network & init
    N = 3
    A = np.array([[0,0.5,0.5],
                  [0.5,0,0.5],
                  [0.5,0.5,0 ]])
    x0 = np.array([0.5, 1/3, 1/5])
    steps = 60

    # choose motif
    motif_key = 'd'  # try 'a'..'p'

    # build & init
    Ap = build_Ap_from_motif(N, A, motif_key)
    s, vL, xP0 = distributed_solve_alg4(N, Ap, x0)

    # simulate
    A_norm = row_normalize(A)
    Xo = np.zeros((N,steps+1))
    Xa = np.zeros((5*N,steps+1))
    Xo[:,0], Xa[:,0] = x0, xP0
    for k in range(steps):
        Xo[:,k+1] = A_norm @ Xo[:,k]
        Xa[:,k+1] = Ap      @ Xa[:,k]

    # plot
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))
    cmap = cm.get_cmap('tab10',N)

    # (a)
    for i in range(N):
        ax1.plot(Xo[i], color=cmap(i), label=f'$x_{{{i+1}}}[k]$')
    ax1.axhline(x0.mean(), ls='--', color='k', label=f'Consensus={x0.mean():.2f}')
    ax1.set(title='(a) Consensus on $G(A)$', xlabel='Step k', ylabel='Value')
    ax1.legend(loc='upper right'); ax1.grid(True)

    # (b)
    for i in range(N):
        ax2.plot(Xa[i], label=f'$\\tilde x_{{{i+1},1}}$')
        for j in range(4):
            ax2.plot(Xa[N+4*i+j], label=f'$\\tilde x_{{{i+1},{j+2}}}$')
    ax2.axhline(x0.mean(), ls='--', color='k', label=f'Consensus={x0.mean():.2f}')
    ax2.set(title=f'(b) Consensus on $G(A^P)$ — motif {motif_key}',
            xlabel='Step k', ylabel='Value')
    ax2.legend(ncol=5,fontsize='x-small',bbox_to_anchor=(1.05,1),loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # display computed weights
    spokes = sum(1 for (u,v) in motifs[motif_key] if u==0 or v==0)
    w_agent = 0.5 / spokes
    print(f"Motif {motif_key}: spokes={spokes}, w_agent_slot={w_agent:.4f}")
    print("detailed-balance s =", np.round(s,4))
