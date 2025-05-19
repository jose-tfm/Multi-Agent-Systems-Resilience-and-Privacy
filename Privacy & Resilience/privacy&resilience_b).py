import numpy as np
import matplotlib.pyplot as plt
import itertools
import os, sys

# — ensure plot_state is importable —
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.utils import plot_state


# ——— Algorithm 2 helpers —————————————————————————————————————————

def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(n, A):
    """
    Build the 4n×4n privacy-augmented matrix A^P from
    an n×n base matrix A.
    """
    P = np.zeros((4*n, 4*n))
    P[:n, :n] = A
    for i in range(n):
        a1, a2, a3 = n+3*i, n+3*i+1, n+3*i+2
        P[i,  a1] = 1
        P[a1, i ] = 2
        P[i,  a2] = 1
        P[a2, a3] = 1
        P[a3, i ] = 1
    return row_normalize(P)

def minor(A, F):
    """Remove rows & cols in index-set F from A."""
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep


# ——— Problem setup for Fig 2(e): G₂, A₂={2,3} —————————————————————

all_agents = [1,2,3,4,5]
N          = len(all_agents)
f          = 2           # resilience parameter
ε          = 0.05        # precision threshold
T          = 20          # number of iterations


x0_dict = {1: 0.10, 2: 0.30, 3: 0.35, 4: 0.60, 5: 0.55}

# attacked agents 2 and 3 with time-varying misbehavior
attacker_vals = {
    2: lambda k: 0.3 - 0.5/(1 + k**2),
    3: lambda k: 0.3 + 0.6/(1 + k**2),
}

# adjacency list of G₂ (undirected)
adj = {
    1: [3,5,4],
    2: [1,3,4,5],
    3: [1,2,4,5],
    4: [2,1,5],
    5: [1,2,3],
}

# build the N×N row-stochastic A from adj
A = np.zeros((N,N))
for u in all_agents:
    nbrs = adj[u]
    w    = 1/len(nbrs)
    for v in nbrs:
        A[u-1, v-1] = w


# ——— 1) Enumerate all failure-sets F ⊆ V, |F| ≤ f ——————————————————

F_list = []
for k in range(f+1):
    for combo in itertools.combinations(all_agents, k):
        F_list.append(frozenset(combo))
F_list = sorted(F_list, key=lambda S: (len(S), sorted(S)))
idx_of = {F:i for i, F in enumerate(F_list)}


# ——— 2) Algorithm 2: compute each survivor’s private init for each F ——

# tilde_init[F][u] = the scalar initial candidate for u under subset F
tilde_init = {F: {} for F in F_list}

for F in F_list:
    # a) form minor of A & x0 without F
    A_sub, surv_idx = minor(A, [u-1 for u in F])
    x_sub           = np.array([ x0_dict[u] for u in all_agents if u not in F ])
    n_sub           = len(surv_idx)

    # b) privacy-augment this subgraph
    P_sub = build_Ap(n_sub, A_sub)

    # c) compute left eigenvector v0 of P_sub^T
    w, V  = np.linalg.eig(P_sub.T)
    i1    = np.argmin(np.abs(w - 1))
    v0    = np.real(V[:, i1])
    v0   /= v0.sum()

    # d) split each x_sub into three shares
    x_priv = np.zeros(4*n_sub)
    for j_sub, u_idx in enumerate(surv_idx):
        u = u_idx + 1
        # privacy-splitting coefficients
        a, b, g = 1.4, 1.0, 1.0
        S       = a + b + g
        coeff   = 4 * x_sub[j_sub] / S
        base    = n_sub + 3*j_sub
        x_priv[base+0] = coeff * a
        x_priv[base+1] = coeff * b
        x_priv[base+2] = coeff * g

    # e) global rescale so v0^T x_priv = average(x_sub)
    target  = x_sub.mean()
    current = v0 @ x_priv
    x_priv *= (target / current)

    # f) collapse each triple back to one scalar
    for j_sub, u_idx in enumerate(surv_idx):
        u    = u_idx + 1
        base = n_sub + 3*j_sub
        tilde_init[F][u] = (
            x_priv[base] +
            x_priv[base+1] +
            x_priv[base+2]
        )


# ——— 3) Algorithm 3 init: build c[u][0][F] for every u, F —————————

# c[u][t] will be a list of length |F_list|
c = {u: {0: [None]*len(F_list)} for u in all_agents}

for i, F in enumerate(F_list):
    for u in all_agents:
        if u in attacker_vals:
            # attacked nodes seed with their own misbehavior
            c
        else:
            # benign nodes: if u∈F, fallback to raw x0; else use tilde_init
            if u not in tilde_init[F]:
                c[u][0][i] = x0_dict[u]
            else:
                c[u][0][i] = tilde_init[F][u]


# ——— 4) Candidate update & selection loop (Alg 3) ——————————————————

def normal_step(u, k, F):
    nbrs = [v for v in adj[u] if v not in F]
    vals = [c[v][k][ idx_of[F] ] for v in nbrs]
    # drop any None
    vals = [v for v in vals if v is not None]
    if not vals:
        return c[u][k][ idx_of[F] ]
    return sum(vals) / len(vals)

def adv_step(u, k, F):
    return attacker_vals[u](k)

step = {
    u: (adv_step if u in attacker_vals else normal_step)
    for u in all_agents
}

# store the selected x_u[k] values
x_hist = {u: [ x0_dict[u] ] for u in all_agents}

for k in range(T):
    # (a) compute every candidate c_u[k+1][F]
    for u in all_agents:
        c[u][k+1] = [None]*len(F_list)
        for i, F in enumerate(F_list):
            c[u][k+1][i] = step[u](u, k, F)

    # (b) selection: pick x_u[k+1]
    for u in all_agents:
        full = c[u][k+1][ idx_of[frozenset()] ]
        bad  = []
        for i, F in enumerate(F_list):
            if u not in F and len(F)>0:
                if abs(c[u][k+1][i] - full) >= ε:
                    bad.append(i)
        x_next = c[u][k+1][ bad[0] ] if len(bad)==1 else full
        x_hist[u].append(x_next)


# ——— 5) Plot the evolution —————————————————————————————————

states   = np.vstack([np.array(x_hist[u]) for u in sorted(all_agents)]).T
true_avg = np.mean([x0_dict[u] for u in all_agents if u not in attacker_vals])

plot_state(
    states,
    true_avg,
    attacker_vals,
    sorted(all_agents),
    title  = "Alg 3: Private & Resilient Consensus on G₂",
    xlabel = "Iteration k",
    ylabel = "x_u[k]"
)
