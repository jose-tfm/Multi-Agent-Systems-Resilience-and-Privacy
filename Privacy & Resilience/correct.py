import numpy as np
import matplotlib.pyplot as plt
import itertools

# ——————————————————————————————————————————————————————————————
# 1) HELPERS & PRIVACY-AUGMENTATION
# ——————————————————————————————————————————————————————————————
np.set_printoptions(precision=3, suppress=True)

def row_normalize(M):
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(n, A):
    Ap = np.zeros((4*n, 4*n))
    Ap[:n, :n] = A
    for i in range(N):
        inds = {
            0: i,          
            1: N + 3*i,   
            2: N + 3*i+1, 
            3: N + 3*i+2 
            }
        
        Ap[inds[0], inds[1]] = 0.043
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1.134
        Ap[inds[2], inds[0]] = 2.027
        Ap[inds[0], inds[2]] = 0.09
        Ap[inds[0], inds[3]] = 0.043
        Ap[inds[3], inds[2]] = 1
        
    return row_normalize(Ap)

def minor(A, F):
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep

# ——————————————————————————————————————————————————————————————
# 2) PROBLEM SETUP (hard-coded A)
# ——————————————————————————————————————————————————————————————
# Privacy-augmented adjacency → now directly given as an 11×11 row-stochastic matrix
A = np.array([
 [0.   , 0.751, 0.196, 0.033, 0.021, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
 [0.   , 0.   , 0.75 , 0.196, 0.033, 0.021, 0.   , 0.   , 0.   , 0.   , 0.   ],
 [0.019, 0.   , 0.   , 0.752, 0.197, 0.033, 0.   , 0.   , 0.   , 0.   , 0.   ],
 [0.   , 0.029, 0.   , 0.   , 0.752, 0.196, 0.023, 0.   , 0.   , 0.   , 0.   ],
 [0.   , 0.   , 0.029, 0.   , 0.   , 0.752, 0.196, 0.023, 0.   , 0.   , 0.   ],
 [0.   , 0.   , 0.   , 0.029, 0.   , 0.   , 0.751, 0.196, 0.023, 0.   , 0.   ],
 [0.   , 0.   , 0.   , 0.   , 0.029, 0.   , 0.   , 0.751, 0.196, 0.023, 0.   ],
 [0.   , 0.   , 0.   , 0.   , 0.   , 0.029, 0.   , 0.   , 0.751, 0.196, 0.023],
 [0.023, 0.   , 0.   , 0.   , 0.   , 0.   , 0.029, 0.   , 0.   , 0.751, 0.196],
 [0.196, 0.023, 0.   , 0.   , 0.   , 0.   , 0.   , 0.029, 0.   , 0.   , 0.752],
 [0.752, 0.196, 0.023, 0.   , 0.   , 0.   , 0.   , 0.   , 0.029, 0.   , 0.   ]
])
print("Using hard-coded A:\n", A, "\n")

# automatically pick up the size
N      = A.shape[0]
agents = list(range(1, N+1))

# fault-tolerance, privacy parameter, simulation length
f = 1
ε = 0.05
T = 60

# ——————————————————————————————————————————————————————————————
# 2b) INITIAL STATES
# ——————————————————————————————————————————————————————————————
# Define an initial state x0[u] for each u in 1…11
# (replace these example values with your true initial values)
x0 = {
    1:  0.10,  2: 0.30,  3: 0.35,  4: 0.60,  5: 0.55,
    6:  0.50,  7: 0.45,  8: 0.40,  9: 0.35, 10: 0.30,
   11:  0.25
}

# attacker behavior (agent 2 always sends 0.30 in this example)
attacker_val = {
    2: (lambda k: 0.30)
}

honest_avg = np.mean([x0[u] for u in agents if u not in attacker_val])
print("Honest average =", honest_avg, "\n")

# ——————————————————————————————————————————————————————————————
# 3) ALL FAULT-SETS |F|≤f
# ——————————————————————————————————————————————————————————————
F_list = sorted(
    [frozenset(c)
     for k in range(f+1)
     for c in itertools.combinations(agents, k)],
    key=lambda S: (len(S), sorted(S))
)
idx_of = {F: i for i, F in enumerate(F_list)}
print("Fault sets:", F_list, "\n")

# ——————————————————————————————————————————————————————————————
# 4) ALGORITHM 2: PRIVATE INIT (build P_pa, v0, x_priv)
# ——————————————————————————————————————————————————————————————
P_list      = []
surv_idxs   = []
x_priv_list = []

for F in F_list:
    print("=== F =", F, "===")
    A_sub, surv = minor(A, [u-1 for u in F])
    A_sub = row_normalize(A_sub)
    n_sub = len(surv)
    print(" A_sub =\n", np.round(A_sub,3), "\n")

    P_pa = build_Ap(n_sub, A_sub)
    print(" P_pa =\n", np.round(P_pa,3), "\n")

    # stationary left eigenvector
    w, V = np.linalg.eig(P_pa.T)
    i1   = np.argmin(np.abs(w-1))
    v0   = np.real(V[:, i1]);  v0 /= v0.sum()
    print(" v^0 =", np.round(v0,3))

    # closed-form private init
    x_sub   = np.array([x0[agents[i]] for i in surv])
    x_priv  = np.zeros(4*n_sub)
    for j, u_idx in enumerate(surv):
        xj    = x_sub[j]
        slots = [j, n_sub+3*j, n_sub+3*j+1, n_sub+3*j+2]
        for ℓ in slots:
            x_priv[ℓ] = xj / (4 * n_sub * v0[ℓ])

    # zero out the “real” slots
    x_priv[:n_sub] = 0.0

    # scale so that v0·x_priv = target = mean of x_sub
    target  = x_sub.mean()
    current = v0 @ x_priv
    x_priv *= (target / current)

    print(" v0·x_priv =", float(v0 @ x_priv), " target=", target)

    P_list.append(P_pa)
    surv_idxs.append(surv)
    x_priv_list.append(x_priv)

# ——————————————————————————————————————————————————————————————
# 5) SIMULATION: USE P_pa, THEN CLAMP ALL 3 SLOTS OF AGENT 2 TO 0.3
# ——————————————————————————————————————————————————————————————
X_list = [np.zeros((4*len(s), T+1)) for s in surv_idxs]
for i in range(len(F_list)):
    X_list[i][:,0] = x_priv_list[i]

for k in range(T):
    for i, Pm in enumerate(P_list):
        surv = surv_idxs[i]
        n_sub = len(surv)
        X_list[i][:, k+1] = Pm @ X_list[i][:, k]
        # clamp attacker (agent 2) if present
        if 2 in surv:
            j_real = surv.index(2-1)
            a1, a2, a3 = n_sub+3*j_real, n_sub+3*j_real+1, n_sub+3*j_real+2
            X_list[i][[a1, a2, a3], k+1] = 0.3

# ——————————————————————————————————————————————————————————————
# 6) COLLAPSE → candidate vectors c[u][k]
# ——————————————————————————————————————————————————————————————
c = {u: {} for u in agents}
for u in agents:
    for k in range(T+1):
        vec = []
        for i, F in enumerate(F_list):
            surv = surv_idxs[i]
            if u in attacker_val:
                vec.append(attacker_val[u](k))
            elif u in F:
                vec.append(x0[u])
            else:
                j = surv.index(u-1)
                vec.append(X_list[i][j, k])
        c[u][k] = vec

# ——————————————————————————————————————————————————————————————
# 7) ALGORITHM 3: RESILIENT SELECTION & HISTORY
# ——————————————————————————————————————————————————————————————
x_hist = {u: [x0[u]] for u in agents}

for k in range(1, T+1):
    print(f"\n--- Round {k} ---")
    for u in agents:
        full = c[u][k][idx_of[frozenset()]]
        if u in attacker_val:
            x_next = attacker_val[u](k)
        else:
            bads = [
                (i, abs(c[u][k][i] - full))
                for i, F in enumerate(F_list)
                if F and (u not in F) and abs(c[u][k][i] - full) >= ε
            ]
            if len(bads) == 1:
                x_next = c[u][k][bads[0][0]]
                print(f"  Agent {u}: outlier → F={F_list[bads[0][0]]}")
            else:
                x_next = full
                print(f"  Agent {u}: keep full={full:.3f}")
        x_hist[u].append(x_next)

# ——————————————————————————————————————————————————————————————
# 8) PLOT REAL-SLOT TRAJECTORIES
# ——————————————————————————————————————————————————————————————
i0 = idx_of[frozenset()]    # the “no-faults” case
X0 = X_list[i0]             # shape is (4*N, T+1)

plt.figure(figsize=(8,4))
for u in agents:
    plt.plot(x_hist[u], label=f'$x_{{{u}}}^{{(k)}}$')

plt.axhline(honest_avg, color='k', ls='--', lw=1,
            label=f'honest avg={honest_avg:.2f}')
plt.title('Alg 3: Private & Resilient Consensus')
plt.xlabel('Iteration $k$')
plt.ylabel('State')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
