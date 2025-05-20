import numpy as np
import matplotlib.pyplot as plt
import itertools

# ——————————————————————————————————————————————————————————————
# 1) HELPERS & PRIVACY-AUGMENTATION (Alg 1)
# ——————————————————————————————————————————————————————————————
np.set_printoptions(precision=3, suppress=True)

def row_normalize(M):
    """Make each row of M sum to 1."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(n, A):
    """
    Build the 4n×4n privacy-augmented matrix P^P:
      - P[:n,:n] = A
      - real↔aug1 weight 1, real→aug2 weight 2,
      - aug2→aug3→real chain weight 1
    """
    P = np.zeros((4*n, 4*n))
    P[:n, :n] = A
    for i in range(n):
        a1, a2, a3 = n + 3*i, n + 3*i + 1, n + 3*i + 2
        P[i,   a1] = 1; P[a1,  i] = 1
        P[i,   a2] = 2; P[a2, a3] = 1; P[a3, i] = 1
    return row_normalize(P)

def minor(A, F):
    """Remove rows/cols in F from A (fault-set removal)."""
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep


# ——————————————————————————————————————————————————————————————
# 2) PROBLEM SETUP (Alg 3, input)
# ——————————————————————————————————————————————————————————————
agents       = [1,2,3,4,5]      # V = [5]
N            = len(agents)      # N = 5
f            = 1                # resilience parameter
ε            = 0.05             # precision parameter
T            = 100              # number of iterations

# initial states x⁽⁰⁾
x0           = {1:0.10, 2:0.30, 3:0.35, 4:0.60, 5:0.55}
attacker_val = {2: (lambda k: 0.30)}

# adjacency for G₁
adj = {
    1:[3,4,5],
    2:[3,4],
    3:[1,2,4],
    4:[1,2,3,5],
    5:[1,4]
}

# build the real-to-real matrix A
A = np.zeros((N,N))
for u in agents:
    for v in adj[u]:
        A[u-1,v-1] = 1/len(adj[u])

honest_avg = np.mean([x0[u] for u in agents if u not in attacker_val])
print("Original A:\n", A, "\nHonest average =", honest_avg, "\n")


# ——————————————————————————————————————————————————————————————
# 3) ENUMERATE FAULT-SETS F with |F| ≤ f (Alg 3, line 3)
# ——————————————————————————————————————————————————————————————
F_list = sorted(
    [frozenset(c)
     for k in range(f+1)
     for c in itertools.combinations(agents, k)],
    key=lambda S: (len(S), sorted(S))
)
idx_of = {F:i for i,F in enumerate(F_list)}
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
    print(" A_sub =\n", A_sub, "\n")

    # build privacy-augmented
    P_pa = build_Ap(n_sub, A_sub)
    print(" P_pa =\n", np.round(P_pa,3), "\n")

    # stationary left eigenvector
    w, V = np.linalg.eig(P_pa.T)
    i1   = np.argmin(np.abs(w-1))
    v0   = np.real(V[:,i1]); v0 /= v0.sum()
    print(" v^0 =", np.round(v0,3))

    # closed-form private init
    x_sub  = np.array([x0[agents[i]] for i in surv])
    x_priv = np.zeros(4*n_sub)
    for j,u_idx in enumerate(surv):
        xj    = x_sub[j]
        slots = [ j,
                  n_sub+3*j,
                  n_sub+3*j+1,
                  n_sub+3*j+2 ]
        for ℓ in slots:
            x_priv[ℓ] = xj / (4 * n_sub * v0[ℓ])
    print("  v0·x_priv =", round(float(v0@x_priv),3),
          " target=", round(x_sub.mean(),3), "\n")

    P_list.append(P_pa)
    surv_idxs.append(surv)
    x_priv_list.append(x_priv)


# ——————————————————————————————————————————————————————————————
# 5) SIMULATION: USE P_pa, THEN CLAMP ALL 3 SLOTS OF AGENT 2 TO 0.3
# ——————————————————————————————————————————————————————————————
X_list = [np.zeros((4*len(s), T+1)) for s in surv_idxs]
for i in range(len(F_list)):
    X_list[i][:,0] = x_priv_list[i]

for k in range(T):
    for i,Pm in enumerate(P_list):
        n_sub = len(surv_idxs[i])
        # step
        X_list[i][:,k+1] = Pm @ X_list[i][:,k]

        # clamp agent 2's privacy-slots so collapse=0.3
        if 2 not in F_list[i]:
            j_real = surv_idxs[i].index(2-1)
            a1 = n_sub + 3*j_real
            a2 = a1+1
            a3 = a1+2
            X_list[i][a1, k+1] = 0.3
            X_list[i][a2, k+1] = 0.3
            X_list[i][a3, k+1] = 0.3

# ——————————————————————————————————————————————————————————————
# 6) COLLAPSE → candidate vectors c[u][k]
# ——————————————————————————————————————————————————————————————
c = {u:{} for u in agents}
for u in agents:
    for k in range(T+1):
        vec = []
        for i,F in enumerate(F_list):
            surv = surv_idxs[i]
            if u in attacker_val:
                # attacker always sends 0.3
                vec.append(attacker_val[u](k))
            elif u in F:
                # removed nodes stay at initial
                vec.append(x0[u])
            else:
                j = surv.index(u-1)
                vec.append( X_list[i][ j, k ] )
        c[u][k] = vec

# ——————————————————————————————————————————————————————————————
# 7) ALGORITHM 3: RESILIENT SELECTION & HISTORY
# ——————————————————————————————————————————————————————————————
x_hist = {u:[x0[u]] for u in agents}

for k in range(1, T+1):
    print(f"\n--- Round {k} ---")
    for u in agents:
        entries = ", ".join(f"{list(F_list[i])}:{c[u][k][i]:.3f}"
                            for i in range(len(F_list)))
        print(f" Agent {u}: c[{k}] = [{entries}]")

    for u in agents:
        if u in attacker_val:
            x_next = attacker_val[u](k)
        else:
            full = c[u][k][idx_of[frozenset()]]
            bads = [(i,abs(c[u][k][i]-full))
                    for i,F in enumerate(F_list)
                    if F and (u not in F) and abs(c[u][k][i]-full)>=ε]
            if len(bads)==1:
                x_next = c[u][k][bads[0][0]]
                print(f"  Agent {u}: outlier→ F={F_list[bads[0][0]]}")
            else:
                x_next = full
                print(f"  Agent {u}: keep full={full:.3f}")
        x_hist[u].append(x_next)

# ——————————————————————————————————————————————————————————————
# 8) PLOT
# ——————————————————————————————————————————————————————————————
X_resil = np.vstack([x_hist[u] for u in agents]).T

plt.figure(figsize=(8,4))
for i,u in enumerate(agents):
    plt.plot(X_resil[:,i], label=f'$x_{{{u}}}$')
plt.axhline(honest_avg, color='k', ls='--', lw=1,
            label=f'honest avg={honest_avg:.2f}')
plt.title('Alg 3: Private & Resilient Consensus on $G_1$')
plt.xlabel('Step'); plt.ylabel('State')
plt.legend(fontsize='small', ncol=3)
plt.grid(True); plt.tight_layout()
plt.show()
