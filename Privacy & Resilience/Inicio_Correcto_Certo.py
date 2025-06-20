import numpy as np
import matplotlib.pyplot as plt
import itertools

# ——————————————————————————————————————————————————————————————
# 1) HELPERS & PRIVACY-AUGMENTATION
# ——————————————————————————————————————————————————————————————
np.set_printoptions(precision=3, suppress=True)

def row_normalize(M):
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N: int, A: np.ndarray) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A  
    for i in range(N):

        inds = {
            0: i,          
            1: N + 3*i,   
            2: N + 3*i+1, 
            3: N + 3*i+2  
        }


        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1

        # ——————————————————————
        # b)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # c)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[3]] = 1
        Ap[inds[2], inds[3]] = 1
        Ap[inds[3], inds[0]] = 2
        '''
        # ——————————————————————
        # d)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        '''
        # ——————————————————————
        # e)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # f)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # g)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # h)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # i)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # j)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # k)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 2
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # l)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        '''
        # ——————————————————————
        # m)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
    return row_normalize(Ap)

def minor(A, F):
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep,keep)], keep

# ——————————————————————————————————————————————————————————————
# 2) PROBLEM SETUP
# ——————————————————————————————————————————————————————————————
agents       = [1,2,3,4,5]
N            = len(agents)
f            = 1
ε            = 0.05
T            = 60

x0           = {1:0.10, 2:0.30, 3:0.35, 4:0.60, 5:0.55}
attacker_val = {2: (lambda k: 0.30)}

adj = {
    1:[3,4,5],
    2:[3,4],
    3:[1,2,4],
    4:[1,2,3,5],
    5:[1,4]
}
'''
adj = {
    1:[2,3,4,5],
    2:[1,3,4,5],
    3:[1,2,4,5],
    4:[1,2,3],
    5:[1,2,3]
}
'''

# build A
A = np.zeros((N,N))
for u in agents:
    for v in adj[u]:
        A[u-1,v-1] = 1/len(adj[u])
print("Original A:\n", A, "\n")

honest_avg = np.mean([x0[u] for u in agents if u not in attacker_val])
print("Honest average =", honest_avg, "\n")

# ——————————————————————————————————————————————————————————————
# 3) ALL FAULT-SETS |F|≤f
# ——————————————————————————————————————————————————————————————
F_list = sorted(
    [frozenset(c)
     for k in range(f+1)
     for c in itertools.combinations(agents, k)],
    key=lambda S:(len(S), sorted(S))
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

    # force real-slots back to zero
    for j in range(n_sub):
        x_priv[j] = 0.0

    target  = x_sub.mean()
    current = v0 @ x_priv
    x_priv *= (target / current)

    print(" v0·x_priv =", float(v0@x_priv),
        " target=", target)

    P_list.append(P_pa)
    surv_idxs.append(surv)
    x_priv_list.append(x_priv)
    for i, x in enumerate(x_priv_list):
        vals = [round(v, 3) for v in x.tolist()]
        print(f"x_augmentato[{i}] = {vals}")

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
# 8) PLOT REAL-SLOT TRAJECTORIES
# ——————————————————————————————————————————————————————————————
i0 = idx_of[frozenset()]   # find the “no faults” case
X0 = X_list[i0]            # shape is (4*N, T+1)

plt.figure(figsize=(8,4))
for u in agents:
    real_traj = x_hist[u]    # the real slot of agent u
    plt.plot(real_traj, label=f'$x_{{{u}}}^{{(k)}}$')

plt.axhline(honest_avg, color='k', ls='--', lw=1,
            label=f'honest avg={honest_avg:.2f}')
plt.title('Alg 3: Private & Resilient Consensus on $G_1$')
plt.xlabel('Iteration $k$')
plt.ylabel('State')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
