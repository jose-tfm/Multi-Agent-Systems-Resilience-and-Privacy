import numpy as np
import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import io
from openpyxl.drawing.image import Image as OpenPyXLImage
from pandas import ExcelWriter

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
    return A[np.ix_(keep, keep)], keep


def simulate_resilient_consensus(A, x0_dict, attacker_val, f, epsilon, T):
    """
    Run private & resilient consensus on adjacency matrix A.
    Returns:
      - x_hist: dict of real-slot trajectories
      - conv_rounds: dict of convergence rounds per agent
      - honest_avg: float
      - x_priv_no_fault: np.ndarray of private init (no-fault case)
      - lambda2_no_fault: second-largest eigenvalue magnitude of P_pa (no-fault)
    """
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    # Enumerate fault sets up to f
    F_list = sorted(
        [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
        key=lambda S: (len(S), sorted(S))
    )
    idx_of = {F: i for i, F in enumerate(F_list)}

    # Private init: build P_list, record private vectors
    P_list, surv_idxs, x_priv_list = [], [], []
    for F in F_list:
        A_sub, surv = minor(A, [u-1 for u in F])
        A_sub = row_normalize(A_sub)
        n_sub = len(surv)
        P_pa = build_Ap(n_sub, A_sub)
        # compute closed-form private init
        w, V = np.linalg.eig(P_pa.T)
        idx1 = np.argmin(np.abs(w - 1))
        v0 = np.real(V[:, idx1]); v0 /= v0.sum()
        x_sub = np.array([x0_dict[u] for u in [s+1 for s in surv]])
        x_priv = np.zeros(4 * n_sub)
        for j in range(n_sub):
            for slot in (j, n_sub+3*j, n_sub+3*j+1, n_sub+3*j+2):
                x_priv[slot] = x_sub[j] / (4 * n_sub * v0[slot])
        x_priv[:n_sub] = 0
        x_priv *= (x_sub.mean() / (v0 @ x_priv))
        P_list.append(P_pa)
        surv_idxs.append(surv)
        x_priv_list.append(x_priv)

    # extract no-fault augmented data
    no_fault_idx = idx_of[frozenset()]
    x_priv_no_fault = x_priv_list[no_fault_idx]
    P0 = P_list[no_fault_idx]
    eigs = np.abs(np.linalg.eigvals(P0))
    eigs.sort()
    lambda2_no_fault = eigs[-2]

    # initialize histories
    x_hist = {u: [None] * (T+1) for u in agents}
    conv_rounds = {u: None for u in agents}
    X_vals = [None] * len(F_list)

    # simulation + resilient selection
    for k in range(T+1):
        # step augmented states
        for i, F in enumerate(F_list):
            n_sub = len(surv_idxs[i])
            if k == 0:
                X = np.zeros((4*n_sub, T+1))
                X[:, 0] = x_priv_list[i]
                X_vals[i] = X
            else:
                X = X_vals[i]
                X[:, k] = P_list[i] @ X[:, k-1]
                # clamp attacker slots
                for att, atk in attacker_val.items():
                    if att not in F:
                        j = surv_idxs[i].index(att-1)
                        X[n_sub+3*j:n_sub+3*j+3, k] = atk(k-1)
        # collapse & resilient selection
        for u in agents:
            if k == 0:
                x_hist[u][0] = x0_dict[u]
            else:
                if u in attacker_val:
                    val = attacker_val[u](k-1)
                else:
                    # full value from no-fault
                    surv0 = surv_idxs[no_fault_idx]
                    if (u-1) in surv0:
                        j0 = surv0.index(u-1)
                        full = X_vals[no_fault_idx][j0, k]
                    else:
                        full = x0_dict[u]
                    # detect single outlier
                    outliers = []
                    for i, FF in enumerate(F_list):
                        if FF and u not in FF:
                            surv_i = surv_idxs[i]
                            if (u-1) in surv_i:
                                ji = surv_i.index(u-1)
                                cand = X_vals[i][ji, k]
                                if abs(cand - full) >= epsilon:
                                    outliers.append(cand)
                    val = outliers[0] if len(outliers) == 1 else full
                x_hist[u][k] = val
                if conv_rounds[u] is None and abs(val - honest_avg) < epsilon:
                    conv_rounds[u] = k

    return x_hist, conv_rounds, honest_avg, x_priv_no_fault, lambda2_no_fault


def run_50_random_trials(N=11, p_edge=0.8, f=1, epsilon=0.05, T=200, seed0=4):
    summary_records = []
    with ExcelWriter('50_trials.xlsx', engine='openpyxl') as writer:
        wb = writer.book
        ws_plots = wb.create_sheet('plots')

        for t in range(50):
            seed = seed0 + t
            np.random.seed(seed)
            # random connected ER
            while True:
                D = nx.gnp_random_graph(N, p_edge, seed=seed, directed=True)
                if nx.is_strongly_connected(D):
                    G = D
                    break
                seed += 1
            for u, v in G.edges(): G[u][v]['weight'] = np.random.rand()
            raw_A = nx.to_numpy_array(G, nodelist=range(N), weight='weight')
            A = row_normalize(raw_A)
            x0_dict = {u: np.random.rand() for u in range(1, N+1)}
            attacker_val = {2: (lambda k: 0.8)}

            x_hist, conv_rounds, honest_avg, x_priv, lambda2 = \
                simulate_resilient_consensus(A, x0_dict, attacker_val, f, epsilon, T)

            # summary for this trial: include lambda2, conv_steps, x0 list, x_priv vector
            # convergence of the slowest agent (last to converge)
            conv_time = max([conv_rounds[u] for u in conv_rounds if conv_rounds[u] is not None])
            rec = {
                'trial': t,
                'lambda2': lambda2,
                'conv_steps': conv_time,
                'x0': [round(x0_dict[u], 3) for u in sorted(x0_dict.keys())],
                'x_priv': [round(v, 3) for v in x_priv]
            }
            summary_records.append(rec)

            # plot trajectories
            fig, ax = plt.subplots(figsize=(8, 4))
            for u, traj in x_hist.items(): ax.plot(traj, label=f'x_{u}')
            ax.axhline(honest_avg, ls='--', color='k', label='honest_avg')
            ax.set_title(f'Trial {t} Trajectories')
            ax.set_xlabel('k'); ax.set_ylabel('value')
            ax.legend(ncol=2, fontsize='small'); ax.grid(True)
            buf = io.BytesIO()
            fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig)
            buf.seek(0)
            img = OpenPyXLImage(buf)
            ws_plots.add_image(img, f'A{2 + t*20}')

        # write summary table at top of 'convergence'
        df_sum = pd.DataFrame(summary_records)
        df_sum.to_excel(writer, sheet_name='convergence', index=False)

    print("Saved summary and all plots to '50_trials.xlsx'")
    return pd.DataFrame(summary_records)

if __name__ == '__main__':
    run_50_random_trials()
