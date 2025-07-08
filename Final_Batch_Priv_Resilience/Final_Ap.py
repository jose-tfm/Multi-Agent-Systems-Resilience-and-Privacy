import numpy as np
import itertools
import networkx as nx
from scipy.optimize import minimize
from numpy.linalg import eigvals, eig
import matplotlib.pyplot as plt
import io
import pandas as pd
from pandas import DataFrame, ExcelWriter
from openpyxl.drawing.image import Image as OpenPyXLImage

# For neat printing of small arrays
np.set_printoptions(precision=3, suppress=True)

def row_normalize(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N: int, A: np.ndarray, optimized: bool=False) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        b = N + 3*i
        if not optimized:
            Ap[i, b    ] = 1;   Ap[b,    i] = 1
            Ap[i, b+2  ] = 2;   Ap[b+2, b+1] = 1
            Ap[b+1, i]   = 1
        else:
            Ap[i, b    ] = 0.186; Ap[b,    i] = 1
            Ap[i, b+2  ] = 0.108; Ap[b+2, b+1] = 1
            Ap[b+1, i]   = 1
    return row_normalize(Ap)

def consensus_rate_P(flat: np.ndarray, size: int) -> float:
    P = row_normalize(flat.reshape((size, size)))
    w = np.abs(eigvals(P))
    return float(np.sort(w)[-2])

def simulate_resilient_consensus(
    A: np.ndarray,
    x0_dict: dict[int,float],
    attacker_val: dict[int,callable],
    f: int,
    eps: float,
    T: int,
    optimize_subgraphs: bool=False
):
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    # enumerate all fault‐sets up to size f
    F_list = sorted(
        [frozenset(c)
         for k in range(f+1)
         for c in itertools.combinations(agents, k)],
        key=lambda S: (len(S), sorted(S))
    )
    idx0 = F_list.index(frozenset())

    labels    = ['orig'] + (['opt'] if optimize_subgraphs else [])
    P_dict    = {lab: [] for lab in labels}
    priv_dict = {lab: [] for lab in labels}
    surv_idxs = []

    # build & optionally optimize each minor’s augmented matrix
    for F in F_list:
        surv_idxs.append([i-1 for i in agents if i not in F])
        surv = surv_idxs[-1]
        A_sub = row_normalize(A[np.ix_(surv, surv)])
        n_sub = len(surv)

        # build the unoptimized Augmented P
        P0 = build_Ap(n_sub, A_sub, optimized=False)
        versions = {'orig': P0}

        if optimize_subgraphs:
            sizeP = P0.shape[0]
            flat0 = P0.flatten()
            mask  = flat0 > 0
            bounds= [(0,0) if not m else (1e-2,1) for m in mask]
            res = minimize(
                lambda p: consensus_rate_P(p, sizeP),
                flat0, method='SLSQP', bounds=bounds,
                options={'ftol':1e-9,'maxiter':500}
            )
            P_opt = row_normalize(res.x.reshape((sizeP,sizeP)))
            versions['opt'] = P_opt

        # compute private‐init exactly as in your single‐P code
        for lab, Ppa in versions.items():
            w, V = eig(Ppa.T)
            idx = np.argmin(np.abs(w - 1))
            v0  = np.real(V[:, idx]); v0 /= v0.sum()

            x_sub  = np.array([x0_dict[i+1] for i in surv])
            target = x_sub.mean()
            sizeP  = Ppa.shape[0]

            x0_aug = np.zeros(sizeP)
            x0_aug[:n_sub] = x_sub
            for j in range(n_sub):
                for kslot in (0,1,2):
                    x0_aug[n_sub + 3*j + kslot] = x_sub[j]/3

            x0_aug *= target / (v0 @ x0_aug + 1e-12)

            P_dict[lab].append(Ppa)
            priv_dict[lab].append(x0_aug)

    # now run the time‐stepping & filter logic (identical to before)
    histories = {lab:{u:[None]*(T+1) for u in agents} for lab in labels}
    filters   = {lab:{u:[False]*(T+1) for u in agents} for lab in labels}
    conv_rnds = {lab:{u:None          for u in agents} for lab in labels}
    X_store   = {lab:[None]*len(F_list)           for lab in labels}

    for k in range(T+1):
        # step each augmented system
        for lab in labels:
            for i, F in enumerate(F_list):
                surv = surv_idxs[i]
                X0   = priv_dict[lab][i]
                rows = X0.size
                n_sub= rows // 4

                if k==0:
                    X = np.zeros((rows, T+1))
                    X[:,0] = X0
                    X_store[lab][i] = X
                else:
                    X = X_store[lab][i]
                    X[:,k] = P_dict[lab][i] @ X[:,k-1]
                    # attacker injection
                    for att,atk in attacker_val.items():
                        if att not in F:
                            j = surv.index(att-1)
                            st = n_sub + 3*j
                            X[st:st+3, k] = atk(k-1)

        # collapse & resilient filter
        for lab in labels:
            for u in agents:
                if k==0:
                    histories[lab][u][0] = x0_dict[u]
                    filters  [lab][u][0] = False
                else:
                    if u in attacker_val:
                        val, flag = attacker_val[u](k-1), False
                    else:
                        surv0 = surv_idxs[idx0]
                        if (u-1) in surv0:
                            j0   = surv0.index(u-1)
                            full = X_store[lab][idx0][j0, k]
                        else:
                            full = x0_dict[u]

                        outs = []
                        for j, F in enumerate(F_list):
                            survj = surv_idxs[j]
                            if F and u not in F and (u-1) in survj:
                                cj   = survj.index(u-1)
                                cand = X_store[lab][j][cj, k]
                                if abs(cand-full) >= eps:
                                    outs.append(cand)

                        if len(outs)==1:
                            val,flag = outs[0], True
                        else:
                            val,flag = full,    False

                    histories[lab][u][k] = val
                    filters  [lab][u][k] = flag
                    if conv_rnds[lab][u] is None and abs(val-honest_avg)<eps:
                        conv_rnds[lab][u] = k

    return histories, filters, conv_rnds, F_list, P_dict

def step_subgraph(filters, honest, T):
    for k in range(T+1):
        if all(filters[u][k] for u in honest):
            return k
    return None

def run_trials(
    N=11, p_edge=0.8, f=1,
    eps=0.08, T=100,
    seed0=4, trials=10
) -> pd.DataFrame:
    records     = []
    matrix_rows = []

    with ExcelWriter('PerMinor_Optimized.xlsx', engine='openpyxl') as writer:
        wb       = writer.book
        ws_plots = wb.create_sheet('plots')

        for t in range(trials):
            # generate random strongly-connected digraph
            np.random.seed(seed0+t)
            seed = seed0+t
            while True:
                G = nx.gnp_random_graph(N, p_edge, seed=seed, directed=True)
                if nx.is_strongly_connected(G): break
                seed += 1
            for u,v in G.edges(): G[u][v]['weight'] = np.random.rand()
            A = row_normalize(nx.to_numpy_array(G, weight='weight'))

            # 1) full N×N
            for i in range(N):
                rec = {'trial':t,'version':'full','row':i}
                for j in range(N):
                    rec[f'col_{j}'] = A[i,j]
                matrix_rows.append(rec)

            # 2) run simulations
            x0_dict = {u:np.random.rand() for u in range(1,N+1)}
            attacker={2: lambda k:0.8}

            ho, fo, co, F_list, Po_dict = simulate_resilient_consensus(
                A, x0_dict, attacker, f, eps, T, optimize_subgraphs=False
            )
            hp, fp, cp, _,     Pp_dict = simulate_resilient_consensus(
                A, x0_dict, attacker, f, eps, T, optimize_subgraphs=True
            )
            honest = [u for u in ho['orig'] if u not in attacker]

            # extract the lists
            orig_P_list = Po_dict['orig']
            opt_P_list  = Pp_dict['opt']

            # 3) write each augmented 4n×4n
            for lab, P_list in (('orig_aug', orig_P_list),
                                ('opt_aug',  opt_P_list)):
                for i, F in enumerate(F_list):
                    Ppa = P_list[i]
                    size= Ppa.shape[0]
                    for r in range(size):
                        rec = {
                          'trial':t,
                          'version':lab,
                          'faulty_set':tuple(sorted(F)),
                          'row':r
                        }
                        for c in range(size):
                            rec[f'col_{c}'] = Ppa[r,c]
                        matrix_rows.append(rec)

            # 4) measure detection and plot
            ko = step_subgraph(fo['orig'], honest, T)
            kp = step_subgraph(fp['opt'],   honest, T)
            records.append({'trial':t,'Orig_k':ko,'Opt_k':kp})

            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
            avg=np.mean([x0_dict[u] for u in honest])
            for u in honest:
                ax1.plot(ho['orig'][u],'--',label=f'x_{u}')
                ax2.plot(hp['opt'][u],  '-',label=f'x_{u}')
            for ax,title in zip((ax1,ax2),('Orig','Opt')):
                ax.axhline(avg,ls=':',c='k')
                ax.set_title(f'{title} Trial {t}'); ax.set_xlabel('k')
            ax1.legend(ncol=2,fontsize='small')
            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
            plt.close(fig); buf.seek(0)
            ws_plots.add_image(OpenPyXLImage(buf),f'A{2+t*20}')

        # summary sheet
        df = DataFrame(records)
        df.to_excel(writer, sheet_name='summary', index=False)

        # boxplot sheet
        orig_vals = df['Orig_k'].dropna().tolist()
        opt_vals  = df['Opt_k'].dropna().tolist()
        fig,ax=plt.subplots(figsize=(6,4))
        ax.boxplot([orig_vals,opt_vals],labels=['Orig','Opt'])
        ax.set_ylabel('Detection k')
        buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
        plt.close(fig); buf.seek(0)
        ws2=wb.create_sheet('boxplot')
        ws2.add_image(OpenPyXLImage(buf),'A1')

        # all_matrices
        mat_df = DataFrame(matrix_rows)
        mat_df.to_excel(writer, sheet_name='all_matrices', index=False)

    print("Saved 'PerMinor_Optimized.xlsx' with full, orig_aug & opt_aug.")
    return DataFrame(records)

if __name__=='__main__':
    df = run_trials(trials=1)
    print(df.describe())
