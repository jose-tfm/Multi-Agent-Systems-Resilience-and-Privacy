import numpy as np
import itertools
import networkx as nx
from scipy.optimize import minimize
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import io
import pandas as pd
from pandas import DataFrame, ExcelWriter
from openpyxl.drawing.image import Image as OpenPyXLImage

np.set_printoptions(precision=3, suppress=True)

def row_normalize(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N: int, A: np.ndarray, optimized_weights: bool = False) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        base = N + 3*i
        if not optimized_weights:
            Ap[i,    base  ] = 1
            Ap[base, i     ] = 1
            Ap[i,    base+2] = 2
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
        else:
            Ap[i,    base  ] = 0.186
            Ap[base, i     ] = 1
            Ap[i,    base+2] = 0.108
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
    return row_normalize(Ap)

def step_subgraph(filters: dict[int, list[bool]], honest: list[int], T: int) -> int | None:
    for k in range(T+1):
        if all(filters[u][k] for u in honest):
            return k
    return None

def validate_minor(A_sub: np.ndarray) -> bool:
    M = A_sub.copy()
    np.fill_diagonal(M, 0)
    if np.any(M.sum(axis=1)==0):
        return False
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    return nx.is_strongly_connected(G)

def simulate_resilient_consensus(
    A, x0, attacker, f, eps, T
):
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0[u] for u in agents if u not in attacker])

    F_list = sorted(
        [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
        key=lambda S:(len(S), sorted(S))
    )
    idx0 = F_list.index(frozenset())

    P_dict    = {'orig': [], 'opt': []}
    priv_dict = {'orig': [], 'opt': []}
    eig_dict  = {'orig': [], 'opt': []}
    surv_idxs = []

    for F in F_list:
        surv = [i-1 for i in agents if i not in F]
        surv_idxs.append(surv)
        A_sub = row_normalize(A[np.ix_(surv, surv)].copy())
        assert validate_minor(A_sub), f"Minor {F} not SC"
        n_sub = len(surv)

        for lab, opt in (('orig', False), ('opt', True)):
            Ppa = build_Ap(n_sub, A_sub, optimized_weights=opt)
            P_dict[lab].append(Ppa)

            w, V = np.linalg.eig(Ppa.T)
            v = np.abs(np.real(V[:, np.argmin(np.abs(w-1))])); v /= v.sum()
            eig_dict[lab].append(v)

            x_sub = np.array([x0[i+1] for i in surv])
            priv = np.zeros(4*n_sub)
            for j in range(n_sub):
                base = n_sub + 3*j
                priv[base:base+3] = x_sub[j] / (4 * n_sub)
            priv *= (x_sub.mean() / (v @ priv))
            priv_dict[lab].append(priv)

    histories   = {lab:{u:[None]*(T+1) for u in agents} for lab in P_dict}
    filters     = {lab:{u:[False]*(T+1) for u in agents} for lab in P_dict}
    conv_rounds = {lab:{u:None for u in agents}       for lab in P_dict}
    full_trajs  = {lab:{} for lab in P_dict}
    X_store     = {
        lab: [np.zeros((4*len(surv_idxs[i]), T+1)) for i in range(len(F_list))]
        for lab in P_dict
    }

    for k in range(T+1):
        for lab in P_dict:
            for i, F in enumerate(F_list):
                surv = surv_idxs[i]; n_sub = len(surv)
                X = X_store[lab][i]
                if k == 0:
                    X[:,0] = priv_dict[lab][i]
                else:
                    X[:,k] = P_dict[lab][i] @ X[:,k-1]
                    for att, atk in attacker.items():
                        if att not in F:
                            j = surv.index(att-1)
                            X[n_sub+3*j:n_sub+3*j+3, k] = atk(k-1)
        surv0 = surv_idxs[idx0]
        for lab in P_dict:
            for u in agents:
                if k == 0:
                    histories[lab][u][0] = x0[u]
                else:
                    if u in attacker:
                        val = attacker[u](k-1)
                    else:
                        full = (X_store[lab][idx0][surv0.index(u-1), k]
                                if (u-1) in surv0 else x0[u])
                        outs = []
                        for j, F in enumerate(F_list):
                            survj = surv_idxs[j]
                            if F and u not in F and (u-1) in survj:
                                cand = X_store[lab][j][survj.index(u-1), k]
                                if abs(cand-full) >= eps:
                                    outs.append(cand)
                        if len(outs) == 1:
                            filters[lab][u][k] = True
                            val = outs[0]
                        else:
                            val = full
                    histories[lab][u][k] = val
                    if conv_rounds[lab][u] is None and abs(val-honest_avg) < eps:
                        conv_rounds[lab][u] = k

    for lab in P_dict:
        for u in agents:
            if (u-1) in surv0:
                full_trajs[lab][u] = list(X_store[lab][idx0][surv0.index(u-1), :])
            else:
                full_trajs[lab][u] = [x0[u]]*(T+1)

    return histories, filters, conv_rounds, full_trajs, honest_avg, priv_dict, eig_dict, F_list

def run_trials(
    N=11, p_edge=0.8, f=1,
    eps=0.08, T=100,
    seed0=4, trials=20
) -> DataFrame:
    records = []
    matrix_rows = []

    with ExcelWriter('Full_Ap.xlsx', engine='openpyxl') as writer:
        wb = writer.book
        ws_plots = wb.create_sheet('plots')

        for t in range(trials):
            np.random.seed(seed0 + t)
            seed = seed0 + t
            while True:
                G = nx.gnp_random_graph(N, p_edge, seed=seed, directed=True)
                if nx.is_strongly_connected(G):
                    break
                seed += 1
            for u,v in G.edges():
                G[u][v]['weight'] = np.random.rand()
            A = row_normalize(nx.to_numpy_array(G, weight='weight'))

            for i in range(N):
                row = {'trial':t,'version':'full','row':i}
                for j in range(N):
                    row[f'col_{j}'] = A[i,j]
                matrix_rows.append(row)

            x0 = {u:np.random.rand() for u in range(1,N+1)}
            attacker = {2: lambda k:0.8}

            (hist_o, filt_o, conv_o, _, avg,
             priv_o, eig_o, F_list) = simulate_resilient_consensus(
                A, x0, attacker, f, eps, T
            )
            (hist_p, filt_p, conv_p, _,
             _, priv_p, eig_p, _) = simulate_resilient_consensus(
                A, x0, attacker, f, eps, T
            )

            honest = [u for u in hist_o['orig'] if u not in attacker]
            k_o = step_subgraph(filt_o['orig'], honest, T)
            k_p = step_subgraph(filt_p['opt'],  honest, T)
            records.append({'trial':t,
                            'Orig_subgraph_k':k_o,
                            'Opt_subgraph_k':k_p})

            for idx, F in enumerate(F_list):
                surv = [i-1 for i in range(1,N+1) if i not in F]
                A_sub = row_normalize(A[np.ix_(surv,surv)])

                for ii,u in enumerate(surv):
                    row = {'trial':t,'version':'orig_minor',
                           'faulty_set':tuple(sorted(F)),'row':u}
                    for jj,v in enumerate(surv):
                        row[f'col_{v}'] = A_sub[ii,jj]
                    matrix_rows.append(row)
                for ii,u in enumerate(surv):
                    row = {'trial':t,'version':'opt_minor',
                           'faulty_set':tuple(sorted(F)),'row':u}
                    for jj,v in enumerate(surv):
                        row[f'col_{v}'] = A_sub[ii,jj]
                    matrix_rows.append(row)

                row = {'trial':t,'version':'orig_aug_init',
                       'faulty_set':tuple(sorted(F))}
                for j,val in enumerate(priv_o['orig'][idx]):
                    row[f'x_{j}'] = val
                matrix_rows.append(row)

                row = {'trial':t,'version':'opt_aug_init',
                       'faulty_set':tuple(sorted(F))}
                for j,val in enumerate(priv_p['opt'][idx]):
                    row[f'x_{j}'] = val
                matrix_rows.append(row)

                row = {'trial':t,'version':'orig_eig_vec',
                       'faulty_set':tuple(sorted(F))}
                for j,val in enumerate(eig_o['orig'][idx]):
                    row[f'v_{j}'] = val
                matrix_rows.append(row)

                row = {'trial':t,'version':'opt_eig_vec',
                       'faulty_set':tuple(sorted(F))}
                for j,val in enumerate(eig_p['opt'][idx]):
                    row[f'v_{j}'] = val
                matrix_rows.append(row)

            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))
            for u in honest:
                ax1.plot(hist_o['orig'][u],'--',label=f'x_{u}')
                ax2.plot(hist_p['opt'][u],'-', label=f'x_{u}')
            for ax,title in zip((ax1,ax2),('Original','Optimized')):
                ax.axhline(avg,ls=':',c='k')
                ax.set_xlabel('k'); ax.set_title(f'{title} Trial {t}')
            ax1.legend(ncol=2,fontsize='small')
            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
            plt.close(fig); buf.seek(0)
            ws_plots.add_image(OpenPyXLImage(buf),f'A{2+t*20}')

        # summary
        df_summary = DataFrame(records)
        df_summary.to_excel(writer, sheet_name='summary', index=False)

        # boxplot
        fig, ax = plt.subplots(figsize=(6,4))
        base_vals = df_summary['Orig_subgraph_k'].dropna().tolist()
        opt_vals  = df_summary['Opt_subgraph_k'].dropna().tolist()
        ax.boxplot([base_vals, opt_vals], labels=['Orig','Opt'])
        ax.set_title('Subgraph Detection Steps'); ax.set_ylabel('k')
        buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
        plt.close(fig); buf.seek(0)
        ws2 = wb.create_sheet('boxplot')
        ws2.add_image(OpenPyXLImage(buf),'A1')

        # all_matrices
        DataFrame(matrix_rows).to_excel(writer, sheet_name='all_matrices', index=False)

    print("Saved 'Full_Ap.xlsx'")
    return df_summary

if __name__=='__main__':
    df = run_trials(trials=1)
    print(df.describe())
