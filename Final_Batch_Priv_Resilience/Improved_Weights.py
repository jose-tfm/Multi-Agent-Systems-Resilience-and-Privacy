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

np.set_printoptions(precision=3, suppress=True)

def row_normalize(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N: int, A: np.ndarray, optimized_weights: bool = False) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N,:N] = A
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

def step_subgraph(filters: dict[int,list[bool]], honest: list[int], T:int) -> int|None:
    for k in range(T+1):
        if all(filters[u][k] for u in honest):
            return k
    return None

def simulate_resilient_consensus(
    A: np.ndarray,
    x0_dict: dict[int,float],
    attacker_val: dict[int,callable],
    f: int,
    eps: float,
    T: int,
    optimized_weights: bool
):
    N = A.shape[0]
    agents = list(range(1,N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    # enumerate minors
    F_list = sorted(
        [frozenset(c) for k in range(f+1)
                      for c in itertools.combinations(agents,k)],
        key=lambda S:(len(S),sorted(S))
    )
    idx0 = F_list.index(frozenset())

    label = 'optW' if optimized_weights else 'base'
    P_dict   = []
    priv_dict= []
    eig_dict = []
    surv_idxs = []

    # build minors
    for F in F_list:
        surv = [i-1 for i in agents if i not in F]
        surv_idxs.append(surv)
        A_sub = row_normalize(A[np.ix_(surv,surv)].copy())
        n_sub = len(surv)

        Ppa = build_Ap(n_sub, A_sub, optimized_weights)
        P_dict.append(Ppa)

        # left eigenvector
        w,V = eig(Ppa.T)
        v = np.abs(np.real(V[:,np.argmin(np.abs(w-1))])); v/=v.sum()
        eig_dict.append(v)

        # private init
        x_sub = np.array([x0_dict[i+1] for i in surv])
        priv = np.zeros(4*n_sub)
        for j in range(n_sub):
            base = n_sub+3*j
            priv[base:base+3] = x_sub[j]/(4*n_sub)
        priv *= (x_sub.mean()/(v@priv))
        priv_dict.append(priv)

    # prep storage
    histories   = {label:{u:[None]*(T+1) for u in agents}}
    filters     = {label:{u:[False]*(T+1) for u in agents}}
    conv_rounds = {label:{u:None for u in agents}}
    X_store     = [None]*len(F_list)

    # simulate
    for k in range(T+1):
        for i,F in enumerate(F_list):
            surv = surv_idxs[i]; n_sub = len(surv)
            if k==0:
                X = np.zeros((4*n_sub,T+1))
                X[:,0] = priv_dict[i]
                X_store[i] = X
            else:
                X_prev = X_store[i]
                X = P_dict[i]@X_prev[:,k-1]
                # attacker injection
                for att,atk in attacker_val.items():
                    if att not in F:
                        j = surv.index(att-1)
                        X[n_sub+3*j:n_sub+3*j+3] = atk(k-1)
                X_store[i][:,k] = X

        # collapse & filter
        for u in agents:
            for i,F in enumerate(F_list):
                if k==0:
                    histories[label][u][0] = x0_dict[u]
                else:
                    if u in attacker_val:
                        val = attacker_val[u](k-1)
                    else:
                        surv0 = surv_idxs[idx0]
                        if (u-1) in surv0:
                            full = X_store[idx0][surv0.index(u-1),k]
                        else:
                            full = x0_dict[u]
                        outs=[]
                        for j,F2 in enumerate(F_list):
                            surv2 = surv_idxs[j]
                            if F2 and u not in F2 and (u-1) in surv2:
                                cand = X_store[j][surv2.index(u-1),k]
                                if abs(cand-full)>=eps:
                                    outs.append(cand)
                        if len(outs)==1:
                            filters[label][u][k]=True
                            val=outs[0]
                        else:
                            val=full
                    histories[label][u][k]=val
                    if conv_rounds[label][u] is None and abs(val-honest_avg)<eps:
                        conv_rounds[label][u]=k

    # full trajectories
    full_trajs = {label:{}}
    surv0 = surv_idxs[idx0]
    for u in agents:
        if (u-1) in surv0:
            full_trajs[label][u] = list(X_store[idx0][surv0.index(u-1),:])
        else:
            full_trajs[label][u] = [x0_dict[u]]*(T+1)

    return histories, filters, conv_rounds, full_trajs, honest_avg, priv_dict, eig_dict, F_list

def run_trials(
    N=11,p_edge=0.8,f=1,eps=0.08,T=100,seed0=4,trials=20
):
    records=[]
    matrix_rows=[]

    with ExcelWriter('OptW_Comparison.xlsx',engine='openpyxl') as writer:
        wb=writer.book
        ws_plots=wb.create_sheet('plots')

        for t in range(trials):
            np.random.seed(seed0+t); seed=seed0+t
            # build SC digraph
            while True:
                G=nx.gnp_random_graph(N,p_edge,seed=seed,directed=True)
                if nx.is_strongly_connected(G): break
                seed+=1
            for u,v in G.edges():
                G[u][v]['weight']=np.random.rand()
            A=row_normalize(nx.to_numpy_array(G,weight='weight'))

            # record full adjacency
            for i in range(N):
                row={'trial':t,'version':'full','row':i}
                for j in range(N):
                    row[f'col_{j}']=A[i,j]
                matrix_rows.append(row)

            x0={u:np.random.rand() for u in range(1,N+1)}
            attacker={2:lambda k:0.8}

            # base run
            h_b,f_b,c_b,ft_b,avg_b,priv_b,eig_b,F_list=simulate_resilient_consensus(
                A,x0,attacker,f,eps,T,optimized_weights=False
            )
            # optW run
            h_o,f_o,c_o,ft_o,avg_o,priv_o,eig_o,_=simulate_resilient_consensus(
                A,x0,attacker,f,eps,T,optimized_weights=True
            )

            honest=[u for u in h_b['base'] if u not in attacker]
            k_b=step_subgraph(f_b['base'],honest,T)
            k_o=step_subgraph(f_o['optW'],honest,T)
            records.append({'trial':t,'Base_k':k_b,'OptW_k':k_o})

            # write each minor's rows
            for idx,F in enumerate(F_list):
                surv=[i-1 for i in range(1,N+1) if i not in F]
                A_sub=row_normalize(A[np.ix_(surv,surv)])

                # base_minor
                for ii,u in enumerate(surv):
                    row={'trial':t,'version':'base_minor','faulty_set':tuple(sorted(F)),'row':u}
                    for jj,v in enumerate(surv):
                        row[f'col_{v}']=A_sub[ii,jj]
                    matrix_rows.append(row)
                # base_aug_init
                row={'trial':t,'version':'base_aug_init','faulty_set':tuple(sorted(F))}
                for j,val in enumerate(priv_b[idx]):
                    row[f'x_{j}']=val
                matrix_rows.append(row)
                # base_eig_vec
                row={'trial':t,'version':'base_eig_vec','faulty_set':tuple(sorted(F))}
                for j,val in enumerate(eig_b[idx]):
                    row[f'v_{j}']=val
                matrix_rows.append(row)

                # optW_minor
                for ii,u in enumerate(surv):
                    row={'trial':t,'version':'optW_minor','faulty_set':tuple(sorted(F)),'row':u}
                    for jj,v in enumerate(surv):
                        row[f'col_{v}']=A_sub[ii,jj]
                    matrix_rows.append(row)
                # optW_aug_init
                row={'trial':t,'version':'optW_aug_init','faulty_set':tuple(sorted(F))}
                for j,val in enumerate(priv_o[idx]):
                    row[f'x_{j}']=val
                matrix_rows.append(row)
                # optW_eig_vec
                row={'trial':t,'version':'optW_eig_vec','faulty_set':tuple(sorted(F))}
                for j,val in enumerate(eig_o[idx]):
                    row[f'v_{j}']=val
                matrix_rows.append(row)

            # plots
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))
            for u in honest:
                ax1.plot(h_b['base'][u],'--',label=f'x_{u}')
                ax2.plot(h_o['optW'][u],'-', label=f'x_{u}')
            for ax,title in zip((ax1,ax2),('Baseline','OptWeights')):
                ax.axhline(avg_b,ls=':',c='k')
                ax.set_xlabel('k'); ax.set_title(f'{title} Trial {t}')
            ax1.legend(ncol=2,fontsize='small')
            buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
            plt.close(fig); buf.seek(0)
            ws_plots.add_image(OpenPyXLImage(buf),f'A{2+t*20}')

        # summary
        df=DataFrame(records); df.to_excel(writer,'summary',index=False)

        # boxplot
        fig,ax=plt.subplots(figsize=(6,4))
        ax.boxplot([df['Base_k'],df['OptW_k']],labels=['Base','OptW'])
        ax.set_title('Detection Steps'); ax.set_ylabel('k')
        buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png')
        plt.close(fig); buf.seek(0)
        ws2=wb.create_sheet('boxplot'); ws2.add_image(OpenPyXLImage(buf),'A1')

        # all_matrices
        DataFrame(matrix_rows).to_excel(writer,'all_matrices',index=False)

    print("Saved 'OptW_Comparison.xlsx'")
    return DataFrame(records)

if __name__=='__main__':
    df=run_trials(trials=10)
    print(df.describe())
