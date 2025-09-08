import numpy as np
import itertools
import networkx as nx
from scipy.optimize import minimize
from numpy.linalg import eigvals, eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
from pandas import DataFrame, ExcelWriter
from openpyxl.drawing.image import Image as OpenPyXLImage
from multiprocessing import Pool, cpu_count
from functools import partial

np.set_printoptions(precision=3, suppress=True)

# Create a random network ER, BA, WS
def create_network(N, model='ER', p_edge=0.8, seed=None):
    """Create a strongly connected network with edge addition if needed"""
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if seed is not None:
            current_seed = seed + attempt
        else:
            current_seed = None
            
        if model == 'ER':
            G = nx.gnp_random_graph(N, p_edge, seed=current_seed, directed=True)
        elif model == 'BA':
            # Calculate m to get similar density to ER
            target_edges = int(N * (N-1) * p_edge)
            m = max(10, min(N-1, target_edges // (N-3))) if N > 3 else 2
            G = nx.barabasi_albert_graph(N, m=m, seed=current_seed)
            G = G.to_directed()
        elif model == 'WS': 
            # Calculate k to get similar density to ER
            target_edges = int(N * (N-1) * p_edge)
            k = max(4, min(N-1, 2 * target_edges // N))
            k = k if k % 2 == 0 else k + 1  # k must be even
            G = nx.watts_strogatz_graph(N, k=k, p=0.1, seed=current_seed)
            G = G.to_directed()
        else: 
            raise ValueError("Unknown model type. Use 'ER', 'BA', or 'WS'.")        

        # Make strongly connected by adding minimal edges
        if not nx.is_strongly_connected(G):
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(G))
            
            if len(sccs) > 1:
                # Connect components in a cycle to ensure strong connectivity
                # This adds minimal edges while preserving the network structure
                for i in range(len(sccs)):
                    current_scc = list(sccs[i])
                    next_scc = list(sccs[(i + 1) % len(sccs)])
                    
                    # Add edge from current component to next component
                    source_node = current_scc[0]  # Take first node from current SCC
                    target_node = next_scc[0]     # Take first node from next SCC
                    
                    if not G.has_edge(source_node, target_node):
                        G.add_edge(source_node, target_node)
        
        # Verify it's now strongly connected
        if nx.is_strongly_connected(G):
            return G
    
    # If we still can't create a connected graph, create a simple cycle
    print(f"Warning: Creating fallback cycle graph for {model} model")
    G = nx.cycle_graph(N, create_using=nx.DiGraph)
    
    # Add some random edges to match the desired density
    import random
    if seed is not None:
        random.seed(seed)
    
    target_edges = int(N * (N-1) * p_edge)
    current_edges = G.number_of_edges()
    
    while current_edges < target_edges:
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            current_edges += 1
    
    return G

def row_normalize(M: np.ndarray) -> np.ndarray:
    """Vectorized row normalization"""
    row_sums = M.sum(axis=1, keepdims=True)
    # Use np.divide with where parameter for better numerical stability
    return np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums!=0)

def build_Ap(N: int, A: np.ndarray, optimized: bool = False) -> np.ndarray:
    """Vectorized augmented matrix construction"""
    size = 4 * N
    Ap = np.zeros((size, size), dtype=np.float32)  # Use float32 for memory efficiency
    Ap[:N, :N] = A
    
    # Vectorized construction of augmented edges
    base_indices = N + 3 * np.arange(N)
    
    if not optimized:
        # Original weights
        weights = np.array([1, 1, 2, 1, 1])
        connections = np.array([
            [0, 1], [1, 0], [0, 3], [3, 2], [2, 0]
        ])
    else:
        # Optimized weights
        weights = np.array([0.186, 1, 0.108, 1, 1])
        connections = np.array([
            [0, 1], [1, 0], [0, 3], [3, 2], [2, 0]
        ])
    
    # Vectorized assignment
    for i in range(N):
        base = base_indices[i]
        for j, (src_offset, dst_offset) in enumerate(connections):
            if src_offset == 0:
                src_idx = i
            else:
                src_idx = base + src_offset - 1
            
            if dst_offset == 0:
                dst_idx = i
            else:
                dst_idx = base + dst_offset - 1
            
            Ap[src_idx, dst_idx] = weights[j]
    
    return row_normalize(Ap)

def first_full_switch(full: list[float], hist: list[float], eps: float) -> int:
    T = len(full) - 1
    for k in range(T+1):
        if all(abs(hist[t] - full[t]) < eps for t in range(k, T+1)):
            return k
    return T

def validate_minor(A_sub: np.ndarray) -> bool:
    """Validate if a minor subgraph is strongly connected"""
    M = A_sub.copy()
    np.fill_diagonal(M, 0)
    if np.any(M.sum(axis=1) == 0):
        return False
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    return nx.is_strongly_connected(G)

def consensus_rate(A_flat: np.ndarray, n: int, mask: np.ndarray) -> float:
    """Vectorized consensus rate calculation"""
    A = A_flat.reshape((n, n))
    # Vectorized row normalization
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums!=0)
    
    P = build_Ap(n, A, optimized=True)
    w = np.abs(eigvals(P))
    return float(np.partition(w, -2)[-2])  # Faster than sort for finding second largest

def minor(A: np.ndarray, F: list[int]) -> tuple[np.ndarray, list[int]]:
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep

def simulate_resilient_consensus(
    A: np.ndarray,
    x0_dict: dict[int, float],
    attacker_val: dict[int, callable],
    f: int,
    eps: float,
    T: int,
    optimize_subgraphs: bool = False
) -> tuple[
    dict[str, dict[int, list[float]]],  # histories
    dict[str, dict[int, list[bool]]],   # filter flags
    dict[str, dict[int, int | None]],   # conv_rounds
    dict[str, dict[int, list[float]]],  # full trajectories
    float,                              # honest average
    dict[str, list[np.ndarray]],        # priv_dict
    dict[str, list[np.ndarray]]         # eig_dict
]:
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    F_list = sorted(
        [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
        key=lambda S: (len(S), sorted(S))
    )
    idx0 = F_list.index(frozenset())

    labels    = ['orig'] + (['opt'] if optimize_subgraphs else [])
    P_dict    = {lab: [] for lab in labels}
    priv_dict = {lab: [] for lab in labels}
    eig_dict  = {lab: [] for lab in labels}
    surv_idxs = []

    for F in F_list:
        surv   = [i-1 for i in agents if i not in F]
        surv_idxs.append(surv)
        A_sub  = row_normalize(A[np.ix_(surv, surv)].copy())
        assert validate_minor(A_sub), f"Minor {F} is not strongly connected"
        n_sub  = len(surv)

        versions = {'orig': A_sub}
        if optimize_subgraphs:
            p0   = A_sub.flatten()
            mask = (p0 > 0).astype(float)
            bounds = [(0,0) if m==0 else (0,1) for m in mask]
            res = minimize(lambda p: consensus_rate(p, n_sub, mask),
                           p0, method='SLSQP', bounds=bounds,
                           options={'ftol':1e-9,'maxiter':500})
            A_opt = row_normalize(res.x.reshape((n_sub, n_sub)))
            assert validate_minor(A_opt), "Optimized minor invalid"
            versions['opt'] = A_opt

        for lab, Asub in versions.items():
            Ppa = build_Ap(n_sub, Asub, optimized=(lab=='opt'))
            P_dict[lab].append(Ppa)

            w, V = eig(Ppa.T)
            v = np.abs(np.real(V[:, np.argmin(np.abs(w-1))]))
            v /= v.sum()
            eig_dict[lab].append(v)

            # Vectorized private initialization
            x_sub = np.array([x0_dict[i+1] for i in surv], dtype=np.float32)
            
            # Use vectorized operations for alpha, beta, gamma
            alpha = np.ones(n_sub, dtype=np.float32)
            beta = np.ones(n_sub, dtype=np.float32)
            gamma = np.ones(n_sub, dtype=np.float32)

            # Vectorized private value computation
            s = alpha + beta + gamma
            coeff = 4 * x_sub / s
            
            priv = np.zeros(4 * n_sub, dtype=np.float32)
            base_indices = n_sub + 3 * np.arange(n_sub)
            
            # Vectorized assignment
            priv[base_indices] = coeff * alpha
            priv[base_indices + 1] = coeff * beta
            priv[base_indices + 2] = coeff * gamma

            target  = x_sub.mean()
            current = v @ priv
            priv   *= (target / current)

            priv_dict[lab].append(priv)

    # Pre-allocate arrays for better memory performance
    histories   = {lab:{u:np.zeros(T+1, dtype=np.float32) for u in agents} for lab in labels}
    filters     = {lab:{u:np.zeros(T+1, dtype=bool) for u in agents} for lab in labels}
    conv_rounds = {lab:{u:None for u in agents} for lab in labels}
    X_store     = {lab:[None]*len(F_list) for lab in labels}

    # Vectorized simulation loop
    for k in range(T+1):
        for lab in labels:
            for i, Fset in enumerate(F_list):
                surv  = surv_idxs[i]; n_sub = len(surv)
                if k == 0:
                    X = np.zeros((4*n_sub, T+1), dtype=np.float32)
                    X[:,0] = priv_dict[lab][i]
                    X_store[lab][i] = X
                else:
                    X = X_store[lab][i]
                    # Vectorized matrix multiplication
                    X[:,k] = P_dict[lab][i] @ X[:,k-1]
                    
                    # Vectorized attacker updates
                    for att, atk in attacker_val.items():
                        if att not in Fset:
                            j = surv.index(att-1)
                            attack_val = atk(k-1)
                            X[n_sub+3*j:n_sub+3*j+3, k] = attack_val

        # Vectorized convergence detection
        for lab in labels:
            for u in agents:
                if k == 0:
                    histories[lab][u][0] = x0_dict[u]
                else:
                    if u in attacker_val:
                        val = attacker_val[u](k-1)
                    else:
                        surv0 = surv_idxs[idx0]
                        full = (
                            X_store[lab][idx0][surv0.index(u-1), k]
                            if (u-1) in surv0 else x0_dict[u]
                        )
                        
                        # Vectorized outlier detection
                        outs = []
                        for j, Fset in enumerate(F_list):
                            surv_j = surv_idxs[j]
                            if Fset and u not in Fset and (u-1) in surv_j:
                                cand = X_store[lab][j][surv_j.index(u-1), k]
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

    full_trajs = {lab:{} for lab in labels}
    surv0 = surv_idxs[idx0]
    for lab in labels:
        X0 = X_store[lab][idx0]
        for u in agents:
            if (u-1) in surv0:
                full_trajs[lab][u] = list(X0[surv0.index(u-1), :])
            else:
                full_trajs[lab][u] = [x0_dict[u]]*(T+1)

    return histories, filters, conv_rounds, full_trajs, honest_avg, priv_dict, eig_dict

def step_subgraph(filter_flags: dict[int,list[bool]], honest: list[int], T:int) -> int | None:
    """Vectorized subgraph detection"""
    # Convert to numpy array for vectorized operations
    honest_filters = np.array([filter_flags[u] for u in honest], dtype=bool)
    # Find first column where all honest agents are filtered
    all_filtered = np.all(honest_filters, axis=0)
    indices = np.where(all_filtered)[0]
    return int(indices[0]) if len(indices) > 0 else None

def process_single_trial(params):
    """Optimized single trial processing with vectorization"""
    t, N, p_edge, f, eps, T, seed0, model = params
    np.random.seed(seed0 + t)
    seed = seed0 + t
    
    # Generate network (now guaranteed to be strongly connected)
    G = create_network(N, model=model, p_edge=p_edge, seed=seed)
    
    # Vectorized weight assignment
    edges = list(G.edges())
    if edges:
        weights = np.random.rand(len(edges))
        for i, (u, v) in enumerate(edges):
            G[u][v]['weight'] = weights[i]
    
    A = row_normalize(nx.to_numpy_array(G, weight='weight').astype(np.float32))
    
    # Initial conditions and attacker
    x0 = {u: np.random.rand() for u in range(1, N+1)}
    attacker = {2: lambda k: 0.8}
    
    # Simulate both cases with vectorized operations
    hist_o, filt_o, conv_o, _, avg, priv_o, eig_o = simulate_resilient_consensus(
        A, x0, attacker, f, eps, T, optimize_subgraphs=False
    )
    hist_p, filt_p, conv_p, _, _, priv_p, eig_p = simulate_resilient_consensus(
        A, x0, attacker, f, eps, T, optimize_subgraphs=True
    )
    honest = [u for u in hist_o['orig'] if u not in attacker]
    
    # Extract subgraph matrices for saving - COMMENTED OUT
    # agents = list(range(1, N+1))
    # F_list = sorted(
    #     [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
    #     key=lambda S: (len(S), sorted(S))
    # )
    
    # subgraph_matrices = {}
    # for i, F in enumerate(F_list):
    #     surv = [j-1 for j in agents if j not in F]
    #     if len(surv) > 0:
    #         A_sub = row_normalize(A[np.ix_(surv, surv)].copy())
    #         subgraph_matrices[f"F_{i}_{str(sorted(F))}"] = {
    #             'matrix': A_sub,
    #             'survivors': surv,
    #             'removed': sorted(F)
    #         }
    
    # Vectorized detection rounds
    k_o = step_subgraph(filt_o['orig'], honest, T)
    k_p = step_subgraph(filt_p['opt'], honest, T)
    
    # Enhanced publication-quality visualization with cleaner design - larger size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # Define publication-quality colors (colorblind-friendly palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    
    # Select subset of agents for cleaner visualization (show every 2nd or 3rd agent)
    max_agents_to_show = min(8, len(honest))  # Limit to 8 agents for clarity
    agent_indices = np.linspace(0, len(honest)-1, max_agents_to_show, dtype=int)
    selected_agents = [honest[i] for i in agent_indices]
    
    # Enhanced plotting with better styling and reduced clutter
    for i, u in enumerate(selected_agents):
        hist_o_u = np.array(hist_o['orig'][u])
        hist_p_u = np.array(hist_p['opt'][u])
        color = colors[i % len(colors)]
        
        # Original subplot with enhanced styling - extra thick lines and huge markers
        ax1.plot(hist_o_u, '--', label=f'Agent {u}', linewidth=6.0, 
                color=color, alpha=0.85, markersize=12, markevery=5, marker='o')
        
        # Optimized subplot with enhanced styling - extra thick lines and huge markers
        ax2.plot(hist_p_u, '-', linewidth=6.0,
                color=color, alpha=0.85, markersize=12, markevery=5, marker='s')
    
    # Enhanced formatting for both subplots
    for ax, title in zip((ax1, ax2), ('Original Algorithm', 'Optimized Algorithm')):
        # True consensus line with better styling - extra thick line
        ax.axhline(avg, ls=':', color='black', linewidth=7.0, alpha=0.9, 
                  label=f'True Consensus = {avg:.3f}')
        
        # Enhanced axis labels and title - extra large fonts
        ax.set_xlabel('Iteration (k)', fontweight='bold', fontsize=26)
        ax.set_ylabel('Agent State Value', fontweight='bold', fontsize=26)
        ax.set_title(f'{title}', fontweight='bold', fontsize=30, pad=35)
        
        # Enhanced grid with subtle styling
        ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
        ax.set_axisbelow(True)
        
        # Clean spines design
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        
        # Enhanced tick formatting - extra large numbers
        ax.tick_params(axis='both', which='major', labelsize=22, width=3, length=10)
        ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=6)
    
    # Enhanced legend positioned outside plot area - extra large text
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, 
              fancybox=False, shadow=False, framealpha=0.95, edgecolor='#333333', 
              fontsize=18, ncol=1, columnspacing=0.8, handlelength=3.0)
    
    # Main title for the entire figure - extra large font
    fig.suptitle(f'Resilient Consensus Convergence Comparison (N={N}, Trial {t})', 
                fontweight='bold', fontsize=32, y=0.96, color='#333333')
    
    # Enhanced plot saving with publication quality and space for external legend
    buf = io.BytesIO()
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])  # Leave space for external legend and suptitle
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.3)  # More padding for legend
    plt.close(fig)
    buf.seek(0)
    
    return {
        'trial': t,
        'Orig_subgraph_k': k_o,
        'Opt_subgraph_k': k_p,
        'plot': buf,
        'matrix': A,
        # 'subgraphs': subgraph_matrices,  # Commented out - no subgraph storage
        'honest': honest,
        'avg': avg
    }

def run_trials(
    N:int=50, p_edge:float=0.5, f:int=1,
    eps:float=0.08, T:int=300,
    seed0:int=4, trials:int=10,
    model:str='ER'
) -> DataFrame:
    """Optimized parallel trial runner with vectorization"""
    # Prepare parameters for parallel processing
    params = [(t, N, p_edge, f, eps, T, seed0, model) for t in range(trials)]
    
    # Use number of CPU cores minus 1 to avoid overloading
    num_processes = max(1, cpu_count() - 1)
    print(f"Running {trials} trials using {num_processes} processes for {model} model...")
    
    # Run trials in parallel with chunking for better memory management
    chunk_size = max(1, trials // (num_processes * 2))  # Smaller chunks for better load balancing
    with Pool(num_processes) as pool:
        results = pool.map(process_single_trial, params, chunksize=chunk_size)
    
    # Prepare Excel file
    excel_filename = f'Both_{model}_{N}x{N}_optimized.xlsx'
    
    # Pre-allocate arrays for better performance
    records = []
    # matrix_rows = []  # Commented out - no matrix storage
    
    # Process results efficiently
    print(f"Processing {len(results)} results...")
    
    with ExcelWriter(excel_filename, engine='openpyxl') as writer:
        wb = writer.book
        ws_plots = wb.create_sheet('plots')
        
        # Vectorized processing of results
        for i, result in enumerate(results):
            records.append({
                'trial': result['trial'],
                'Orig_subgraph_k': result['Orig_subgraph_k'],
                'Opt_subgraph_k': result['Opt_subgraph_k']
            })
            
            # Add plot to Excel
            ws_plots.add_image(OpenPyXLImage(result['plot']), f'A{2 + i*20}')
            
            # COMMENTED OUT - Matrix data processing (not needed for now)
            # A = result['matrix']
            # trial_num = result['trial']
            
            # # Create matrix rows for full network
            # for row_idx in range(N):
            #     row_data = {'trial': trial_num, 'version': 'full', 'subgraph': 'none', 'row': row_idx}
            #     # Vectorized column assignment
            #     for col_idx in range(N):
            #         row_data[f'col_{col_idx}'] = A[row_idx, col_idx]
            #     matrix_rows.append(row_data)
            
            # # Create matrix rows for each subgraph
            # for subgraph_name, subgraph_info in result['subgraphs'].items():
            #     A_sub = subgraph_info['matrix']
            #     survivors = subgraph_info['survivors']
            #     removed = subgraph_info['removed']
                
            #     for row_idx in range(A_sub.shape[0]):
            #         row_data = {
            #             'trial': trial_num, 
            #             'version': 'subgraph', 
            #             'subgraph': subgraph_name,
            #             'removed_agents': str(removed),
            #             'survivors': str(survivors),
            #             'row': row_idx
            #         }
            #         # Add matrix values
            #         for col_idx in range(A_sub.shape[1]):
            #             row_data[f'col_{col_idx}'] = A_sub[row_idx, col_idx]
            #         matrix_rows.append(row_data)
        
        # Create DataFrames with vectorized operations
        df = DataFrame(records)
        df.to_excel(writer, sheet_name='summary', index=False)
        
        # Optimized boxplot creation
        fig, ax = plt.subplots(figsize=(6, 4))
        orig_data = df['Orig_subgraph_k'].dropna().values
        opt_data = df['Opt_subgraph_k'].dropna().values
        
        # Use numpy arrays for faster boxplot computation
        ax.boxplot([orig_data, opt_data], labels=['Orig', 'Opt'])
        ax.set_title(f'Subgraph Detection Steps - {model} Network')
        ax.set_ylabel('Detection Round (k)')
        ax.grid(True, alpha=0.3)
        
        # Optimized plot saving
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        ws2 = wb.create_sheet('boxplot')
        ws2.add_image(OpenPyXLImage(buf), 'A1')
        
        # COMMENTED OUT - Write matrices efficiently (not needed for now)
        # if matrix_rows:  # Only create if we have data
        #     mat_df = DataFrame(matrix_rows)
        #     mat_df.to_excel(writer, sheet_name='all_matrices', index=False)
    
    print(f"Saved '{excel_filename}' with {len(records)} trials.")
    print(f"Performance summary - Orig median: {np.median(orig_data):.2f}, Opt median: {np.median(opt_data):.2f}")
    
    return df

if __name__ == '__main__':
    # Windows multiprocessing fix
    from multiprocessing import freeze_support
    freeze_support()
    
    # Test only ER and WS network models (skip BA for now due to connectivity issues)
    for model in ['ER']:
        print(f"\nTesting {model} network model:")
        df = run_trials(trials=1, model=model)
        print(f"\n{model} Network Results:")
        print(df.describe())
