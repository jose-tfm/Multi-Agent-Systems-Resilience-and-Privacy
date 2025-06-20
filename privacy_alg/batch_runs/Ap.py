import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage


def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s


def build_Ap(N, A):
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
        # ——————————————————————
        # a)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        '''
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
        
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[3], inds[2]] = 1
        
    return row_normalize(Ap)

def reconstruct_P(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> np.ndarray:
    p_full = np.zeros(mask_flat.size)
    p_full[mask_flat] = p_vars
    P = p_full.reshape(size, size)
    return row_normalize(P)

def objective_vars(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> float:
    P = reconstruct_P(p_vars, mask_flat, size)
    vals = np.abs(eigvals(P))
    return np.sort(vals)[-2]

def build_initial_x_aug(P: np.ndarray, A: np.ndarray, x0: np.ndarray):
    """Compute true consensus target from A and build augmented initial x_p0."""
    N = x0.size
    # 1) Left Perron of A
    wA, WA = eig(A.T)
    idxA = np.argmin(np.abs(wA - 1))
    vA = np.real(WA[:, idxA])
    vA /= vA.sum()
    target = vA @ x0

    # 2) Distribute each x0[j] into its three augmented states
    x_p0 = np.zeros(4*N)
    for j in range(N):
        # uniform α=β=γ=1 by default
        coeff = 4.0 * x0[j] / 3.0
        x_p0[N+3*j    ] = coeff
        x_p0[N+3*j + 1] = coeff
        x_p0[N+3*j + 2] = coeff

    # 3) Scale so that v0^T x_p0 == target
    w0, V0 = eig(P.T)
    idx0 = np.argmin(np.abs(w0 - 1))
    v0 = np.real(V0[:, idx0])
    v0 /= v0.sum()
    x_p0 *= (target / (v0 @ x_p0))

    return x_p0, target

def simulate_P_metrics(P: np.ndarray, x_p0: np.ndarray,
                       target: float, tol: float = 1e-4, max_steps: int = 500) -> dict:
    size = P.shape[0]

    # second‐largest eigenvalue
    lambda2 = np.sort(np.abs(eigvals(P)))[-2]

    # simulate consensus
    x = x_p0.copy()
    conv_step = None
    for k in range(1, max_steps+1):
        x = P @ x
        if conv_step is None and np.max(np.abs(x - target)) < tol:
            conv_step = k
            break

    return {
        'lambda2':    float(lambda2),
        'conv_steps': conv_step,
        'final_val':  float(x[0]) if conv_step is not None else None
    }

# -----------------------------------------------------------------
# Main script
# -----------------------------------------------------------------
if __name__ == "__main__":

    N        = 11
    p_edge   = 0.8
    num_runs = 70
    seed0    = 42

    rng = np.random.RandomState(1234)
    x0  = rng.rand(N)

    results = []
    all_A0  = []

    for run in range(1, num_runs+1):
        seed    = seed0 + run
        rng_run = np.random.RandomState(seed)

        # build random ER until strongly connected
        while True:
            D = nx.gnp_random_graph(N, p_edge, seed=rng, directed=True)
            if nx.is_strongly_connected(D):
                G = D
                break

        # assign random weights
        for u, v in G.edges():
            G[u][v]['weight'] = rng_run.rand()

        # build A0 and P0
        raw_A = nx.to_numpy_array(G, nodelist=range(N),
                                  weight='weight', dtype=float)
        A0    = row_normalize(raw_A)
        all_A0.append(A0)

        wA, WA   = eig(A0.T)
        idxA     = np.argmin(np.abs(wA - 1))
        vA       = np.real(WA[:, idxA])
        vA      /= vA.sum()
        print(f"  Left eigenvector vA (sum=1):\n    {vA}\n")

        P0    = build_Ap(N, A0)

        # initial augmented state
        x_p0, target = build_initial_x_aug(P0, A0, x0)

        priv = x_p0[N:].reshape(N, 3)
        print("  Private‐state triples [ᾱᵢ, βᵢ, γᵢ] for each agent:")
        for j, triple in enumerate(priv):
            print(f"    agent {j:2d}: {triple}")
        print()

        # hop-diameter of augmented P0
        H = nx.from_numpy_array((P0>0).astype(int),
                                create_using=nx.DiGraph)
        lengths_P = dict(nx.all_pairs_shortest_path_length(H))
        hop_diam = 0
        for dist in lengths_P.values():
            hop_diam = max(hop_diam, max(dist.values()))
        print(f"Run {run}: Augmented hop-diameter = {hop_diam}")

        # simulate before optimization
        m0 = simulate_P_metrics(P0, x_p0, target)

        # optimize P
        size      = 4*N
        mask_flat = P0.flatten() > 0
        p0_vars   = P0.flatten()[mask_flat]
        res = minimize(
            lambda v: objective_vars(v, mask_flat, size),
            p0_vars,
            method='SLSQP',
            bounds=[(0.01, None)] * p0_vars.size,
            options={'ftol': 1e-9, 'maxiter': 500}
        )

        # build optimized P and its initial state
        P_opt    = reconstruct_P(res.x, mask_flat, size)
        x_p_opt, _ = build_initial_x_aug(P_opt, A0, x0)

        # simulate after optimization
        m1 = simulate_P_metrics(P_opt, x_p_opt, target)

        pct = 100*(m0['conv_steps'] - m1['conv_steps'])/m0['conv_steps'] \
              if m0['conv_steps'] and m1['conv_steps'] else np.nan

        results.append({
            'run':            run,
            'hop_diameter':   hop_diam,
            'λ2_init':        m0['lambda2'],
            'steps_init':     m0['conv_steps'],
            'λ2_opt':         m1['lambda2'],
            'steps_opt':      m1['conv_steps'],
            'pct_speedup':    pct
        })

        print(f"   λ2_init={m0['lambda2']:.4f}, steps={m0['conv_steps']}; "
              f"λ2_opt={m1['lambda2']:.4f}, steps={m1['conv_steps']}; "
              f"speedup={pct:.1f}%")

    # save and plot
    df = pd.DataFrame(results)
    out_dir = Path("ExcelDataRuns"); out_dir.mkdir(exist_ok=True)
    excel_path = out_dir / f"Ap_N={N}_P={p_edge}.xlsx"
    df.to_excel(excel_path, sheet_name="Summary", index=False)

    plt.figure()
    df[['steps_init','steps_opt']].boxplot()
    plt.title("Convergence Steps: Before vs After")
    plt.ylabel("Iterations to Converge")
    graph = df[['run','hop_diameter','steps_init','steps_opt']]
