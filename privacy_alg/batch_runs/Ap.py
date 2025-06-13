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

def reconstruct_P(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> np.ndarray:
    p_full = np.zeros(mask_flat.size)
    p_full[mask_flat] = p_vars
    P = p_full.reshape(size, size)
    return row_normalize(P)

def objective_vars(p_vars: np.ndarray, mask_flat: np.ndarray, size: int) -> float:
    P = reconstruct_P(p_vars, mask_flat, size)
    vals = np.abs(eigvals(P))
    return np.sort(vals)[-2]

def simulate_P_metrics(P: np.ndarray, x0: np.ndarray,
                       tol: float = 1e-4, max_steps: int = 500) -> dict:
    size = P.shape[0]
    N = x0.size

    # second‐largest eigenvalue
    eigs = np.abs(eigvals(P))
    lambda2 = np.sort(eigs)[-2]

    # left stationary eigenvector
    w, V = eig(P.T)
    idx = np.argmin(np.abs(w - 1))
    v0 = np.real(V[:, idx])
    v0 /= v0.sum()

    # build augmented x0
    target = x0.mean()
    x_aug = np.zeros(size)
    x_aug[:N] = x0
    for i in range(N, size):
        x_aug[i] = x0[(i-N) % N] / 3
    x_aug *= (target / (v0 @ x_aug))

    # simulate
    X = np.zeros((size, max_steps+1))
    X[:,0] = x_aug
    conv_step = None
    for k in range(max_steps):
        X[:, k+1] = P @ X[:, k]
        if conv_step is None and np.max(np.abs(X[:, k+1] - target)) < tol:
            conv_step = k+1
            break

    final_val = float(X[0, conv_step]) if conv_step is not None else None
    return {
        'lambda2':    float(lambda2),
        'conv_steps': conv_step,
        'final_val':  final_val
    }

# ------------------------------------------------------------

if __name__ == "__main__":

    N = 20
    p_edge = 0.6
    num_runs = 70
    
    seed0 = 42

    # common x0
    rng = np.random.RandomState(1234)
    x0 = rng.rand(N)

    results = []
    all_A0   = []

    for run in range(num_runs):
        seed = seed0 + run
        rng_run = np.random.RandomState(seed)

        # build connected ER graph
        while True:
            G = nx.erdos_renyi_graph(N, p_edge, seed=rng_run)
            if nx.is_connected(G):
                break

        # random positive weights
        for u, v in G.edges():
            G[u][v]['weight'] = rng_run.rand()

        # initial A0
        raw_A = nx.to_numpy_array(G, nodelist=range(N), weight='weight', dtype=float)
        A0    = row_normalize(raw_A)
        all_A0.append(A0)

        # augmented Ap0
        Ap0      = build_Ap(N, A0)
        size     = 4*N
        mask_flat= Ap0.flatten() > 0
        p0_vars  = Ap0.flatten()[mask_flat]

        # simulate original
        m0 = simulate_P_metrics(Ap0, x0)

        # optimize
        res = minimize(
            lambda v: objective_vars(v, mask_flat, size),
            p0_vars,
            method='SLSQP',
            bounds=[(0.01, None)] * p0_vars.size,
            options={'ftol':1e-9,'maxiter':500}
        )

        # build & simulate optimized P_opt
        P_opt = reconstruct_P(res.x, mask_flat, size)
        m1    = simulate_P_metrics(P_opt, x0)

        # percentage speedup
        if m0['conv_steps'] and m1['conv_steps']:
            pct = 100*(m0['conv_steps'] - m1['conv_steps'])/m0['conv_steps']
        else:
            pct = np.nan

        results.append({
            'run':         run+1,
            'λ2_init':     m0['lambda2'],
            'steps_init':  m0['conv_steps'],
            'λ2_opt':      m1['lambda2'],
            'steps_opt':   m1['conv_steps'],
            'pct_speedup': pct
        })

        print(f"Run {run+1}/{num_runs}: "
              f"λ2_init={m0['lambda2']:.4f}, steps={m0['conv_steps']};  "
              f"λ2_opt={m1['lambda2']:.4f}, steps={m1['conv_steps']};  "
              f"speedup={pct:.1f}%")

    df = pd.DataFrame(results)

    # prepare output folder
    out_dir = Path("ExcelDataRuns"); out_dir.mkdir(exist_ok=True)
    excel_path = out_dir/f"Ap_N=20_P=0.6.xlsx"


    df.to_excel(excel_path, sheet_name="Summary", index=False)

    # boxplot
    plt.figure()
    df[['steps_init', 'steps_opt']].boxplot()
    plt.title("Convergence Steps: Before vs After")
    plt.ylabel("Iterations to Converge")
    box_png = out_dir/"boxplot.png"
    plt.tight_layout(); plt.savefig(box_png); plt.close()

    wb = load_workbook(excel_path)
    ws = wb["Summary"]

    cur = 1 + num_runs + 2
    for idx, A0 in enumerate(all_A0, start=1):
        ws.cell(row=cur, column=1).value = f"Run {idx} — initial A₀"
        for i in range(N):
            for j in range(N):
                ws.cell(row=cur+1+i, column=1+j).value = float(f"{A0[i,j]:.3f}")
        cur += N + 2

    # insert boxplot
    img = XLImage(str(box_png))
    ws.add_image(img, f"A{cur}")

    wb.save(excel_path)
    print(f"\n✅ Saved batch report to: {excel_path}")