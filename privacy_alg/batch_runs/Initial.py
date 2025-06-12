import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize
import pandas as pd

# ------------------------------------------------------------
# Row-stochastic normalization
def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

# Build augmented P with fixed gadget weights
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
        ''''
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

def build_A_from_p(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N = mask.shape[0]
    A = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                A[i, j] = p[idx]
                idx += 1
    return row_normalize(A)

def simulate_and_metrics(A_mat: np.ndarray,
                         x0: np.ndarray,
                         tol: float = 1e-4,
                         max_steps: int = 500) -> dict:
    N = x0.size
    P_mat = build_Ap(N, A_mat)

    # second‐largest eigenvalue
    eigs = np.abs(eigvals(P_mat))
    eigs.sort()
    lambda2 = eigs[-2]

    # left stationary eigenvector → v0
    wv, V = eig(P_mat.T)
    idx = np.argmin(np.abs(wv - 1))
    v0 = np.real(V[:, idx])
    v0 /= v0.sum()

    # build augmented x0
    target = x0.mean()
    x_p0 = np.zeros(4*N)
    for j in range(N):
        x_p0[N+3*j:N+3*j+3] = 4 * x0[j] / 3
    x_p0 *= target / (v0 @ x_p0)

    # simulate both systems
    Xo = np.zeros((N,   max_steps+1))
    Xa = np.zeros((4*N, max_steps+1))
    Xo[:, 0] = x0
    Xa[:, 0] = x_p0

    orig_steps = None
    aug_steps  = None

    for k in range(max_steps):
        Xo[:, k+1] = A_mat @ Xo[:, k]
        Xa[:, k+1] = P_mat @ Xa[:, k]
        if orig_steps is None and np.max(np.abs(Xo[:, k+1] - target)) < tol:
            orig_steps = k+1
        if aug_steps  is None and np.max(np.abs(Xa[:, k+1] - target)) < tol:
            aug_steps = k+1

    return {
        'lambda2':    lambda2,
        'orig_steps': orig_steps,
        'aug_steps':  aug_steps
    }

# ------------------------------------------------------------
# 2) Closure‐factory for the optimizer

def make_obj(N: int, mask: np.ndarray):
    def objective(p):
        A = build_A_from_p(p, mask)
        vals = np.abs(eigvals(build_Ap(N, A)))
        vals.sort()
        return vals[-2]
    return objective

# ------------------------------------------------------------
# 3) Single‐experiment runner

def one_experiment(N: int,
                   seed: int,
                   p_edge: float,
                   x0: np.ndarray,
                   tol: float = 1e-4,
                   max_steps: int = 500) -> dict:
    # -- build random connected weighted ER graph
    np.random.seed(seed)
    while True:
        G = nx.erdos_renyi_graph(N, p_edge, seed=seed)
        if nx.is_connected(G):
            break
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.rand()

    raw_A = row_normalize(nx.to_numpy_array(G,
                                           nodelist=range(N),
                                           weight='weight'))
    mask  = raw_A > 0

    # flatten free variables
    p0   = np.array([raw_A[i, j]
                     for i in range(N) for j in range(N) if mask[i, j]])
    A0   = build_A_from_p(p0, mask)

    # simulate original
    init_m = simulate_and_metrics(A0, x0, tol, max_steps)

    # optimize with correct mask
    obj = make_obj(N, mask)
    res = minimize(obj,
                   p0,
                   method='SLSQP',
                   bounds=[(0.1, None)] * p0.size,
                   options={'ftol': 1e-9, 'maxiter': 500})
    p_opt = res.x
    A_opt = build_A_from_p(p_opt, mask)

    # simulate optimized
    opt_m = simulate_and_metrics(A_opt, x0, tol, max_steps)

    return {
        'lambda2_init':   init_m['lambda2'],
        'lambda2_opt':    opt_m['lambda2'],
        'steps_init':     init_m['orig_steps'],
        'steps_opt':      opt_m['orig_steps'],
        'aug_steps_init': init_m['aug_steps'],
        'aug_steps_opt':  opt_m['aug_steps'],
        'success':        res.success
    }


if __name__ == "__main__":
    N        = 11     
    p_edge   = 0.8 
    num_runs = 50 

    # fix a common initial state
    x0_common = np.array([0.1, 0.3, 0.6, 0.43, 0.85,
                          0.9, 0.45, 0.11, 0.06, 0.51, 0.13])

    results = []
    for run in range(num_runs):
        seed = 4 + run
        m = one_experiment(N,
                           seed,
                           p_edge,
                           x0_common,
                           tol=1e-4,
                           max_steps=500)
        results.append(m)

        # original A convergence
        si  = m['steps_init']     if m['steps_init']     is not None else ">=500"
        sai = m['aug_steps_init'] if m['aug_steps_init'] is not None else ">=500"
        
        # optimized A convergence
        so  = m['steps_opt']      if m['steps_opt']      is not None else ">=500"
        sao = m['aug_steps_opt']  if m['aug_steps_opt']  is not None else ">=500"

        print(f"Run {run+1:2d}/{num_runs:2d} — "
              f"λ2_init={m['lambda2_init']:.4f}, P_init steps={sai};  "
              f"λ2_opt={m['lambda2_opt']:.4f}, P_opt steps={sao} "
              f"[opt {'OK' if m['success'] else 'FAIL'}]")

    df = pd.DataFrame(results)
    print("\n=== Summary over {:d} runs ===".format(num_runs))
    for col in ['lambda2_init','lambda2_opt',
                'aug_steps_init','aug_steps_opt']:
        mean_, std_ = df[col].mean(), df[col].std(ddof=1)
        print(f"{col:15s}: mean={mean_:.4f}, std={std_:.4f}")

    # optional: histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()
    for ax, col in zip(axes,
                       ['lambda2_init','lambda2_opt',
                        'aug_steps_init','aug_steps_opt']):
        ax.hist(df[col].dropna(), bins=10)
        ax.axvline(df[col].mean(), linestyle='--')
        ax.set_title(col)
    plt.tight_layout()
    plt.show()
