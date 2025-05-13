import numpy as np
import networkx as nx
from scipy.linalg import eig


def row_normalize(M):
    S = M.sum(axis=1, keepdims=True)
    S[S==0] = 1e-12
    return M / S

def observability_matrix(A, C, steps=None):
    n = A.shape[0]
    if steps is None:
        steps = n
    mats, Ak = [], np.eye(n)
    for _ in range(steps):
        mats.append(C @ Ak)
        Ak = A @ Ak
    return np.vstack(mats)

def in_colspace(O, v, tol=1e-8):
    x, residuals, *_ = np.linalg.lstsq(O, v, rcond=None)
    if residuals.size:
        return residuals[0] < tol
    return np.linalg.norm(O @ x - v) < tol

def passes_privacy(A4s, tol=1e-8):
    """
    Given a 4×4 row‐stochastic A4s, check:
      1) Observability-rank < 4 from C=[1,0,0,0]
      2) e0 ∉ span(O)
      3) p=1/v0 ∉ span(O)  (PBH privacy test)
    """
    C = np.zeros((1,4)); C[0,0]=1
    O = observability_matrix(A4s, C)
    if np.linalg.matrix_rank(O, tol) == 4:
        return False
    e0 = np.zeros(4); e0[0]=1
    if in_colspace(O, e0, tol):
        return False
    vals, vecs = eig(A4s.T)
    idx = np.argmin(np.abs(vals-1))
    v0 = np.real(vecs[:,idx])
    v0 /= v0.sum()
    p = 1.0/v0
    if in_colspace(O, p, tol):
        return False
    return True

def pattern_allows_parameterization(A4, trials=100):
    """
    For a binary pattern A4, try `trials` random positive weightings
    of its nonzeros—if any weighting yields privacy, return True.
    """
    nz = np.argwhere(A4>0)
    m  = len(nz)
    if m==0:
        return False

    for _ in range(trials):
        W = np.zeros_like(A4, dtype=float)
        # sample weights ∈ [0.5, 5.0] uniformly
        ws = 0.5 + 4.5*np.random.rand(m)
        for (i,j),w in zip(nz, ws):
            W[i,j] = w
        A4s = row_normalize(W)
        if passes_privacy(A4s):
            return True
    return False

def enumerate_13_gadgets():
    survivors = []
    # 1) enumerate all binary 4×4 no‐self‐loops
    for code in range(1<<16):
        A4 = np.zeros((4,4),int)
        for bit in range(16):
            if (code>>bit)&1:
                i,j = divmod(bit,4)
                A4[i,j] = 1
        np.fill_diagonal(A4, 0)

        # 2) strong connectivity & aperiodicity (unweighted)
        G = nx.DiGraph(A4)
        if not nx.is_strongly_connected(G): 
            continue
        if not nx.is_aperiodic(G):
            continue

        # 3) existence of some weight assignment that yields privacy
        if pattern_allows_parameterization(A4, trials=200):
            survivors.append(A4)

    # 4) collapse isomorphic classes
    patterns = []
    for A4 in survivors:
        G = nx.DiGraph(A4)
        # only keep one representative per isomorphism class
        if not any(nx.is_isomorphic(G,H) for H in patterns):
            patterns.append(G)

    # convert back to adjacency‐matrices
    return [nx.to_numpy_array(G, dtype=int) for G in patterns]

if __name__=='__main__':
    gadgets = enumerate_13_gadgets()
    print(f"Found {len(gadgets)} non-isomorphic gadgets (should be 13)\n")
    for k,A4 in enumerate(gadgets,1):
        print(f"--- gadget {k} ---")
        print(A4, "\n")
