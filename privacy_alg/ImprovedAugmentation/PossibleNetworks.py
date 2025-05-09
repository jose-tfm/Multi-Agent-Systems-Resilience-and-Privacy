import numpy as np
import networkx as nx
import math
from scipy.linalg import eig

# --------------------------
# 1) Utilities
# --------------------------
def row_normalize(M):
    """Row-stochastic normalization (avoid division-by-zero)."""
    sums = M.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1e-12
    return M / sums


def observability_matrix(A, C, steps=None):
    """Build O = [C; C A; C A^2; ...] up to `steps` (default n)."""
    n = A.shape[0]
    if steps is None:
        steps = n
    mats, Ak = [], np.eye(n)
    for _ in range(steps):
        mats.append(C @ Ak)
        Ak = A @ Ak
    return np.vstack(mats)


def in_column_space(O, v, tol=1e-6):
    """Check if vector v lies in the column space of O."""
    x, residuals, *_ = np.linalg.lstsq(O, v, rcond=None)
    if residuals.size:
        return residuals[0] < tol
    return np.linalg.norm(O @ x - v) < tol

# --------------------------
# 2) Privacy check on a 4×4 gadget
# --------------------------
def check_gadget_privacy(A4, tol=1e-6):
    """
    Test a 4×4 binary adjacency A4 for:
      1) strong connectivity
      2) aperiodicity via NetworkX
      3) unobservability & privacy-parameterization
    """
    # 1) unweighted connectivity
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i, j in np.argwhere(A4 > 0):
        G.add_edge(i, j)
    if not nx.is_strongly_connected(G):
        return False

    # 2) aperiodicity via NetworkX (requires networkx >= 2.8)
    # nx.is_aperiodic returns False on acyclic or periodic graphs
    if not nx.is_aperiodic(G):
        return False

    # 3) build weighted stochastic matrix
    A4s = row_normalize(A4.astype(float))

    # 3a) Observability from original node 0
    C = np.zeros((1, 4)); C[0, 0] = 1
    O = observability_matrix(A4s, C)
    # must be rank-deficient (<4)
    if np.linalg.matrix_rank(O, tol) == 4:
        return False
    # original state not observable
    e0 = np.zeros(4); e0[0] = 1
    if in_column_space(O, e0, tol):
        return False

    # 3b) privacy-parameterization test
    evals, evecs = eig(A4s.T)
    idx = np.argmin(np.abs(evals - 1))
    v0 = np.real(evecs[:, idx]); v0 /= v0.sum()
    p = 1.0 / v0
    if in_column_space(O, p, tol):
        return False

    return True

# --------------------------
# 3) Main enumeration
# --------------------------
def enumerate_gadgets():
    survivors = []
    for code in range(1 << 16):
        A4 = np.zeros((4, 4), dtype=int)
        for bit in range(16):
            if (code >> bit) & 1:
                i, j = divmod(bit, 4)
                A4[i, j] = 1
        np.fill_diagonal(A4, 0)
        if check_gadget_privacy(A4):
            survivors.append(A4)

    # collapse isomorphic unweighted graphs
    patterns = []
    for A4 in survivors:
        G = nx.DiGraph(); G.add_nodes_from(range(4))
        for i, j in np.argwhere(A4 > 0):
            G.add_edge(i, j)
        if not any(nx.is_isomorphic(G, H) for H in patterns):
            patterns.append(G)

    return [nx.to_numpy_array(G, dtype=int) for G in patterns]

if __name__ == '__main__':
    patterns = enumerate_gadgets()
    print(f"Found {len(patterns)} non-isomorphic privacy gadgets (should be 13)")
    
    def diagnose(A4, tol=1e-6):
        # Build unweighted graph
        G = nx.DiGraph(); G.add_nodes_from(range(4))
        for i,j in np.argwhere(A4>0): G.add_edge(i,j)
        conn = nx.is_strongly_connected(G)
        ap = nx.is_aperiodic(G)
        # Build weighted stochastic
        A4s = row_normalize(A4.astype(float))
        # Observability
        C = np.zeros((1,4)); C[0,0]=1
        O = observability_matrix(A4s, C)
        rankO = np.linalg.matrix_rank(O, tol)
        e0 = np.zeros(4); e0[0]=1; orig_obs = in_column_space(O, e0, tol)
        # Parameterization
        evals, evecs = eig(A4s.T)
        idx = np.argmin(np.abs(evals-1))
        v0 = np.real(evecs[:,idx]); v0/=v0.sum()
        p = 1.0/v0; param_obs = in_column_space(O, p, tol)
        return conn, ap, rankO, orig_obs, param_obs

    for idx, M in enumerate(patterns, 1):
        print(f"--- Pattern {idx} ---")
        print(M)
        conn, ap, rankO, orig_obs, param_obs = diagnose(M)
        print(f"Connectivity: {conn}")
        print(f"Aperiodic: {ap}")
        print(f"Observability rank: {rankO} (<4 is good)")
        print(f"Original observable: {orig_obs} (should be False)")
        print(f"Param observable: {param_obs} (should be False)")
