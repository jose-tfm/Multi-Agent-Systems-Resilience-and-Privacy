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
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.linalg import eig
import math

def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N, A):
    """
    Builds the 4N×4N privacy‐augmented matrix.  
    Edit the five Ap[...]=w lines under “CHANGE THESE LINES” to try any 4‐node gadget.
    """
    size = 4 * N
    Ap = np.zeros((size, size))

    # copy the original adjacency
    Ap[:N, :N] = A

    # augment each node
    for i in range(N):
        # global indices of the 4‐node gadget attached to i
        inds = {
            0: i,         # original
            1: N + 3*i,   # aug1
            2: N + 3*i+1, # aug2
            3: N + 3*i+2  # aug3
        }

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
        
        Ap[inds[0], inds[1]] = 1   
        Ap[inds[1], inds[0]] = 1   
        Ap[inds[0], inds[2]] = 1   
        Ap[inds[0], inds[3]] = 1   
        Ap[inds[2], inds[0]] = 1   
        Ap[inds[3], inds[2]] = 1  
        
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
        # g)
        '''
        Ap[inds[0], inds[1]] = 1   # orig → aug1
        Ap[inds[1], inds[0]] = 1   # aug1 → orig
        Ap[inds[0], inds[2]] = 1   # orig → aug2
        Ap[inds[2], inds[0]] = 1   # aug2 → orig
        Ap[inds[0], inds[3]] = 1   # orig → aug3
        Ap[inds[3], inds[0]] = 1   # aug3 → orig 
        Ap[inds[3], inds[2]] = 1   # aug3 → orig  
        '''

    return row_normalize(Ap)


# Parameters
N = 11
A = np.array([
    [0,   1/3, 1/3, 0,   0,   0,   0,   0,   0,   0,   1/3],
    [1/3, 0,   1/3, 1/3, 0,   0,   0,   0,   0,   0,   0  ],
    [1/3, 1/3, 0,   1/3, 0,   0,   0,   0,   0,   0,   0  ],
    [0,   0,   1/3, 0,   1/3, 0,   0,   1/3, 0,   0,   0  ],
    [0,   1/3, 0,   1/3, 0,   1/3, 0,   0,   0,   0,   0  ],
    [0,   0,   0,   0,   1/3, 0,   1/3, 0,   0,   1/3, 0  ],
    [0,   0,   0,   0,   0,   1/2, 0,   1/2, 0,   0,   0  ],
    [0,   0,   0,   1/3, 0,   0,   1/3, 0,   1/3, 0,   0  ],
    [0,   0,   0,   0,   0,   1/2, 0,   1/2, 0,   0,   0  ],
    [0,   0,   0,   0,   0,   0,   1/3, 0,   0,   1/3, 1/3],
    [1/2, 0,   0,   0,   0,   0,   0,   0,   0,   1/2, 0  ]
])
x0 = np.array([0.1, 0.3, 0.6, 0.43, 0.85, 0.9, 0.45, 0.11, 0.06, 0.51, 0.13])
A_norm = row_normalize(A)


# Build and normalize A^P (with the 2-edge correctly in place)
Ap = build_Ap(N, A)

# Compute the normalized left eigenvector v0
w, V = eig(Ap.T)
idx = np.argmin(np.abs(w - 1))
v0 = np.real(V[:, idx])
v0 /= v0.sum()

# Print v0
print("Left eigenvector v0:")
for i, val in enumerate(v0):
    state = f"orig{i+1}" if i < N else f"aug{(i-N)//3+1}_{(i-N)%3+1}"
    print(f"  {state}: {val:.4f}")

# Observability check
#C = np.hstack([np.eye(4*N), np.zeros((4*N, 4*N))])
#P_list = [C @ np.linalg.matrix_power(Ap, k) for k in range(4*N)]
#P_O = np.vstack(P_list)
#P_sym = sp.Matrix(P_O)
#e10 = sp.Matrix([[1 if i==9 else 0] for i in range(12)])
#e12 = sp.Matrix([[1 if i==11 else 0] for i in range(12)])
#print("e10 observable?", e10 in P_sym.columnspace())
#print("e12 observable?", e12 in P_sym.columnspace())

# Inject initial conditions
alpha = np.full(N, 1.4)
beta  = np.full(N, 1.0)
gamma = np.full(N, 1.0)
x_p0 = np.zeros(4*N)
for j in range(N):
    a,b,g = alpha[j], beta[j], gamma[j]
    s = a+b+g
    coeff = 4 * x0[j] / s
    x_p0[j]            = 0.0
    x_p0[N+3*j+0] = coeff * a
    x_p0[N+3*j+1] = coeff * b
    x_p0[N+3*j+2] = coeff * g

# global rescale so v0⋅x_p0 = mean(x0)
target = x0.mean()
x_p0 *= target / (v0 @ x_p0)

# Simulate
steps = 400
X_orig = np.zeros((N, steps+1))
X_aug  = np.zeros((4*N, steps+1))
X_orig[:,0] = x0
X_aug[:,0]  = x_p0
for k in range(steps):
    X_orig[:,k+1] = A_norm @ X_orig[:,k]
    X_aug[:,k+1]  = Ap     @ X_aug[:,k]


# ---- Convergence detection ----
tol = 1e-4
orig_conv = next((k for k in range(steps+1)
                  if np.max(np.abs(X_orig[:,k] - target)) < tol),
                 None)
aug_conv  = next((k for k in range(steps+1)
                  if np.max(np.abs(X_aug[:,k]  - target)) < tol),
                 None)

print(f"Original convergiu em {orig_conv} passos (tol={tol})")
print(f"Augmented convergiu em {aug_conv} passos (tol={tol})")


# Plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10))
for i in range(N):
    ax1.plot(X_orig[i], label=f'$x_{{{i+1}}}[k]$')
ax1.axhline(target, color='k', ls='--', label=f'Consensus={target:.2f}')
ax1.set_title('(a) Consensus on $G(A)$')
ax1.legend(); ax1.grid()

for i in range(4*N):
    if i < N:
        lbl = f'$x_{{{i+1}}}[k]$'
    else:
        agent = (i-N)//3 + 1
        aug   = (i-N)%3 + 1
        lbl   = f'$\~x_{{{agent},{aug}}}[k]$'
    ax2.plot(X_aug[i], label=lbl)
ax2.axhline(target, color='k', ls='--', label=f'Consensus={target:.2f}')
ax2.set_title('(b) Consensus on $G(A^P)$ with 2-edge gadget')
ax2.legend(ncol=3, fontsize='small'); ax2.grid()

plt.tight_layout()
plt.show()
