import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigvals
from scipy.optimize import minimize
import networkx as nx
import random
np.set_printoptions(
    threshold=np.inf,
    linewidth=200,
    precision=3,
    suppress=True
)

# ------------------------------------------------------------
def row_normalize(M: np.ndarray) -> np.ndarray:
    """Normalização row-stochastic"""

    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

def build_Ap(N: int, A: np.ndarray, w: np.ndarray) -> np.ndarray: 
    """
    Constrói a matriz 4N×4N com pesos w = [w01, w10, w02, w23, w30]
    para cada gadget:
      orig→aug1  = w[0]
      aug1→orig  = w[1]
      orig→aug2  = w[2]
      aug2→aug3  = w[3]
      aug3→orig  = w[4]
    """

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
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[3]] = w[2]
        Ap[inds[3], inds[2]] = w[3] 
        Ap[inds[2], inds[0]] = w[4]
        '''
        # ——————————————————————
        # b)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[0], inds[3]] = w[3]
        Ap[inds[2], inds[0]] = w[4]
        Ap[inds[3], inds[2]] = w[5]
        '''
        # ——————————————————————
        # c)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[0], inds[2]] = w[1]
        Ap[inds[0], inds[3]] = w[2]
        Ap[inds[1], inds[3]] = w[3]
        Ap[inds[2], inds[3]] = w[4]
        Ap[inds[3], inds[0]] = w[5]
        '''
        # ——————————————————————
        # d)
        
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[3]] = w[2]
        Ap[inds[3], inds[0]] = w[3]
        Ap[inds[3], inds[2]] = w[4]
        Ap[inds[2], inds[0]] = w[5]
        
        # ——————————————————————
        # e)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[2]] = w[1]
        Ap[inds[2], inds[0]] = w[2]
        Ap[inds[3], inds[2]] = w[3]
        Ap[inds[0], inds[3]] = w[4]
        Ap[inds[3], inds[0]] = w[5]
        '''
        # ——————————————————————
        # f)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[1], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[0], inds[3]] = w[4]
        Ap[inds[3], inds[0]] = w[5]
        Ap[inds[3], inds[2]] = w[6]
        '''
        # ——————————————————————
        # g)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[0], inds[3]] = w[4]
        Ap[inds[3], inds[0]] = w[5]
        Ap[inds[3], inds[2]] = w[6]
        '''
        # ——————————————————————
        # h)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[1], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[0], inds[2]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[3], inds[2]] = w[6]
        '''
        # ——————————————————————
        # i)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[2]] = w[1]
        Ap[inds[2], inds[1]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[0], inds[2]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[3], inds[2]] = w[6]
        '''
        # ——————————————————————
        # j)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[2], inds[1]] = w[3]
        Ap[inds[2], inds[3]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[3], inds[0]] = w[6]
        '''
        # ——————————————————————
        # k)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[1], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[3], inds[2]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[3], inds[0]] = w[6]
        '''
        # ——————————————————————
        # l)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[3], inds[0]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[2], inds[1]] = w[6]
        Ap[inds[2], inds[3]] = w[7]
        '''
        # ——————————————————————
        # m)
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[3], inds[0]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[1], inds[2]] = w[6]
        Ap[inds[3], inds[2]] = w[7]
       '''
    return row_normalize(Ap)

def consensus_rate(w: np.ndarray) -> float:
    ''' Calcular o segundo maior eigenvalue, quanto mais perto de 1, por a velocidade de convergencia'''
    P = build_Ap(N, A_stochastic, w)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

def make_strong_backbone(n, p, seed=None):
    random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # add a directed cycle 0→1→2→…→n−1→0
    for i in range(n):
        G.add_edge(i, (i+1) % n)
    # sprinkle extra random edges
    for i in range(n):
        for j in range(n):
            if i!=j and random.random() < p:
                G.add_edge(i,j)
    return G

# Parameters
N = 1000                 # number of original nodes
p = 0.1                  # edge creation probability
tol = 1e-4               # convergence tolerance
steps = 100            # maximum iterations

#------------------------------------------------------------
G = make_strong_backbone(N, p, seed=42)
G.remove_edges_from(nx.selfloop_edges(G))
A = nx.to_numpy_array(G, dtype=float)
row_sums = A.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1e-12
A_stochastic = A / row_sums
x0 = np.random.rand(N)

#------------------------------------------------------------
# Helper: row-normalize any matrix

def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

#------------------------------------------------------------
# Build the augmented transition matrix A^P using gadget type 'd'
def build_Ap(N: int, A: np.ndarray, w: np.ndarray) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    # original-to-original block
    Ap[:N, :N] = A

    for i in range(N):
        inds = {0: i, 1: N+3*i, 2: N+3*i+1, 3: N+3*i+2}
        # gadget 'd' edges
        Ap[inds[0], inds[1]] = w[0]  # orig → aug1
        Ap[inds[1], inds[0]] = w[1]  # aug1 → orig
        Ap[inds[0], inds[3]] = w[2]  # orig → aug3
        Ap[inds[3], inds[0]] = w[3]  # aug3 → orig
        Ap[inds[3], inds[2]] = w[4]  # aug3 → aug2
        Ap[inds[2], inds[0]] = w[5]  # aug2 → orig

    return row_normalize(Ap)

#------------------------------------------------------------
# Objective: second-largest eigenvalue magnitude (consensus rate)
def consensus_rate(w: np.ndarray) -> float:
    P = build_Ap(N, A_stochastic, w)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

#------------------------------------------------------------
# Optimization: find homogeneous weights w* subject to w[1]=2*w[3]
w0 = np.ones(6)
bounds = [(0.1, 5)] * 6
cons = {'type': 'eq', 'fun': lambda w: w[1] - 2*w[3]}
res = minimize(consensus_rate, w0, method='SLSQP', bounds=bounds, constraints=[cons])
w_opt = res.x
print("Optimal weights w*:", np.round(w_opt, 4))
print("λ₂(P) minimal =", np.round(res.fun, 4), "\n")

# Recompute P* and left eigenvector v0
target = x0.mean()
P_opt = build_Ap(N, A_stochastic, w_opt)
w_vals, V = eig(P_opt.T)
idx = np.argmin(np.abs(w_vals - 1))
v0 = np.real(V[:, idx]); v0 /= v0.sum()
print("Left eigenvector v0 =", np.round(v0, 4), "\n")

#------------------------------------------------------------
# Build augmented initial state x_p0
alpha = np.random.rand(N)
beta  = np.random.rand(N)
gamma = np.random.rand(N)

x_p0 = np.zeros(4*N)
for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    coeff = 4 * x0[j] / (a + b + g)
    x_p0[N+3*j    ] = coeff * a
    x_p0[N+3*j + 1] = coeff * b
    x_p0[N+3*j + 2] = coeff * g
# Rescale to consensus target
data_scale = target / (v0 @ x_p0)
x_p0 *= data_scale

#------------------------------------------------------------
# Simulate consensus dynamics
Xo = np.zeros((N, steps+1))
Xa = np.zeros((4*N, steps+1))
Xo[:, 0] = x0
Xa[:, 0] = x_p0
for k in range(steps):
    Xo[:, k+1] = A_stochastic @ Xo[:, k]
    Xa[:, k+1] = P_opt        @ Xa[:, k]

# Determine convergence iterations with default
orig_conv = next((k for k in range(steps+1)
                  if np.max(np.abs(Xo[:,k] - target)) < tol), None)
aug_conv  = next((k for k in range(steps+1)
                  if np.max(np.abs(Xa[:,k] - target)) < tol), None)
print(f"Original converged in {orig_conv} steps")
print(f"Augmented converged in {aug_conv} steps")

#------------------------------------------------------------
# Plot trajectories
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10))
for i in range(N):
    ax1.plot(Xo[i], alpha=0.5)
ax1.axhline(target, ls='--', color='k')
ax1.set_title('Consensus on original network')
ax1.grid()

for i in range(4*N):
    ax2.plot(Xa[i], alpha=0.3)
ax2.axhline(target, ls='--', color='k')
ax2.set_title('Consensus on augmented network')
ax2.grid()
plt.tight_layout()
plt.show()
