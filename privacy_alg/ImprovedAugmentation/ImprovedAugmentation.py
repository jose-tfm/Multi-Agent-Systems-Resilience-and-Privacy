import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigvals
from scipy.optimize import minimize

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
        '''
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[3]] = w[2]
        Ap[inds[3], inds[0]] = w[3]
        Ap[inds[3], inds[2]] = w[4]
        Ap[inds[2], inds[0]] = w[5]
        '''
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
        
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[2]] = w[2]
        Ap[inds[2], inds[0]] = w[3]
        Ap[inds[3], inds[0]] = w[4]
        Ap[inds[0], inds[3]] = w[5]
        Ap[inds[1], inds[2]] = w[6]
        Ap[inds[3], inds[2]] = w[7]
       


    return row_normalize(Ap)

def consensus_rate(w: np.ndarray) -> float:
    ''' Calcular o segundo maior eigenvalue, quanto mais perto de 1 pior'''
    P = build_Ap(N, A, w)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

N = 3
A = np.array([
    [0,   0.3, 0.7],
    [0.3, 0,   0.7],
    [0.3, 0.7, 0  ]
], dtype=float)
x0 = np.array([0.5, 1/3, 0.2])

'''
N = 11
A = np.array([
    [0.   , 0.75 , 0.20 , 0.03 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.75 , 0.20 , 0.03 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.02 , 0.   , 0.   , 0.75 , 0.20 , 0.03 , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 , 0.02 ],
    [0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 , 0.20 ],
    [0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   , 0.75 ],
    [0.75 , 0.20 , 0.02 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.03 , 0.   , 0.   ]
], dtype=float)
x0 = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
'''


w0 = np.ones(8)
bounds = [(0.01, None)] * len(w0)

res = minimize(
    consensus_rate,
    w0,
    method="SLSQP",
    bounds=bounds
)
w_opt = res.x

print("Pesos ótimos (w*):", np.round(w_opt, 4))
print("λ₂(P) mínimo =", np.round(res.fun, 4), "\n")

# --- Build augmented matrix Ap and its left eigenvector ---
Ap = build_Ap(N, A, w_opt)
print("Matriz A^P completa:\n", Ap, "\n")

w_vals, V = eig(Ap.T)
idx = np.argmin(np.abs(w_vals - 1))
v0 = np.real(V[:, idx])
v0 /= v0.sum()
print("Left eigenvector v0 (Ap):", np.round(v0, 3), "\n")

# --- Compute true consensus target for the original network ---
w_vals_A, W = eig(A.T)
idx_A = np.argmin(np.abs(w_vals_A - 1))
vA = np.real(W[:, idx_A])
vA /= vA.sum()
target = vA @ x0
print(f"Consensus target (vA^T x0) = {target:.4f}\n")

# --- Build and normalize initial augmented state x_p0 ---
alpha = np.ones(N)
beta  = np.ones(N)
gamma = np.ones(N)

x_p0 = np.zeros(4 * N)
# carry over the original x0 into the augmented vector
x_p0[:N] = x0

for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    coeff = 4 * x0[j] / (a + b + g)
    x_p0[N + 3*j : N + 3*j + 3] = [coeff * a, coeff * b, coeff * g]

# scale so that the augmented system also converges to the same target
x_p0 *= target / (v0 @ x_p0)

# --- Simulate both systems ---
steps, tol = 600, 1e-3
Xo = np.zeros((N, steps + 1))
Xa = np.zeros((4 * N, steps + 1))
Xo[:, 0] = x0
Xa[:, 0] = x_p0

for k in range(steps):
    Xo[:, k+1] = A @ Xo[:, k]
    Xa[:, k+1] = Ap @ Xa[:, k]

# --- Find convergence steps (with safe default) ---
orig_conv = next(
    (k for k in range(steps+1)
     if np.max(np.abs(Xo[:, k] - target)) < tol),
    None
)
aug_conv = next(
    (k for k in range(steps+1)
     if np.max(np.abs(Xa[:, k] - target)) < tol),
    None
)

if orig_conv is None:
    print(f"Original network did NOT converge within {steps} steps (tol={tol}).")
else:
    print(f"Original convergiu em {orig_conv} steps")

if aug_conv is None:
    print(f"Augmented network did NOT converge within {steps} steps (tol={tol}).")
else:
    print(f"Augmented convergiu em {aug_conv} steps")

# --- Final consensus states ---
if orig_conv is not None:
    consensus_orig = Xo[:, orig_conv]
    print(" → Estado final (original):", np.round(consensus_orig, 4))
if aug_conv is not None:
    consensus_aug = Xa[:, aug_conv]
    print(" → Estado final (augmented):", np.round(consensus_aug, 4))

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

for i in range(N):
    ax1.plot(Xo[i], label=f'$x_{{{i+1}}}[k]$')
ax1.axhline(target, ls='--', color='k', label=f'Consensus = {target:.2f}')
ax1.set_title('(a) Consenso na rede original')
ax1.legend()
ax1.grid()

for i in range(4 * N):
    lbl = f'$x_{{{i+1}}}$' if i < N else f'$\\tilde x_{{{(i-N)//3+1},{(i-N)%3+1}}}$'
    ax2.plot(Xa[i], label=lbl)
ax2.axhline(target, ls='--', color='k', label=f'Consensus = {target:.2f}')
ax2.set_title('(b) Consenso na rede augmentada (Ap)')
ax2.legend(ncol=3, fontsize='small')
ax2.grid()

plt.tight_layout()
plt.show()