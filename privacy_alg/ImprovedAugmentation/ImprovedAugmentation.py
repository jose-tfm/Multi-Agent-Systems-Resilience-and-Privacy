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
       
        Ap[inds[0], inds[1]] = 1   # orig → aug1
        Ap[inds[1], inds[0]] = 1   # aug1 → orig 
        Ap[inds[0], inds[3]] = 2   # orig → aug2
        Ap[inds[3], inds[2]] = 1   # aug2 → aug3
        Ap[inds[2], inds[0]] = 1   # aug3 → orig
        
        # ——————————————————————
        # b)
        '''
        Ap[inds[0], inds[1]] = 1   # orig → aug1
        Ap[inds[1], inds[0]] = 1   # aug1 → orig   (the special “2”)
        Ap[inds[0], inds[2]] = 1   # orig → aug2
        Ap[inds[0], inds[3]] = 1   # orig → aug3
        Ap[inds[2], inds[0]] = 1   # aug2 → orig   (the special “2”)
        Ap[inds[3], inds[2]] = 1   # aug3 → aug2
        '''
        # ——————————————————————
        # c)
        '''
        Ap[inds[0], inds[1]] = 1   # orig → aug1
        Ap[inds[0], inds[2]] = 1   # orig → aug2
        Ap[inds[0], inds[3]] = 1   # orig → aug3
        Ap[inds[1], inds[3]] = 1   # aug1 → aug3
        Ap[inds[2], inds[3]] = 1   # aug2 → aug3
        Ap[inds[3], inds[0]] = 2   # aug3 → orig  (the “2” shown as two heads)
        '''

        # ——————————————————————
        # g)
        
        Ap[inds[0], inds[1]] = 1   # orig → aug1
        Ap[inds[1], inds[0]] = 1   # aug1 → orig
        Ap[inds[0], inds[2]] = 1   # orig → aug2
        Ap[inds[2], inds[0]] = 1   # aug2 → orig
        Ap[inds[0], inds[3]] = 1   # orig → aug3
        Ap[inds[3], inds[0]] = 1   # aug3 → orig 
        Ap[inds[3], inds[2]] = 1   # aug3 → orig  
        

    return row_normalize(Ap)


# Parameters
N = 3
A = np.array([[0,1,1],
              [1,0,1],
              [1,1,0]], dtype=float)
A_norm = row_normalize(A)
x0 = np.array([0.5, 0.33, 0.2])

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
C = np.hstack([np.eye(3*2), np.zeros((3*2, 3*2))])
P_list = [C @ np.linalg.matrix_power(Ap, k) for k in range(4*N)]
P_O = np.vstack(P_list)
P_sym = sp.Matrix(P_O)
e10 = sp.Matrix([[1 if i==9 else 0] for i in range(12)])
e12 = sp.Matrix([[1 if i==11 else 0] for i in range(12)])
print("e10 observable?", e10 in P_sym.columnspace())
print("e12 observable?", e12 in P_sym.columnspace())

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
steps = 100
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
