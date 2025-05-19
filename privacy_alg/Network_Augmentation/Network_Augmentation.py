import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

def row_normalize(M):
    """Make each row of M sum to 1 (row-stochastic)."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N, A):

    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        aug1 = N + 3*i
        aug2 = N + 3*i + 1
        aug3 = N + 3*i + 2
        Ap[i, aug1] = 1
        Ap[aug1, i] = 1
        Ap[i, aug2] = 2
        Ap[aug2, aug3] = 1
        Ap[aug3, i] = 1
    return row_normalize(Ap)

# Parameters
N = 3
A = np.array([
    [0,   0.5, 0.5],
    [0.5, 0,   0.5],
    [0.5, 0.5, 0  ]
], dtype=float)
x0 = np.array([0.5, 1/3, 0.2])


# Build and normalize A^P
Ap = build_Ap(N, A)

# Compute left-eigenvector v0 for A^P
w, V = np.linalg.eig(Ap.T)
idx = np.argmin(np.abs(w - 1))
v0 = np.real(V[:, idx])
v0 = v0 / v0.sum()
abs_w = np.abs(w)
idx_desc = np.argsort(-abs_w)
second_idx = idx_desc[1]
second_largest_mod = abs_w[second_idx]
second_eig = w[second_idx]
print("2º maior módulo de autovalor:", second_largest_mod)


#  Inject initial conditions with alpha, beta, gamma 
alpha = np.full(N, 1.0)
beta  = np.full(N, 1.0)
gamma = np.full(N, 1.0)

x_p0 = np.zeros(4*N)
for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    s = a + b + g
    coeff = 4 * x0[j] / s

    # original slot stays zero
    x_p0[j] = 0.0

    # fill the three privacy slots
    x_p0[N+3*j+0] = coeff * a    
    x_p0[N+3*j+1] = coeff * b    
    x_p0[N+3*j+2] = coeff * g    

# Global rescale so that v0^T x_p0 = average(x0)
target = x0.mean()
current = v0 @ x_p0
x_p0 *= (target / current)


print("Augmented A:", Ap)
print("Left_Eigenvector:", v0)
print("Augmented initial x_p0:", x_p0)


# ---- Simulate both dynamics ----
steps = 200
X_orig = np.zeros((N, steps+1))
X_aug  = np.zeros((4*N, steps+1))
X_orig[:,0] = x0
X_aug[:,0]  = x_p0

for k in range(steps):
    X_orig[:,k+1] = A @ X_orig[:,k]
    X_aug[:,k+1]  = Ap     @ X_aug[:,k]

# ---- Plotting ----
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10))

# (a) Original states
for i in range(N):
    ax1.plot(X_orig[i], label=f'$x_{{{i+1}}}[k]$')
ax1.axhline(target, color='k', lw=1, label=f'Consensus={target:.2f}')
ax1.set_title('(a) Consensus on $G(A)$')
ax1.set_xlabel('Step k')
ax1.set_ylabel('Value')
ax1.legend(loc='upper right')
ax1.grid()

# (b) Augmented states
for i in range(4*N):
    if i < N:
        lbl = f'$x_{{{i+1}}}[k]$'
    else:
        agent = (i-N)//3 + 1
        aug   = (i-N)%3 + 1
        lbl   = f'$\\tilde x_{{{agent},{aug}}}[k]$'
    ax2.plot(X_aug[i], label=lbl)
ax2.axhline(target, color='k', lw=1, label=f'Consensus={target:.2f}')
ax2.set_title('(b) Consensus on $G(A^P)$')
ax2.set_xlabel('Step k')
ax2.set_ylabel('Value')
ax2.legend(ncol=3, loc='upper right', fontsize='small')
ax2.grid()

plt.tight_layout()
plt.show()
