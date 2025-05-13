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
def row_normalize(M):
    """Normalização row-stochastic (evita divisão por zero)."""
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

def build_Ap_homo(N, A, w):
    """
    Constrói a matriz 4N×4N com *os mesmos* pesos w = [w01, w10, w02, w23, w30]
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
        
        Ap[inds[0], inds[1]] = w[0]
        Ap[inds[1], inds[0]] = w[1]
        Ap[inds[0], inds[3]] = w[2]
        Ap[inds[3], inds[2]] = w[3] 
        Ap[inds[2], inds[0]] = w[4]
        
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

def consensus_rate_homo(w):

    P = build_Ap_homo(N, A_norm, w)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]

# Definição da rede base
N = 3
A = np.array([
    [0,   0.5, 0.5],
    [0.5, 0,   0.5],
    [0.5, 0.5, 0  ]
], dtype=float)
A_norm = row_normalize(A)
x0 = np.array([0.5, 1/3, 0.2])

# 
w0 = np.ones(5)
bounds = [(0.1, 5)] * 5

#
res = minimize(
    consensus_rate_homo,
    w0,
    method="trust-constr",
    bounds=bounds
)
w_opt = res.x


print("Pesos homogêneos ótimos (w*):", np.round(w_opt, 4))
print("λ₂(P) mínimo =", np.round(res.fun, 4), "\n")


Ap = build_Ap_homo(N, A_norm, w_opt)
print("Matriz A^P completa:\n", Ap, "\n")

# Cálculo do autovetor esquerdo v0 normalizado
w_vals, V = eig(Ap.T)
idx = np.argmin(np.abs(w_vals - 1))
v0 = np.real(V[:, idx])
v0 /= v0.sum()
print("v₀ (autovetor esquerdo):")
for i, val in enumerate(v0):
    label = f"orig{i+1}" if i < N else f"aug{(i-N)//3+1}_{(i-N)%3+1}"
    print(f"  {label}: {val:.4f}")
print("Left eigenvector v0 =", np.round(v0,3), "\n")


alpha = np.ones(N)
beta  = np.ones(N)
gamma = np.ones(N)

x_p0 = np.zeros(4*N)
for j in range(N):
    a, b, g = alpha[j], beta[j], gamma[j]
    coeff = 4 * x0[j] / (a + b + g)
    x_p0[N+3*j : N+3*j+3] = [coeff*a, coeff*b, coeff*g]
target = x0.mean()
x_p0 *= target / (v0 @ x_p0)

steps, tol = 100, 1e-4
Xo = np.zeros((N, steps+1)); Xa = np.zeros((4*N, steps+1))
Xo[:,0], Xa[:,0] = x0, x_p0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]
    Xa[:,k+1] = Ap     @ Xa[:,k]

orig_conv = next(k for k in range(steps+1)
                 if np.max(np.abs(Xo[:,k] - target)) < tol)
aug_conv  = next(k for k in range(steps+1)
                 if np.max(np.abs(Xa[:,k] - target)) < tol)

consensus_orig = Xo[:, orig_conv]
consensus_aug  = Xa[:,  aug_conv]

print(f"\nOriginal convergiu em {orig_conv} steps")
print(" → Estado final (original):", np.round(consensus_orig, 4))

print(f"Augmented convergiu em {aug_conv} steps")
print(" → Estado final (augmented):", np.round(consensus_aug, 4))


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10))
for i in range(N):
    ax1.plot(Xo[i], label=f'$x_{{{i+1}}}[k]$')
ax1.axhline(target, ls='--', color='k', label=f'$\\bar x$={target:.2f}')
ax1.set_title('(a) Consenso na rede original')
ax1.legend(); ax1.grid()

for i in range(4*N):
    lbl = f'$x_{{{i+1}}}$' if i<N else f'$\\tilde x_{{{(i-N)//3+1},{(i-N)%3+1}}}$'
    ax2.plot(Xa[i], label=lbl)
ax2.axhline(target, ls='--', color='k', label=f'$\\bar x$={target:.2f}')
ax2.set_title('(b) Consenso na rede aumentada (pesos homogêneos)')
ax2.legend(ncol=3, fontsize='small'); ax2.grid()

plt.tight_layout()
plt.show()
