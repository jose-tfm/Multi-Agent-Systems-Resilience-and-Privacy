import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

# ------------------ utilitários ------------------
def row_normalize(M):
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N, A_norm, w):
    """Monta A^P parametrizada por w=[w01,w10,w02,w23,w30]."""
    size = 4*N
    Ap = np.zeros((size, size))
    Ap[:N,:N] = A_norm
    for i in range(N):
        orig = i
        aug1 = N+3*i
        aug2 = N+3*i+1
        aug3 = N+3*i+2
        w01, w10, w02, w23, w30 = w
        Ap[orig, aug1] = w01
        Ap[aug1, orig] = w10
        Ap[orig, aug2] = w02
        Ap[aug2, aug3] = w23
        Ap[aug3, orig] = w30
    return row_normalize(Ap)

def lambda2_for_w(w):
    P = build_Ap(N, A_norm, w)
    vals = np.abs(eigvals(P))
    return np.sort(vals)[-2]

# ----------- configuração do problema ------------
N = 3
A = np.array([[0,0.5,0.5],
              [0.5,0,0.5],
              [0.5,0.5,0]], float)
A_norm = row_normalize(A)

# chute inicial
w0 = np.ones(5)

# 1) varrendo cada peso individualmente
sweep = np.linspace(0.1, 5, 200)
fig, axes = plt.subplots(5, 1, figsize=(6, 18), tight_layout=True)

for idx, ax in enumerate(axes):
    lambdas = []
    for val in sweep:
        w = w0.copy()
        w[idx] = val
        lambdas.append(lambda2_for_w(w))
    ax.plot(sweep, lambdas, lw=1)
    ax.axvline(1.0, ls='--', color='gray', label='w=1 baseline')
    ax.set_title(f'Variação de $\\lambda_2$ vs peso $w_{{{idx}}}$')
    ax.set_xlabel(f'$w_{{{idx}}}$')
    ax.set_ylabel('$\\lambda_2$')
    ax.grid(True)
    ax.legend()

plt.show()

# 2) mapa de contorno para a combinação (w01, w10)
w01_vals = np.linspace(0.1,5,100)
w10_vals = np.linspace(0.1,5,100)
Lambda2 = np.zeros((len(w01_vals), len(w10_vals)))

for i,w01 in enumerate(w01_vals):
    for j,w10 in enumerate(w10_vals):
        w = w0.copy()
        w[0] = w01
        w[1] = w10
        Lambda2[i,j] = lambda2_for_w(w)

W01, W10 = np.meshgrid(w01_vals, w10_vals, indexing='ij')
plt.figure(figsize=(6,5))
cs = plt.contourf(W01, W10, Lambda2, levels=30, cmap='viridis')
plt.colorbar(cs, label='$\\lambda_2$')
plt.xlabel('$w_{01}$')
plt.ylabel('$w_{10}$')
plt.title('Contorno de $\\lambda_2$ em função de $w_{01},w_{10}$')
plt.show()
