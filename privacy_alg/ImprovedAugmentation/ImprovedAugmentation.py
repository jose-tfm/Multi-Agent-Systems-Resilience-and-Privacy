import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.linalg import eig, eigvals
from scipy.optimize import minimize

# --------------------------
# Utilities
# --------------------------
def row_normalize(M):
    """Faz cada linha somar 1, evitando divisões por zero."""
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap_weights_hetero(N, A, w):
    """
    Constrói a matriz 4N×4N onde cada agente i tem 5 pesos próprios:
      w é um vetor de dimensão 5*N, indexado por
        [w01^(0),w10^(0),w02^(0),w23^(0),w30^(0),
         w01^(1),w10^(1),…,
         …,
         w30^(N-1)]
    As arestas do gadget:
      orig→aug1, aug1→orig,
      orig→aug2, aug2→aug3, aug3→orig
    """
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A  # cópia da rede original

    for i in range(N):
        # extrai os 5 pesos de w relativos ao agente i
        w01, w10, w02, w23, w30 = w[5*i:5*i+5]

        orig = i
        aug1 = N + 3*i
        aug2 = N + 3*i + 1
        aug3 = N + 3*i + 2

        Ap[orig, aug1] = w01  # orig → aug1
        Ap[aug1, orig] = w10  # aug1 → orig
        Ap[orig, aug2] = w02  # orig → aug2
        Ap[aug2, aug3] = w23  # aug2 → aug3
        Ap[aug3, orig] = w30  # aug3 → orig
        

    return row_normalize(Ap)


def consensus_rate_hetero(w):
    """
    Objetivo: segundo maior autovalor em módulo de P =
    build_Ap_weights_hetero(N,A_norm,w). Menor = mais rápido.
    w tem dimensão 5*N.
    """
    P = build_Ap_weights_hetero(N, A_norm, w)
    vals = np.abs(eigvals(P))
    vals.sort()
    return vals[-2]  # λ₂(P)


# --------------------------
# Main
# --------------------------
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
x0 = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
A_norm = row_normalize(A)

# chute inicial (todos iguais) e limites
w0 = np.ones(5*N)
bounds = [(0, 5)] * (5*N)   # cada peso ∈ [0.1, 5]

# ========== otimização interior‐point ==========
res = minimize(
    consensus_rate_hetero,
    w0,
    method="trust-constr",
    bounds=bounds,
    options={"verbose": 2, "maxiter": 200, "gtol": 1e-6}
)
w_opt = res.x
print("\nPesos ótimos W* por agente:")
for i in range(N):
    wi = w_opt[5*i:5*i+5]
    print(f" agente {i+1}: {np.round(wi,3)}")
print("λ₂(P) mínimo =", np.round(res.fun,4))

# reconstruir Ap
Ap = build_Ap_weights_hetero(N, A_norm, w_opt)

# ---- left‐eigenvector v0 ----
w_vals, V = eig(Ap.T)
idx = np.argmin(np.abs(w_vals - 1))
v0 = np.real(V[:, idx]); v0 /= v0.sum()
print("\nv₀ (left eigenvector) faz PRIVACIDADE:")
for i,val in enumerate(v0):
    if i < N:
        print(f" orig{i+1}: {val:.4f}")
    else:
        ag = (i-N)//3+1
        au = (i-N)%3+1
        print(f" aug{au}_{ag}: {val:.4f}")

# ---- inject x_p0 e simula ----
alpha, beta, gamma = np.full(N,1.4), np.full(N,1.0), np.full(N,1.0)
x_p0 = np.zeros(4*N)
for j in range(N):
    a,b,g = alpha[j],beta[j],gamma[j]
    s = a+b+g
    coeff = 4*x0[j]/s
    x_p0[j] = 0
    x_p0[N+3*j+0] = coeff*a
    x_p0[N+3*j+1] = coeff*b
    x_p0[N+3*j+2] = coeff*g
target = x0.mean()
x_p0 *= target/(v0@x_p0)

# ---- simulação e convergência ----
steps, tol = 400, 1e-4
Xo = np.zeros((N,steps+1)); Xa = np.zeros((4*N,steps+1))
Xo[:,0], Xa[:,0] = x0, x_p0
for k in range(steps):
    Xo[:,k+1] = A_norm @ Xo[:,k]
    Xa[:,k+1] = Ap     @ Xa[:,k]
print(f'Matrix Ap = \n {Ap}')
oc = next((k for k in range(steps+1)
            if np.max(np.abs(Xo[:,k]-target))<tol), None)
ac = next((k for k in range(steps+1)
            if np.max(np.abs(Xa[:,k]-target))<tol), None)
print(f"\nOriginal convergiu em {oc} passos\nAugmented convergiu em {ac} passos")

# ---- plots ----
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,10))
for i in range(N):
    ax1.plot(Xo[i], label=f'$x_{{{i+1}}}[k]$')
ax1.axhline(target,ls='--',color='k',label=f'Cons={target:.2f}')
ax1.set_title('(a) Original'); ax1.legend(); ax1.grid()
for i in range(4*N):
    lbl = f'$x_{{{i+1}}}$' if i<N else f'$\\tilde x_{{{(i-N)//3+1},{(i-N)%3+1}}}$'
    ax2.plot(Xa[i],label=lbl)
ax2.axhline(target,ls='--',color='k',label=f'Cons={target:.2f}')
ax2.set_title('(b) Augmented otimizado'); ax2.legend(ncol=3,fontsize='small'); ax2.grid()
plt.tight_layout()
plt.show()
