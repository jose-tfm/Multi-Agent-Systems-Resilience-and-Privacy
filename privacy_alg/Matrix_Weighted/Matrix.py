import numpy as np
import matplotlib.pyplot as plt

# 1) simulation parameters (as in §5.1)
n, d, d_prime = 5, 3, 3
D = d + d_prime
sigma = 2
max_iter = 400
d_star = D - 1                       
bound = 1.0 / (4*(n-1)*sigma)       

# 2) topology
neighbors = {
    0: [1,2],
    1: [0,2,4],
    2: [0,1,3],
    3: [2,4],
    4: [1,3]
}

# 3) initial real states
x_real0 = np.array([[1,2,3],
                    [4,5,6],
                    [7,8,9],
                    [10,11,12],
                    [13,14,15]], float)
avg_real = x_real0.mean(axis=0)

# 4) build orthogonal basis V (App. A.1 eqs 37–42),
#    first d' coords = virtual, last d = real
V = []
v1 = np.concatenate([np.ones(d_prime), np.zeros(d)])  # v1 = (1,1,1,0,0,0)
V.append(v1)
for i in range(2, D):  # i = 2..D-1
    v = np.zeros(D)
    inv = 1.0 / i
    v[0]   = +inv
    v[1:i] = -inv
    v[i]   =  1.0     
    V.append(v)
vD = np.full(D, -1.0/D)
vD[0] = +1.0/D
V.append(vD)
print("Basis V:")
for idx, v in enumerate(V, start=1):
    print(f" v{idx} =", np.round(v,6))


# 5) static projector Φ at k=0 (eq 37), normalized
Phi = np.outer(v1, v1) / (v1 @ v1)
print(f'Phi = \n {Phi}')

# 6) lift into R^{d+d'} with uniform-[0,1] virtual init
xv0 = np.random.rand(n, d_prime)
xt  = np.hstack((xv0, x_real0))
print(f'Este é o xt = \n {xt}')


# 7)

P_edge = {}
for i in range(n):
    for j in neighbors[i]:
        if j > i:
            M = np.random.rand(D, D)
            P = M @ M.T
            P_edge[(i,j)] = P_edge[(j,i)] = P

deltas = np.zeros_like(xt)    
for i in range(n):
    total = np.zeros(D)
    for j in neighbors[i]:
        term = P_edge[(i,j)] @ (Phi @ xt[j] - Phi @ xt[i])
        total += term
    deltas[i] = sigma * total

xt += deltas

print(f' Este é o xt = \n {xt}')


# 8) main loop k=1…max_iter
traj = np.zeros((max_iter+1, n, D))
traj[0] = xt.copy()
y32   = np.zeros((max_iter+1, d))

for k in range(1, max_iter+1):

    rho    = k % d_star or d_star
    vr, vlast = V[rho-1], V[-1]

    P_vr    = np.outer(vr,      vr)   / (vr   @ vr)
    P_vl    = np.outer(vlast, vlast)/ (vlast@ vlast)

    A_edge = {}
    for i in range(n):
        for j in neighbors[i]:
            if j > i:
                γ = np.random.uniform(0, bound)
                ζ = np.random.uniform(0, bound)
                A = γ * P_vr + ζ * P_vl
                A_edge[(i,j)] = A_edge[(j,i)] = A

    A23      = A_edge[(2,1)]    
    y32[k]   = (A23 @ xt[2])[d_prime:]

    new_xt = xt.copy()
    for i in range(n):
        acc = np.zeros(D)
        for j in neighbors[i]:
            acc += A_edge[(i,j)] @ (xt[j] - xt[i])
        new_xt[i] += sigma * acc

    xt      = new_xt
    traj[k] = xt.copy()
print(f' Isto é A = \n {A}')

real_traj = traj[:, :, d_prime:]    # shape (max_iter+1, n, d)



# 10) Plot Fig.5: evolution of x_iℓ(k)
k0, ks = 0, np.arange(0, max_iter+1)
fig5, ax5 = plt.subplots(1, 3, figsize=(12,4), sharey=True)

for ℓ, ax in enumerate(ax5):
    for i in range(n):
        ax.plot(ks, real_traj[k0:, i, ℓ], label=f'Agent {i+1}')
    ax.hlines(avg_real[ℓ], k0, max_iter, 'k', '--',
              label=rf'$[\mathrm{{Avg}}(x(0))]_{{{ℓ+1}}}$')

    ax.set_title(f'({chr(97+ℓ)})')
    ax.set_xlabel('k')
    ax.set_ylabel(rf'$x_{{i{ℓ+1}}}(k)$')
    if ℓ == 0:
        ax.set_ylabel(r'$x_{i\ell}(k)$')
    ax.grid(True)
    ax.legend(ncol=2, fontsize='small', loc='upper right')
    ax.yaxis.set_tick_params(labelleft=True)


fig5.suptitle('Fig.5: Evolution of real agent states')
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()



# 11) Plot Fig.6: x₃(k) vs y₃→₂(k)
fig6, ax6 = plt.subplots(1,3,figsize=(12,4), sharey=True)
for ℓ, ax in enumerate(ax6):
    ax.plot(ks, real_traj[k0:, 2, ℓ],
            label=rf'$[x_3(k)]_{{{ℓ+1}}}$')
    ax.plot(ks, y32[k0:,    ℓ],
            label=rf'$[y_{{3\to2}}(k)]_{{{ℓ+1}}}$')
    ax.hlines(avg_real[ℓ], ks[0], ks[-1], 'k','--',
              label=rf'$[\mathrm{{Avg}}(x(0))]_{{{ℓ+1}}}$')

    ax.set_title(f'({chr(97+ℓ)})')
    ax.set_xlabel('k')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.set_ylim(0, 15)
    ax.yaxis.set_tick_params(labelleft=True)

    # legend in each subplot:
    ax.legend(fontsize='small', loc='upper right')

fig6.suptitle('Fig.6: $x_3(k)$ vs $y_{3\\to2}(k)$')
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()