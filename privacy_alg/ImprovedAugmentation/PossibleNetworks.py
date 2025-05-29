import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import networkx as nx

def privacy_span_report(Ap, tol=1e-6):

    size = Ap.shape[0]
    N = size // 4

    safe, unsafe = {i:[] for i in range(N)}, {i:[] for i in range(N)}

    for i in range(N):
        C = np.zeros((1, size))
        C[0, i] = 1
        O = observability_matrix(Ap, C, steps=size)

        for j in range(N):
            if j == i:
                continue
            idxs = [j, N+3*j, N+3*j+1, N+3*j+2]
            v = np.zeros(size)
            v[idxs] = 1

            if in_column_space(O, v, tol=tol):
                unsafe[i].append((j, idxs))
            else:
                safe[i].append((j, idxs))

    return safe, unsafe


def observability_matrix(A, C, steps=None):

    n = A.shape[0]
    if steps is None:
        steps = n
    mats, Ak = [], np.eye(n)
    for _ in range(steps):
        mats.append(C @ Ak)
        Ak = A @ Ak
    return np.vstack(mats)

def in_column_space(O, v, tol=1e-6):

    x, residuals, *_ = np.linalg.lstsq(O, v, rcond=None)
    if residuals.size:
        return residuals[0] < tol
    return np.linalg.norm(O @ x - v) < tol

def check_privacy_condition2(Ap, tol=1e-6):

    size = Ap.shape[0]
    N = size // 4
    for i in range(N):
        C = np.zeros((1, size))
        C[0, i] = 1
        O = observability_matrix(Ap, C, steps=size)

        for j in range(N):
            if j == i:
                continue
            v = np.zeros(size)
            idxs = [j, N+3*j, N+3*j+1, N+3*j+2]
            v[idxs] = 1

            if in_column_space(O, v, tol=tol):
                print(f"Privacy FAILS for observer i={i} recovering agent j={j}")
                return False
    return True

def row_normalize(M):

    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def build_Ap(N, A):

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
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        '''
        # ——————————————————————
        # b)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
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
        # f)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 2
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # g)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # h)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # i)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[2]] = 1
        '''
        # ——————————————————————
        # j)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # k)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[1], inds[2]] = 2
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[2]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[3], inds[0]] = 1
        '''
        # ——————————————————————
        # l)
        '''
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[2], inds[1]] = 1
        Ap[inds[2], inds[3]] = 1
        '''
        # ——————————————————————
        # m)
        
        Ap[inds[0], inds[1]] = 1
        Ap[inds[1], inds[0]] = 1
        Ap[inds[0], inds[2]] = 1
        Ap[inds[2], inds[0]] = 1
        Ap[inds[3], inds[0]] = 1
        Ap[inds[0], inds[3]] = 1
        Ap[inds[1], inds[2]] = 1
        Ap[inds[3], inds[2]] = 1
        
    return row_normalize(Ap)


if __name__ == "__main__":

    N = 3
    A = np.array([[0,0.7,0.3],
                  [0.7,0,0.3],
                  [0.7,0.3 ,0]], dtype=float)
    x0 = np.array([0.5,1/3,0.2])

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
    ])
    x0    = np.array([0.1,0.3,0.6,0.43,0.85,0.9,0.45,0.11,0.06,0.51,0.13])
    '''
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


    np.set_printoptions(precision=3, suppress=True)
    print("Augmented A^P =\n", Ap, "\n")


    vals, vecs = eig(Ap.T)
    v0 = np.real(vecs[:, np.argmin(np.abs(vals-1))])
    v0 /= v0.sum()
    print("Left eigenvector v0 =", np.round(v0,3), "\n")

    alpha = beta = gamma = np.ones(N)
    x_p0 = np.zeros(4*N)
    for j in range(N):
        s = alpha[j]+beta[j]+gamma[j]
        coeff = 4*x0[j]/s
        x_p0[j] = 0
        x_p0[N+3*j  ] = coeff*alpha[j]
        x_p0[N+3*j+1] = coeff*beta[j]
        x_p0[N+3*j+2] = coeff*gamma[j]
    target = x0.mean()
    x_p0 *= target/(v0 @ x_p0)

    steps = 400
    tol = 1e-4
    X_orig = np.zeros((N, steps+1))
    X_aug  = np.zeros((4*N, steps+1))
    X_orig[:,0] = x0
    X_aug[:, 0] = x_p0

    for k in range(steps):
        X_orig[:,k+1] = A @ X_orig[:,k]
        X_aug[:,k+1]  = Ap @ X_aug[:,k]

    orig_conv = next((k for k in range(steps+1)
                      if np.max(np.abs(X_orig[:,k]-target))<tol), None)
    aug_conv  = next((k for k in range(steps+1)
                      if np.max(np.abs(X_aug[:,k] -target))<tol), None)

    consensus_orig = X_orig[:, orig_conv]
    consensus_aug  = X_aug[:,  aug_conv]

    print(f"\nOriginal convergiu em {orig_conv} steps")
    print(" → Estado final (original):", np.round(consensus_orig, 4))

    print(f"Augmented convergiu em {aug_conv} steps")
    print(" → Estado final (augmented):", np.round(consensus_aug, 4))

        
    # 8) plot
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,8))
    for i in range(N):
        ax1.plot(X_orig[i], label=f'$x_{{{i+1}}}$')
    ax1.axhline(target, ls='--', color='k')
    ax1.set_title('Original')
    ax1.legend(); ax1.grid()

    for i in range(4*N):
        lbl = f'$x_{{{i+1}}}$' if i<N else f'$\~x_{{{(i-N)//3+1},{(i-N)%3+1}}}$'
        ax2.plot(X_aug[i], label=lbl)
    ax2.axhline(target, ls='--', color='k')
    ax2.set_title('Augmented')
    ax2.legend(ncol=2, fontsize='small'); ax2.grid()

    plt.tight_layout()
    plt.show()