import numpy as np
from numpy.linalg import svd

# --- build Ap (Algorithm 3 raw) --------------------------------------

def build_Ap_bidir_raw(N, A):
    size = 5*N
    Ap = np.zeros((size,size))
    Ap[:N,:N] = A/2
    for i in range(N):
        b = N + 4*i
        i1,i2,i3,i4 = b,b+1,b+2,b+3
        Ap[i,  i1] = 1/12; Ap[i,  i2] = 1/8
        Ap[i,  i3] = 1/4;  Ap[i,  i4] = 1/24
        Ap[i1,i]    = 1/11;Ap[i2,i]    = 1/2
        Ap[i3,i]    = 3/4; Ap[i4,i]    = 1/16
        Ap[i2,i1]   = 1/2; Ap[i3,i1]   = 1/4
        Ap[i4,i1]   = 15/16;Ap[i1,i2]  = 3/22
        Ap[i1,i3]   = 1/11;Ap[i1,i4]   = 15/22
    return Ap

# --- build C_i for PBH@λ=0 ------------------------------------------

def build_Ci_for_proof(i, N):
    Ci = np.zeros((N+2, 5*N))
    Ci[:N, :N] = np.eye(N)
    base = N + 4*i
    Ci[N,   base+2] = 1
    Ci[N+1, base+3] = 1
    return Ci

# --- SVD nullspace and row-space test ------------------------------

def nullspace(A, tol=1e-8):
    U,S,Vh = svd(A, full_matrices=False)
    return Vh.T[:, S<=tol]

def in_row_space(P, v, tol=1e-6):
    ns = nullspace(P)
    return np.allclose(ns.T @ v, 0, atol=tol)

# --- get left‐eigenvector from Algorithm 4 ---------------------------

def distributed_solve_alg4(N, Ap):
    size = 5*N
    s = np.zeros(size); s[0]=1
    visited,rem={0},set(range(1,N))
    while rem:
        for j in list(rem):
            for i in visited:
                if Ap[i,j]>0 and Ap[j,i]>0:
                    s[j]=s[i]*Ap[i,j]/Ap[j,i]
                    visited.add(j); rem.remove(j)
                    break
            else:
                continue
            break
    for i in range(N):
        si,base = s[i], N+4*i
        s[base+0],s[base+1],s[base+2],s[base+3] = (
            11/12*si, 1/4*si, 1/3*si, 2/3*si
        )
    return s / s.sum()

# --- main -------------------------------------------------------------

if __name__=="__main__":
    N = 3
    A = np.array([[0,0.5,0.5],
                  [0.5,0,0.5],
                  [0.5,0.5,0]], float)

    Ap = build_Ap_bidir_raw(N, A)
    vL = distributed_solve_alg4(N, Ap)

    print("\n--- Privacy tests for each agent i ---\n")

    for i in range(N):
        print(f">>> Agent i={i+1} observes:")
        Ci   = build_Ci_for_proof(i, N)
        P0_i = np.vstack([Ci, Ci @ Ap])

        for j in range(N):
            if j == i: continue

            # 1) raw sum of j's slots
            v_sum = np.zeros(5*N)
            v_sum[N+4*j-3 : N+4*j+1] = 1
            ok_sum = in_row_space(P0_i, v_sum)
            print(f"  j={j+1} raw‐sum observable? {ok_sum}", end='')
            if not ok_sum:
                print(f"  -- test vector v_sum = {v_sum}")
            else:
                print()

            # 2) weighted combo w_j
            wj = np.zeros(5*N)
            wj[j] = 1/vL[j]
            for k in range(N+4*j-3, N+4*j+1):
                wj[k] = 1/vL[k]
            ok_w = in_row_space(P0_i, wj)
            print(f"       weighted‐w_j observable? {ok_w}", end='')
            if not ok_w:
                print(f"  -- test vector w_j = {wj}")
            else:
                print()

        print()
