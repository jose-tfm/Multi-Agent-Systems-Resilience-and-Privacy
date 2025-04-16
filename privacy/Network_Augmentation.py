import numpy as np

def build_Ap(N, A):

    size = 4 * N
    Ap = np.zeros((size, size))
    A_mat = np.array(A).reshape((N, N))
    Ap[0:N, 0:N] = A_mat

    for i0 in range(N):
        i1 = i0 + 1  
        # rules
        aug1 = N + 3*i1 - 3   
        aug2 = N + 3*i1 - 2   
        aug3 = N + 3*i1 - 1   

        Ap[i0, aug1] = 2     
        Ap[aug1, i0] = 1    
        Ap[i0, aug2] = 1    
        Ap[aug2, aug3] = 1   
        Ap[aug3, i0] = 1    

    return Ap

def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    epsilon = 1e-12
    return matrix / (row_sums + epsilon)

# Example usage:
if __name__ == "__main__":
    N = 3

    A = [0,1,1,
         1,0,1,
         1,1,0]
    
    Ap = build_Ap(N, A)
    print("Unnormalized A^P:")
    print(np.around(Ap, 2))
    
    Ap_norm = row_normalize(Ap)
    print("\nRow-normalized A^P:")
    print(np.around(Ap_norm, 2))

