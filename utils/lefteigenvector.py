import numpy as np 
import math
from scipy.linalg import eig

A = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
], dtype=float)

P = A / A.sum(axis=1, keepdims=True)
w, v_left = eig(P, left=True, right=False)
k = np.argmin(np.abs(w - 1)) #eigenvalue 1
v = v_left[:, k].real
v /= v.sum()
print(v, '\n')
print((np.abs(w - 1)))

# (3) Pega os módulos
abs_w = np.abs(w)

# (4) Ordena em ordem crescente
abs_sorted = np.sort(abs_w)

# (5) O último é 1 (autovalor dominante); o penúltimo é o segundo maior
second_largest_mag = abs_sorted[-2]
print("Segundo maior módulo de autovalor:", second_largest_mag)

# (Opcional) Se quiser o autovalor (complexo) em si:
idx_desc = np.argsort(-abs_w)   # índices ordenados decrescentemente por módulo
second_idx = idx_desc[1]        # índice do segundo maior
second_eig = w[second_idx]
print("Segundo autovalor (complexo):", second_eig)