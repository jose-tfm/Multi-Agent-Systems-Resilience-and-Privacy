import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, eig, matrix_rank

A = np.array([[1, 0, 0],
              [0, 1, 1],
              [0, 0, 1]])
B = np.array([[0, 1],
              [1, 1],
              [1, 0]])
C = np.array([[0, 1, -1],
              [1, 0, 0]])
D = np.array([[0, 1],
              [0, 1]])

n = A.shape[0]  
m = B.shape[1]   
p = C.shape[0]  


def compute_O_L(C, A, L):
    """Compute the extended observability matrix O_L = [C; CA; ...; CA^L]."""
    O_list = [C]
    for i in range(1, L+1):
        O_list.append(C @ np.linalg.matrix_power(A, i))
    return np.vstack(O_list)

def compute_J_L(D, C, B, A, L):
    """Compute the extended input matrix J_L recursively.
       J_0 = D, and for L>=1:
       J_L = [D, 0, ..., 0; O_{L-1}B, J_{L-1}]
    """
    if L == 0:
        return D
    else:
        J_prev = compute_J_L(D, C, B, A, L-1)
        O_prev = compute_O_L(C, A, L-1)
        top = np.hstack([D, np.zeros((D.shape[0], J_prev.shape[1]))])
        bottom = np.hstack([O_prev @ B, J_prev])
        return np.vstack([top, bottom])


def find_delay_L(D, C, B, A, m, L_max=None):
    """
    Find the smallest delay L (starting from 1) such that:
      rank(J_L) - rank(J_{L-1}) == m.
    """
    if L_max is None:
        L_max = A.shape[0]
    for L in range(1, L_max+1):
        J_L_val = compute_J_L(D, C, B, A, L)
        J_Lm1_val = compute_J_L(D, C, B, A, L-1)
        if matrix_rank(J_L_val) - matrix_rank(J_Lm1_val) == m:
            return L
    return None


L_found = find_delay_L(D, C, B, A, m, L_max=10)
print("Found Delay L =", L_found)
L = L_found


O2 = compute_O_L(C, A, L)
J2 = compute_J_L(D, C, B, A, L)
print("Extended Observability O2 =\n", O2)
print("Extended Input Matrix J2 =\n", J2)


M = np.hstack([B, np.zeros((B.shape[0], J2.shape[1] - B.shape[1]))])
F = M @ pinv(J2)
print("Observer Gain F =\n", F)


E = A - F @ O2
print("Observer Matrix E =\n", E)


eigvals, _ = eig(E)
print("Eigenvalues of E =", eigvals)


Result_E = E - A + F @ O2
print("E + F*O2 - A (should be close to zero):\n", Result_E)
Result_F = F @ J2
print("F*J2 (should approximate [B 0 ... 0]):\n", Result_F)


num_agents = 4
num_steps = 15

def u_func(k, agent_index):

    if agent_index == 2:
        return np.array([0.5, -0.8])
    else:
        return np.array([0.0, 0.0])


x_true = np.zeros((n, num_steps, num_agents))
x_hat  = np.zeros((n, num_steps, num_agents))
y_meas = np.zeros((p, num_steps, num_agents))


initial_states = [np.array([1, 0, -1]),
                  np.array([1.2, 0.1, -1.1]),
                  np.array([0.5, 0.0, 2.0]), 
                  np.array([1.1, -0.1, -0.9])]
for i in range(num_agents):
    x_true[:, 0, i] = initial_states[i]
    x_hat[:, 0, i] = np.zeros(n)


for i in range(num_agents):
    for k in range(num_steps - 1):
        x_true[:, k+1, i] = A @ x_true[:, k, i] + B @ u_func(k, i)
        y_meas[:, k, i] = C @ x_true[:, k, i] + D @ u_func(k, i)
    y_meas[:, num_steps-1, i] = C @ x_true[:, num_steps-1, i] + D @ u_func(num_steps-1, i)


def get_y_stack(y, k, L):
    stack = []
    for j in range(L+1):
        if k + j < y.shape[1]:
            stack.append(y[:, k+j])
        else:
            stack.append(np.zeros(p))
    return np.concatenate(stack)


for i in range(num_agents):
    for k in range(num_steps - L):
        y_stack = get_y_stack(y_meas[:, :, i], k, L)
        x_hat[:, k+1, i] = E @ x_hat[:, k, i] + F @ y_stack


valid_steps = num_steps - L
error = x_hat[:, :valid_steps, :] - x_true[:, :valid_steps, :]


BD = np.vstack([B, D])  
G = pinv(BD)            
print("Matrix G =\n", G)

# Reconstruct the unknown input:
u_hat = np.zeros((m, valid_steps, num_agents))
for i in range(num_agents):
    for k in range(valid_steps - 1):
        vec = np.concatenate([
            x_hat[:, k+1, i] - A @ x_hat[:, k, i],
            y_meas[:, k, i] - C @ x_hat[:, k, i]
        ])
        u_hat[:, k, i] = G @ vec


time = np.arange(valid_steps)
plt.figure(figsize=(10, 6))
for i in range(num_agents):
    err_norm = np.linalg.norm(error[:, :, i], axis=0)
    plt.plot(time, err_norm, label=f"Agent {i+1}")
plt.xlabel("Time step k")
plt.ylabel("||e[k]||")
plt.title("Estimation Error Norm for Each Agent")
plt.legend()
plt.grid(True)
plt.show()


fig, axs = plt.subplots(num_agents, 3, figsize=(15, 12), sharex=True)
for i in range(num_agents):
    for j in range(n):
        axs[i, j].plot(time, x_true[j, :valid_steps, i], 'ko-', label='True')
        axs[i, j].plot(time, x_hat[j, :valid_steps, i], 'r--o', label='Estimated')
        axs[i, j].set_ylabel(f"x[{j}]")
        axs[i, j].grid(True)
        if i == 0:
            axs[i, j].set_title(f"State x[{j}]")
    axs[i, 0].set_ylabel(f"Agent {i+1}\n x[0]")
plt.xlabel("Time step k")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(num_agents, m, figsize=(10, num_agents * 3), sharex=True)
for i in range(num_agents):
    for j in range(m):
        axs[i, j].plot(np.arange(valid_steps-1), u_hat[j, :valid_steps-1, i], 'b-o')
        axs[i, j].set_title(f"Agent {i+1}, Channel u[{j}]")
        axs[i, j].set_xlabel("Time step k")
        axs[i, j].set_ylabel(f"u[{j}]")
        axs[i, j].grid(True)
plt.tight_layout()
plt.show()
