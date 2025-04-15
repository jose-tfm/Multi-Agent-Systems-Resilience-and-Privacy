import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils import plot_state


max_iter = 30
f = 2
epsilon = 0.1
initial_states = [2, 4.8, 3.7, 3.5, 2.8, 2.5, 0.5, 4.7, 1, 3.6]
attacked = [True, False, False, False, False, False, False, False, True, False]
att_v = {0: 2, 8: 1}

neighbors = [
    [9, 1, 4, 7, 6],      
    [0, 9, 2, 3, 8, 5, 6], 
    [3, 9, 5, 1, 6, 4],
    [2, 9, 5, 8, 4, 6, 1],
    [0, 6, 8, 7, 9, 1],
    [3, 8, 7, 6, 1, 9, 4, 2],
    [5, 8, 7, 4, 1, 3, 0],
    [8, 6, 4, 0, 5],
    [7, 5, 3, 6, 4, 1, 9],
    [2, 0, 3, 1, 5, 8, 6, 4]  
]



def min_f(values, f):
    distinct_vals = sorted(set(values))
    if len(distinct_vals) <= 1:
        return distinct_vals[0]
    # Remove the maximum
    distinct_vals_no_max = distinct_vals[:-1]
    # If you have enough values left, skip f from the front
    if len(distinct_vals_no_max) > f:
        return distinct_vals_no_max[f]
    else:
        #if we can't skip f
        return distinct_vals_no_max[-1]


def update_reputations(x, c_old, neighbors, iteration, f, epsilon):
    """
    For each agent i and each neighbor j, compute new reputation:
      1) Raw reputation: ~c_ij = 1 - (1/|N_i|) * sum_{v in N_i} |x_j - x_v|
      2) Normalize raw values (ignoring up to f extreme outliers) to [0,1].
      3) Apply confidence factor: if normalized value is zero, assign epsilon^(iteration+1).
    """
    N = len(x)
    c_next = np.zeros((N, N))
    for i in range(N):
        Ni = neighbors[i]
        raw = np.zeros(N)
        for j in Ni:
            print("Começa aqui")
            dist_sum = 0
            for v2 in Ni:
                diff = abs(x[j] - x[v2])
                dist_sum += diff
                print(f"  -> Agente i={i}, calcular para j={j}, vizinho={v2}, valor_j={x[j]}, Valor_vizinho={x[v2]}, diff={diff}")
            avg_dist = dist_sum / len(Ni)
            raw[j] = 1 - avg_dist
            print(f"Agente {i}, vizinho j={j}, soma total={dist_sum}, media={avg_dist}, raw={raw[j]}")
        raw_vals = [raw[j] for j in Ni]
        max_raw = max(raw_vals)
        min_raw = min_f(raw_vals, f)
        print("Iteração", iteration, "Agent", i)
        print(" Raw values:", raw_vals)
        print(" Max_raw:", max_raw)
        print(" Min_raw:", min_raw)
        denom = max_raw - min_raw if (max_raw - min_raw) != 0 else 1e-9
        for j in Ni:
            if i == j:
                c_norm = 1.0  
            else:
                c_norm = (raw[j] - min_raw) / denom
                c_norm = max(0, min(c_norm, 1))
            if c_norm > 0:
                c_next[i, j] = c_norm
                print(f"    Para j = {j}: c_norm = {c_norm:.4f} > 0, então c_next[{i},{j}] = {c_norm:.4f}")
            else:
                c_next[i, j] = epsilon**(iteration+1)
                print(f"    Para j = {j}: c_norm = {c_norm:.4f} <= 0, então c_next[{i},{j}] = epsilon^(k+1) = {epsilon**(iteration+1):.4e}")
    return c_next

def update_consensus(x, c, neighbors):
    """
    Each agent i updates its state as a weighted average:
      x_i^(k+1) = (sum_{j in N_i} c[i,j]*x[j]) / (sum_{j in N_i} c[i,j])
    """
    N = len(x)
    x_next = np.zeros(N)
    for i in range(N):
        Ni = neighbors[i]
        numerator = 0.0
        denominator = 0.0
        for j in Ni:
            numerator += c[i, j] * x[j]
            denominator += c[i, j]
        if denominator < 1e-9:
            x_next[i] = x[i]
        else:
            x_next[i] = numerator / denominator
    return x_next

def run_reputation_simulation(initial_states, attacked, max_iter, f, epsilon, att_v, neighbors):
    """
    Runs the reputation-based consensus simulation.
    
    Parameters:
      - initial_states: list or numpy array of initial state values for each agent.
      - attacked: list of booleans indicating attacked nodes (True = attacked).
      - max_iter: number of iterations.
      - f: parameter for outlier removal.
      - epsilon: confidence factor.
      - forced_values: dictionary mapping index of an attacked agent to its forced value.
      - neighbors: list of lists defining neighbors for each agent. If None, a complete graph (except self) is assumed.
    
    Returns:
      - all_states: numpy array of state vectors over iterations.
    """
    N = len(initial_states)
    x = np.array(initial_states, dtype=float)
    
    # Use provided neighbors or create a complete graph (each agent sees all others except itself).
    if neighbors is None:
        neighbors = [[j for j in range(N) if j != i] for i in range(N)]
    
    # Initialize reputation matrix: each agent initially trusts its neighbors equally (weight 1).
    c = np.zeros((N, N))
    for i in range(N):
        for j in neighbors[i]:
            c[i, j] = 1.0
            
    all_states = [x.copy()]
    # Determine indices of attacked agents.
    attacker_indices = [i for i, att in enumerate(attacked) if att]
    
    for iteration in range(max_iter):
        c_new = update_reputations(x, c, neighbors, iteration, f=f, epsilon=epsilon)
        x_next = update_consensus(x, c_new, neighbors)
        
        # Force attacked agents to their specified forced values.
        if att_v is not None:
            for attacker in attacker_indices:
                if attacker in  att_v:
                    x_next[attacker] = att_v[attacker]
        
        all_states.append(x_next.copy())
        x = x_next.copy()
        c = c_new.copy()
    
    return np.array(all_states)


states = run_reputation_simulation(initial_states, attacked, max_iter, f, epsilon, att_v, neighbors)
correct_consensus = np.average(initial_states)
plot_state(states, correct_consensus, att_v, title="Reputation Evolution", xlabel="Iterations", ylabel="States")
