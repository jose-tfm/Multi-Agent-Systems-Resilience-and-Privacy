import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.utils import plot_state


# Define the network topology.
neighbors = [
    [9, 1, 4, 7, 6],        
    [0, 9, 2, 3, 8, 5, 6],   
    [3, 9, 5, 1, 6, 4],      
    [2, 9, 5, 8, 4, 6, 1],   
    [0, 6, 8, 7, 9, 1],      
    [3, 8, 7, 6, 1, 9, 4, 2],
    [5, 8, 7, 4, 1, 3, 0],   
    [8, 6, 4, 0],           
    [7, 5, 3, 6, 4, 1, 9],    
    [2, 0, 3, 1, 5, 8, 6, 4]  
]
'''
neighbors = [
    [0, 1, 2, 3],       
    [0, 1, 2, 3],   
    [0, 1, 2, 3],
    [0, 1, 2, 3],     
]'''


# Define attacked agent values.
att_v = {0: (lambda t: 0.3), 8: (lambda t: 0.8)}
#att_v = {0: (lambda t: 0.2)}

# Initial state for agents.
x0 = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.5, 0.4, 0.6, 0.3]
#x0 = [0.2, 1, 0,  0.6]
indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_iterations = 30
f = 2


def msr_consensus_custom(x0, neighbors, max_iterations, f, attck_vals={}):
    """
    Parameters:
      - x0: Initial state values (list or numpy array).
      - neighbors: A list of lists where neighbors[i] contains the indices of agents that are neighbors of agent i.
      - max_iterations: Number of iterations to run.
      - f: Number of outliers to remove from each agent's neighborhood (from both ends of the sorted values).
      - attck_vals: Dictionary with attacked agents as keys and functions that define the faulty values over time.
    
    Returns:
      - x_vals: List of state vectors over the iterations.
    """
    num_agents = len(x0)
    x_vals = [np.array(x0)]
    
    print("Initial State:", x0)
    print("Lista de vizinhos para cada agente:", neighbors)
    
    # Set initial state for attacked agents.
    for a in attck_vals:
        x_vals[-1][a] = attck_vals[a](0)
    
    for k in range(max_iterations):
        x_new = np.zeros(num_agents)
        print(f"\nIteração {k+1}:")
        for i in range(num_agents):
            inds = neighbors[i][:]
            if i not in inds:
                inds.append(i)
            
            # Collect values from neighbors.
            vals = np.array([x_vals[-1][j] for j in inds])
            weights = np.ones(len(inds)) / len(inds)
            
            print(f"  Agente {i}:")
            print(f"    Índices dos vizinhos incluindo self: {inds}")
            print(f"    Valores: {vals}")
            print(f"    Pesos: {weights}")
            
            # Sort values and corresponding weights.
            sort_order = np.argsort(vals)
            vals_sorted = vals[sort_order]
            weights_sorted = weights[sort_order]
            print(f"    Valores ordenados: {vals_sorted}")
            print(f"    Pesos ordenados: {weights_sorted}")
            
            # Filter out the f lowest and f highest values (if possible).
            if f and len(vals_sorted) > 2 * f:
                filtered_vals = vals_sorted[f:-f]
                filtered_weights = weights_sorted[f:-f]
                print(f"    Valores filtrados (remover os {f} menores e {f} maiores): {filtered_vals}")
            else:
                filtered_vals = vals_sorted
                filtered_weights = weights_sorted
                print(f"    Valores filtrados: {filtered_vals}")
            
            # Update state by computing the weighted average.
            if np.sum(filtered_weights) > 0:
                x_new[i] = np.sum(filtered_vals * filtered_weights) / np.sum(filtered_weights)
            else:
                x_new[i] = np.mean(filtered_vals)
            print(f"    Atualização do estado: {x_new[i]}")
        
        # Force attacked agents to follow their faulty values.
        for a in attck_vals:
            x_new[a] = attck_vals[a](k+1)
            print(f"    Agente atacado {a} forçado a: {x_new[a]}")
        
        print(f"  Estado na iteração {k+1}: {x_new}")
        x_vals.append(x_new.copy())
    
    return x_vals


# Run the consensus algorithm.
r = msr_consensus_custom(x0, neighbors, max_iterations, f, attck_vals=att_v)

# Plotting the evolution of states.
states = np.array(r)
iterations = np.arange(len(r))
correct_consensus = np.average(x0)

plot_state(states, correct_consensus, att_v, indices, title="Evolution of W-MSR", xlabel="Iteration", ylabel="State Value")

