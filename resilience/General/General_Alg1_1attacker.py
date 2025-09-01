import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.utils import plot_state

T = 8             
f = 1              
epsilon = 0.05     
all_agents = [1, 2, 3]
att_v = [3]
correct_values = [0, 1]
x_init = {1: 0.0, 2: 1.0, 3: 0.1}

def attacked_value(agent, t):
    """Attacked node (3) always broadcasts 0.1, for all subsets."""
    if agent in att_v:
        return
    else:
        return None


F_subsets = []
for size in range(f+1):
    for subset in itertools.combinations(all_agents, size):
        F_subsets.append(frozenset(subset))
F_subsets = sorted(F_subsets, key=lambda s: len(s))
print("Candidate Subsets F:", F_subsets)
subset_index = {S: i for i, S in enumerate(F_subsets)}


c = {u: {} for u in all_agents}
for u in all_agents:
    c[u][0] = [x_init[u]] * len(F_subsets)


x_values = {u: [x_init[u]] for u in all_agents}
adjacency = {u: all_agents.copy() for u in all_agents}


def consensus_candidate(u, t, S):
    """
    Return c_u^(t+1)[S] by averaging c_v^(t)[S] for v in adjacency[u]\S.
    If u is attacked, always return 0.1.
    """
    if u in att_v:
        return 0.1
    excluded = S
    included_agents = [v for v in adjacency[u] if v not in excluded]
    vals = [c[v][t][subset_index[S]] for v in included_agents]
    if len(vals) == 0:
        return c[u][t][subset_index[S]]
    return np.mean(vals)


def select_state(u, t):
    """
    For a normal agent:
      - Print the candidate difference for every candidate subset S (even those that exclude u).
      - Then, among the candidates that do NOT exclude u, if exactly one candidate has an
        absolute difference from the full candidate (S = ∅) >= epsilon, return that candidate's value.
      - Otherwise, return the full candidate.
    For an attacked agent, always return 0.1.
    """
    if u in att_v:
        return 0.1
    val_full = c[u][t][0]
    for idx, S in enumerate(F_subsets):
        cand_val = c[u][t][idx]
        diff = abs(cand_val - val_full)
        print(f"Agent {u} at t={t}: For candidate S={S}, candidate = {cand_val:.2f}, full = {val_full:.2f}, diff = {diff:.2f}")
    
    valid_candidates = []
    for idx, S in enumerate(F_subsets):
        diff = abs(c[u][t][idx] - val_full)
        if diff > epsilon: 
            valid_candidates.append(idx)
    
    if len(valid_candidates) == 1:
        print(f"Agent {u} at t={t}: Unique valid candidate found: {F_subsets[valid_candidates[0]]}")
        return c[u][t][ valid_candidates[0] ]
    else:
        print(f"Agent {u} at t={t}: No unique valid candidate, using full candidate")
        return val_full


for t in range(T):
    for u in all_agents:
        c[u][t+1] = [None] * len(F_subsets)
    for u in all_agents:
        for S in F_subsets:
            idxS = subset_index[S]
            c[u][t+1][idxS] = consensus_candidate(u, t, S)
    
    print(f"\nTime t={t+1}:")
    for u in all_agents:
        cand_str = ", ".join([f"{(val):.2f}" for val in c[u][t+1]])
        print(f"  Agent {u} candidate vector: [{cand_str}]")
    
    for u in all_agents:
        x_next = select_state(u, t+1)
        x_values[u].append(x_next)
    
    print(f"  Final selected states at t={t+1}:")
    for u in all_agents:
        print(f"    Agent {u}: x = {(x_values[u][-1]):.2f}")


states = np.array([x_values[u] for u in all_agents]).T
correct_consensus = np.average(correct_values)
plot_state(states, correct_consensus, att_v, all_agents, title="General Resilient Alg 1 Evolution", xlabel="Iterations", ylabel="states" )
