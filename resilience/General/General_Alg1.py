import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from utils.utils import plot_state

T = 25      
f = 2              
epsilon = 0.05     
all_agents = [1, 2, 3, 4, 5]
x_init = {1: 14.0, 2: -5.0, 3: 6.0, 4: 2.0, 5: 8.0}


att_v = {
    3: lambda t: 7 - 2**(-0.3 * (t+1)),
    5: lambda t: 7 + 2**(-0.3 * (t+1))
}

def normal_update(u, t, S):
    """
    For a normal node, update candidate value for subset S by averaging the previous candidate
    values from all nodes in adjacency[u] that are NOT in S.
    """
    included_agents = [v for v in adjacency[u] if v not in S]
    vals = [ c[v][t][subset_index[S]] for v in included_agents ]
    if len(vals) == 0:
        return c[u][t][subset_index[S]]
    return np.mean(vals)

def adversary_update(u, t, S):
    if u == 3:
        return 7 - 2**(-0.3 * (t+1))
    elif u == 5:
        return 7 + 2**(-0.3 * (t+1))
    else:
        return normal_update(u, t, S)

update_func = {}
for u in all_agents:
    if u in [3, 5]:
        update_func[u] = adversary_update
    else:
        update_func[u] = normal_update

F_subsets = []
for size in range(f+1):
    for subset in itertools.combinations(all_agents, size):
        F_subsets.append(frozenset(subset))
F_subsets = sorted(F_subsets, key=lambda s: len(s))
print("Candidate Subsets F:", F_subsets)
subset_index = {S: i for i, S in enumerate(F_subsets)}


c = { u: {} for u in all_agents }
for u in all_agents:
    c[u][0] = [ x_init[u] ] * len(F_subsets)


x_values = { u: [ x_init[u] ] for u in all_agents }
adjacency = { u: all_agents.copy() for u in all_agents }

def consensus_candidate(u, t, S):
    """
    Return c_u^(t+1)[S] by calling the update function assigned to node u.
    (The update function takes care of whether to act adversarially or normally.)
    """
    return update_func[u](u, t, S)

def select_state(u, t):
    """
    For each node u, let val_full = c_u^(t)[∅]. Then, for every candidate subset S,
    print the candidate value, the full candidate, and the absolute difference (rounded to 2 decimals).
    Then, among candidates that do not exclude u, if exactly one candidate has an absolute difference
    from the full candidate >= epsilon, return that candidate's value.
    Otherwise, return the full candidate.
    """
    val_full = c[u][t][0]
    for idx, S in enumerate(F_subsets):
        cand_val = c[u][t][idx]
        diff = abs(cand_val - val_full)
        print(f"Agent {u} at t={t}: For candidate S={S}, candidate = {cand_val:.2f}, full = {val_full:.2f}, diff = {diff:.2f}")
    valid_candidates = []
    for idx, S in enumerate(F_subsets):
        if len(S)==0 or (u in S):
            continue
        diff = abs(c[u][t][idx] - val_full)
        if diff >= epsilon:
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
    
    # Selection: each node picks x_u^(t+1) based on its candidate vector.
    for u in all_agents:
        x_next = select_state(u, t+1)
        x_values[u].append(x_next)
    
    print(f"  Final selected states at t={t+1}:")
    for u in all_agents:
        print(f"    Agent {u}: x = {(x_values[u][-1]):.2f}")

# Convert the x_values dictionary into a states array.
# The columns will correspond to agents sorted by their IDs.
agent_ids = sorted(x_values.keys())
states = np.array([x_values[u] for u in agent_ids]).T

correct_consensus = np.average(list(x_init.values()))
plot_state(states, correct_consensus, att_v, agent_ids, title="General Resilient Evolution", xlabel="Iteration", ylabel="State")
