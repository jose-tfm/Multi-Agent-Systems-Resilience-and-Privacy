import numpy as np
import matplotlib.pyplot as plt
import itertools

T = 8             
f = 1              
epsilon = 0.05     
all_agents = [1, 2, 3]
attackers = {3}
x_init = {1: 0.0, 2: 1.0, 3: 0.1}

def attacked_value(agent, t):
    """Attacked node (3) always broadcasts 0.1, for all subsets."""
    if agent in attackers:
        return 0.1
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

# x_values[u][t] will store the final selected state for agent u at time t.
x_values = {u: [x_init[u]] for u in all_agents}
adjacency = {u: all_agents.copy() for u in all_agents}


# Parallel Consensus Updates (Step 2)
def consensus_candidate(u, t, S):
    """
    Return c_u^(t+1)[S] by averaging c_v^(t)[S] for v in adjacency[u]\S.
    If u is attacked, always return 0.1.
    """
    if u in attackers:
        return 0.1
    excluded = S
    included_agents = [v for v in adjacency[u] if v not in excluded]
    vals = [c[v][t][subset_index[S]] for v in included_agents]
    if len(vals) == 0:
        return c[u][t][subset_index[S]]
    return np.mean(vals)

# -----------------------------
# Selection Step (Step 3)
# -----------------------------
def select_state(u, t):
    """
    For a normal agent:
      - Print the candidate difference for every candidate subset S (even those that exclude u).
      - Then, among the candidates that do NOT exclude u, if exactly one candidate has an
        absolute difference from the full candidate (S = ∅) >= epsilon, return that candidate's value.
      - Otherwise, return the full candidate.
    For an attacked agent, always return 0.1.
    """
    if u in attackers:
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

# -----------------------------
# Main Simulation Loop
# -----------------------------
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


time_axis = np.arange(T+1)
plt.figure(figsize=(8,5))
colors = {1:'g', 2:'b', 3:'r'}
markers = {1:'s', 2:'s', 3:'o'}

for u in all_agents:
    if u in attackers:
        lbl = f'Attacker {u}'
    else:
        lbl = f'Agent {u} (normal)'
    plt.plot(time_axis, [(x) for x in x_values[u]], marker=markers[u], color=colors[u],
             linestyle='-', label=lbl, markersize=8)

plt.xlabel("Time")
plt.ylabel("Final Selected State x_u(t)")
plt.title("Algorithm 1 Attacker agent 3")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
