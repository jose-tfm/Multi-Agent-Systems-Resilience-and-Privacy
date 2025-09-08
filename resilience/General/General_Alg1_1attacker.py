import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys

# Configure matplotlib for PLOS ONE publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0
})

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom utils if available, otherwise define minimal plotting function
try:
    from utils.utils import plot_state
except ImportError:
    def plot_state(states, correct_consensus, att_v, agent_ids, title, xlabel, ylabel):
        """PLOS ONE optimized plotting function"""
        pass  # Will be defined below

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

def plot_state_plos_one(states, correct_consensus, att_v, agent_ids, title, xlabel, ylabel, save_figure=True):
    """
    PLOS ONE publication-quality plotting function for resilient consensus algorithms.
    
    Parameters:
      - states: A numpy array of shape (num_iterations, num_agents) containing the state evolution
      - correct_consensus: The consensus value to mark
      - att_v: List of attacked agents  
      - agent_ids: List of agent IDs corresponding to the columns in states
      - save_figure: Whether to save the figure as PNG and PDF
    """
    iterations = np.arange(states.shape[0])
    
    # Create figure with PLOS ONE recommended size
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # PLOS ONE color palette (colorblind-friendly)
    honest_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    attack_color = '#e74c3c'  # Red for attackers
    consensus_color = '#2c3e50'  # Dark blue-gray for consensus line
    
    # Plot agent trajectories with enhanced styling
    honest_agents = [agent for agent in agent_ids if agent not in att_v]
    
    for col, agent in enumerate(agent_ids):
        if agent in att_v:
            ax.plot(iterations, states[:, col],
                   color=attack_color, marker='s', linestyle='--', 
                   linewidth=3, markersize=7, markevery=1,
                   label=f'Attacker {agent}', alpha=0.9, zorder=3)
        else:
            color_idx = honest_agents.index(agent) % len(honest_colors)
            ax.plot(iterations, states[:, col],
                   color=honest_colors[color_idx], 
                   marker='o', linewidth=2.5, markersize=6, markevery=1,
                   label=f'Honest Agent {agent}', alpha=0.9, zorder=2)

    # Plot true consensus line
    ax.axhline(correct_consensus, color=consensus_color, linestyle=':', 
               linewidth=2.5, label=f"True Consensus = {correct_consensus:.2f}", 
               alpha=0.8, zorder=1)
    
    # Enhanced formatting for publication
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    # Position legend to avoid data overlap
    ax.legend(loc='best', framealpha=1.0, edgecolor='black', 
              facecolor='white', fontsize=10)
    
    # Clean grid and spines for publication
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set reasonable axis limits with padding
    y_range = np.max(states) - np.min(states)
    y_padding = max(0.1, y_range * 0.1)
    y_min = np.min(states) - y_padding
    y_max = np.max(states) + y_padding
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.2, len(iterations) - 0.8)
    
    # Ensure integer ticks on x-axis (iterations)
    ax.set_xticks(np.arange(0, len(iterations)))
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save figures in both PNG and PDF for publication
    if save_figure:
        filename_base = "General_Alg1_1attacker_resilient_consensus"
        
        # High-resolution PNG for manuscript
        plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved: {filename_base}.png")
        
        # Vector PDF for publication
        plt.savefig(f'{filename_base}.pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved: {filename_base}.pdf")
    
    plt.show()
    
    # Print summary statistics for the paper
    print(f"\\n=== SIMULATION SUMMARY FOR PUBLICATION ===")
    print(f"Network size: {len(agent_ids)} agents")
    print(f"Attackers: {len(att_v)} agent(s) - {att_v}")
    print(f"Honest agents: {len(honest_agents)} agent(s) - {honest_agents}")
    print(f"True consensus value: {correct_consensus:.3f}")
    print(f"Simulation duration: {len(iterations)} iterations")
    print(f"Final honest agent states:")
    for agent in honest_agents:
        col = agent_ids.index(agent)
        final_state = states[-1, col]
        error = abs(final_state - correct_consensus)
        print(f"  Agent {agent}: {final_state:.3f} (error: {error:.3f})")
    
    print(f"\\nConvergence analysis:")
    honest_final_states = [states[-1, agent_ids.index(agent)] for agent in honest_agents]
    if len(honest_final_states) > 1:
        consensus_error = np.std(honest_final_states)
        print(f"  Honest agents consensus error (std): {consensus_error:.6f}")
        print(f"  Maximum error from true consensus: {max([abs(s - correct_consensus) for s in honest_final_states]):.6f}")
    
    return fig, ax

# Generate the publication-quality plot
plot_state_plos_one(states, correct_consensus, att_v, all_agents, 
                    title="Resilient Consensus with General Algorithm 1", 
                    xlabel="Iteration", 
                    ylabel="Agent State Value")
