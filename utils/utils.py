import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for LaTeX-quality output
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (16, 10),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 1.5,
    'lines.linewidth': 4,
    'lines.markersize': 10,
    'legend.fontsize': 20,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.labelsize': 26,
    'axes.titlesize': 30
})

def plot_state(states, correct_consensus, att_v, agent_ids, title, xlabel, ylabel):
    """
    Professional plotting function optimized for LaTeX documents and academic publications.
    
    Parameters:
      - states: A numpy array of shape (num_iterations, num_agents) containing the state evolution.
      - correct_consensus: The consensus value to mark.
      - att_v: Dictionary of attacked agents. Keys are agent IDs, values are functions returning the attack value.
      - agent_ids: List of agent IDs corresponding to the columns in states.
    """
    iterations = np.arange(states.shape[0])
    
    # Create figure with optimal proportions for academic papers
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    
    # Professional color palette
    honest_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    attack_color = '#ff0000'  # Red for attackers
    consensus_color = '#000000'  # Black for consensus line

    # Plot agent trajectories
    for col, agent in enumerate(agent_ids):
        if agent in att_v:
            ax.plot(iterations, states[:, col],
                   color=attack_color, marker='v', linestyle='--', 
                   linewidth=5, markersize=12, markevery=2,
                   label=f'Attacker {agent}', alpha=0.9)
        else:
            ax.plot(iterations, states[:, col],
                   color=honest_colors[col % len(honest_colors)], 
                   marker='o', linewidth=4, markersize=10, markevery=3,
                   label=f'Honest Agent {agent}', alpha=0.9)

    # Plot consensus line
    ax.axhline(correct_consensus, color=consensus_color, linestyle=':', 
               linewidth=6, label=f"True Consensus ({correct_consensus:.2f})", alpha=0.8)
    
    # Enhanced formatting
    ax.set_title(title, fontweight='bold', pad=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Legend positioned to avoid data overlap
    legend = ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02),
                      framealpha=0.98, edgecolor='black', 
                      facecolor='white', fontsize=18)
    legend.get_frame().set_linewidth(2)
    
    # Clean grid and spines
    ax.grid(True, alpha=0.3, linewidth=1.5, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Set reasonable axis limits with padding
    y_min = np.min(states) - 1.0
    y_max = np.max(states) + 1.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, len(iterations) - 0.5)
    
    # Ensure integer ticks on x-axis
    ax.set_xticks(np.arange(0, len(iterations), max(1, len(iterations)//10)))
    
    # Standard tight layout
    plt.tight_layout(pad=1.5)
    plt.show()
