import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def plot_enhanced_msr(states, correct_consensus, att_v, agent_ids, title="W-MSR Algorithm Evolution", 
                     xlabel="Iteration", ylabel="State Value", save_path=None):
    """
    Enhanced plotting function for M-MSR algorithm with multiple visualization improvements.
    
    Parameters:
      - states: A numpy array of shape (num_iterations, num_agents) containing the state evolution.
      - correct_consensus: The consensus value to mark
      - att_v: Dictionary of attacked agents. Keys are agent IDs, values are functions returning the attack value.
      - agent_ids: List of agent IDs corresponding to the columns in states.
      - title: Plot title
      - xlabel, ylabel: Axis labels  
      - save_path: Optional path to save the plot
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    iterations = np.arange(states.shape[0])
    
    # Define color schemes
    honest_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len([a for a in agent_ids if a not in att_v])))
    attack_colors = ['red', 'darkred', 'crimson', 'maroon']
    
    # ======================= SUBPLOT 1: Main Evolution Plot =======================
    ax1 = axes[0, 0]
    honest_idx = 0
    attack_idx = 0
    
    for col, agent in enumerate(agent_ids):
        if agent in att_v:
            ax1.plot(iterations, states[:, col], 
                    marker='v', markersize=4, linewidth=2.5, linestyle='--',
                    color=attack_colors[attack_idx % len(attack_colors)],
                    label=f'Attacked Agent {agent}', alpha=0.9)
            attack_idx += 1
        else:
            ax1.plot(iterations, states[:, col], 
                    marker='o', markersize=3, linewidth=1.5,
                    color=honest_colors[honest_idx % len(honest_colors)],
                    label=f'Honest Agent {agent}', alpha=0.8)
            honest_idx += 1
    
    # Add consensus line and convergence region
    ax1.axhline(correct_consensus, color='green', linestyle='-', linewidth=2, 
               label=f"Target Consensus ({correct_consensus:.3f})", alpha=0.8)
    
    # Add convergence tolerance band
    tolerance = 0.01
    ax1.fill_between(iterations, correct_consensus - tolerance, correct_consensus + tolerance,
                    color='green', alpha=0.2, label=f'Convergence Band (±{tolerance})')
    
    ax1.set_title(f"{title} - State Evolution", fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # ======================= SUBPLOT 2: Convergence Analysis =======================
    ax2 = axes[0, 1]
    
    # Calculate convergence metrics
    honest_states = states[:, [i for i, agent in enumerate(agent_ids) if agent not in att_v]]
    convergence_error = np.std(honest_states, axis=1)
    consensus_error = np.abs(np.mean(honest_states, axis=1) - correct_consensus)
    
    ax2.plot(iterations, convergence_error, 'b-', linewidth=2, label='Inter-agent Variance', marker='o', markersize=3)
    ax2.plot(iterations, consensus_error, 'g-', linewidth=2, label='Consensus Error', marker='s', markersize=3)
    
    ax2.set_title("Convergence Analysis", fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Error Magnitude", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ======================= SUBPLOT 3: Attack Impact Analysis =======================
    ax3 = axes[1, 0]
    
    if att_v:
        # Show the effect of attacks on honest agents
        honest_mean = np.mean(honest_states, axis=1)
        ax3.plot(iterations, honest_mean, 'b-', linewidth=2, label='Honest Agents Mean', marker='o')
        
        for col, agent in enumerate(agent_ids):
            if agent in att_v:
                ax3.plot(iterations, states[:, col], 
                        linestyle='--', linewidth=2, marker='v',
                        color=attack_colors[0], alpha=0.7,
                        label=f'Attacker {agent}')
        
        # Show resilience - distance from attacks
        attack_mean = np.mean([states[:, i] for i, agent in enumerate(agent_ids) if agent in att_v], axis=0)
        resilience = np.abs(honest_mean - attack_mean)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(iterations, resilience, 'r:', linewidth=2, label='Resilience Distance', alpha=0.8)
        ax3_twin.set_ylabel('Resilience Distance', color='red', fontsize=12)
        ax3_twin.tick_params(axis='y', labelcolor='red')
    
    ax3.axhline(correct_consensus, color='green', linestyle='-', alpha=0.6)
    ax3.set_title("Attack Impact & Resilience", fontsize=14, fontweight='bold')
    ax3.set_xlabel(xlabel, fontsize=12)
    ax3.set_ylabel("State Value", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # ======================= SUBPLOT 4: Final Statistics =======================
    ax4 = axes[1, 1]
    
    # Create summary statistics
    final_states = states[-1, :]
    honest_final = [final_states[i] for i, agent in enumerate(agent_ids) if agent not in att_v]
    attack_final = [final_states[i] for i, agent in enumerate(agent_ids) if agent in att_v]
    
    categories = ['Honest\nAgents', 'Target\nConsensus']
    values = [np.mean(honest_final), correct_consensus]
    colors = ['lightblue', 'lightgreen']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add error bars for honest agents
    if len(honest_final) > 1:
        ax4.errorbar(0, np.mean(honest_final), yerr=np.std(honest_final), 
                    fmt='none', color='black', capsize=5, capthick=2)
    
    # Add attack values as red dots
    if attack_final:
        ax4.scatter([0] * len(attack_final), attack_final, 
                   color='red', s=100, marker='v', label='Attack Values', zorder=5)
    
    ax4.set_title("Final State Summary", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Final State Value", fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax4.text(i, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    if attack_final:
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence_rate(states, att_v, agent_ids, tolerance=0.01, save_path=None):
    """
    Plot convergence rate analysis for M-MSR algorithm.
    """
    plt.figure(figsize=(12, 8))
    
    honest_agents = [i for i, agent in enumerate(agent_ids) if agent not in att_v]
    honest_states = states[:, honest_agents]
    
    # Calculate when each agent converges
    target = np.mean(states[0, honest_agents])  # Initial honest average
    convergence_times = []
    
    for i in range(honest_states.shape[1]):
        agent_states = honest_states[:, i]
        converged = np.where(np.abs(agent_states - target) < tolerance)[0]
        if len(converged) > 0:
            convergence_times.append(converged[0])
        else:
            convergence_times.append(len(agent_states))
    
    # Plot convergence times
    plt.subplot(2, 1, 1)
    honest_agent_ids = [agent_ids[i] for i in honest_agents]
    bars = plt.bar(range(len(honest_agent_ids)), convergence_times, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xticks(range(len(honest_agent_ids)), [f'Agent {id}' for id in honest_agent_ids])
    plt.ylabel('Convergence Time (iterations)')
    plt.title('Individual Agent Convergence Times', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Plot convergence evolution
    plt.subplot(2, 1, 2)
    # Calculate percentage of agents converged over time
    converged_percentage = []
    for t in range(states.shape[0]):
        current_states = honest_states[t, :]
        converged_count = np.sum(np.abs(current_states - target) < tolerance)
        converged_percentage.append(100 * converged_count / len(current_states))
    
    plt.plot(range(states.shape[0]), converged_percentage, 'g-', linewidth=3, marker='o')
    plt.axhline(100, color='red', linestyle='--', alpha=0.7, label='Full Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Converged Agents (%)')
    plt.title('Convergence Progress Over Time', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_network_topology(neighbors, att_v=None, save_path=None):
    """
    Visualize the network topology used in M-MSR algorithm.
    """
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    num_agents = len(neighbors)
    
    # Add nodes
    for i in range(num_agents):
        G.add_node(i)
    
    # Add edges
    for i, neighbor_list in enumerate(neighbors):
        for j in neighbor_list:
            if i != j:  # Avoid self-loops for visualization
                G.add_edge(i, j)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Color nodes based on attack status
    node_colors = []
    for i in range(num_agents):
        if att_v and i in att_v:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    # Draw network
    nx.draw(G, pos, node_color=node_colors, node_size=800, 
            with_labels=True, font_size=12, font_weight='bold',
            edge_color='gray', width=2, alpha=0.7)
    
    # Add legend
    if att_v:
        red_patch = mpatches.Patch(color='red', label='Attacked Agents')
        blue_patch = mpatches.Patch(color='lightblue', label='Honest Agents')
        plt.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    plt.title(f"Network Topology ({num_agents} agents, {G.number_of_edges()} edges)", 
              fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comparison_plot(results_dict, save_path=None):
    """
    Compare multiple M-MSR runs with different parameters.
    
    results_dict: Dictionary where keys are labels and values are (states, att_v, agent_ids) tuples
    """
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (label, (states, att_v, agent_ids)) in enumerate(results_dict.items()):
        iterations = np.arange(states.shape[0])
        honest_agents = [j for j, agent in enumerate(agent_ids) if agent not in att_v]
        honest_states = states[:, honest_agents]
        honest_mean = np.mean(honest_states, axis=1)
        
        plt.plot(iterations, honest_mean, color=colors[i], linewidth=2, 
                label=f'{label} (Honest Mean)', marker='o', markersize=3)
        
        # Add confidence band
        honest_std = np.std(honest_states, axis=1)
        plt.fill_between(iterations, honest_mean - honest_std, honest_mean + honest_std,
                        color=colors[i], alpha=0.2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('State Value', fontsize=12)
    plt.title('M-MSR Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
