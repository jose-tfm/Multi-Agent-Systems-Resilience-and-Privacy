import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ============================================================================
# CONFIGURATION - Easy to modify for different tests
# ============================================================================

# Network topology (same as original)
neighbors = [
    [9, 1, 4, 7, 6],        
    [0, 9, 2, 3, 8, 5, 6],   
    [3, 9, 5, 1, 6, 4],      
    [2, 9, 5, 8, 4, 6, 1],   
    [0, 6, 8, 7, 9, 1],      
    [5, 8, 7, 6, 1, 9, 4, 2],
    [5, 8, 7, 4, 1, 3, 0],   
    [8, 6, 4, 0],           
    [7, 5, 3, 6, 4, 1, 9],    
    [2, 0, 3, 1, 5, 8, 6, 4]  
]

# Initial state for agents (same as original)
x0 = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.5, 0.4, 0.6, 0.3]
indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_iterations = 30
f = 2

# ============================================================================
# ATTACK SCENARIOS - Choose one by uncommenting
# ============================================================================

# Test 1: Single attacker (agent 0)
# att_v = {0: (lambda t: 0.3)}

# Test 2: Single attacker (agent 8) 
# att_v = {8: (lambda t: 0.8)}

# Test 3: Two attackers (original configuration)
att_v = {0: (lambda t: 0.3), 8: (lambda t: 0.8)}

# Test 4: Two different attackers
# att_v = {2: (lambda t: 0.1), 7: (lambda t: 0.9)}

print(f"Testing with {len(att_v)} attacker(s): {list(att_v.keys())}")

# ============================================================================
# IMPROVED M-MSR ALGORITHM
# ============================================================================

def msr_consensus_custom(x0, neighbors, max_iterations, f, attck_vals={}):
    """
    M-MSR consensus algorithm with improved tracking.
    
    Parameters:
      - x0: Initial state values (list or numpy array).
      - neighbors: List of lists with neighbor indices for each agent.
      - max_iterations: Number of iterations to run.
      - f: Number of outliers to remove (from both ends of sorted values).
      - attck_vals: Dictionary with attacked agents and their faulty values.
    
    Returns:
      - x_vals: List of state vectors over iterations.
      - convergence_info: Dictionary with convergence metrics.
    """
    num_agents = len(x0)
    x_vals = [np.array(x0, dtype=float)]
    
    print(f"Initial State: {x0}")
    print(f"Neighbor topology: {neighbors}")
    print(f"Filter parameter f = {f}")
    print(f"Attacked agents: {list(attck_vals.keys())}")
    print("-" * 60)
    
    # Set initial state for attacked agents
    for a in attck_vals:
        x_vals[-1][a] = attck_vals[a](0)
    
    # Track convergence
    honest_agents = [i for i in range(num_agents) if i not in attck_vals]
    convergence_info = {
        'converged': False,
        'convergence_round': None,
        'final_consensus': None,
        'honest_variance': []
    }
    
    for k in range(max_iterations):
        x_new = np.zeros(num_agents)
        print(f"\nIteration {k+1}:")
        
        for i in range(num_agents):
            # Get neighbor indices including self
            inds = neighbors[i][:]
            if i not in inds:
                inds.append(i)
            
            # Collect values from neighbors
            vals = np.array([x_vals[-1][j] for j in inds])
            weights = np.ones(len(inds)) / len(inds)
            
            print(f"  Agent {i}: neighbors={inds}, values={vals.round(3)}")
            
            # Sort values and weights
            sort_order = np.argsort(vals)
            vals_sorted = vals[sort_order]
            weights_sorted = weights[sort_order]
            
            # Apply MSR filtering
            if f > 0 and len(vals_sorted) > 2 * f:
                filtered_vals = vals_sorted[f:-f]
                filtered_weights = weights_sorted[f:-f]
                print(f"    Filtered (remove {f} min/max): {filtered_vals.round(3)}")
            else:
                filtered_vals = vals_sorted
                filtered_weights = weights_sorted
                print(f"    No filtering applied: {filtered_vals.round(3)}")
            
            # Update state with weighted average
            if np.sum(filtered_weights) > 0:
                x_new[i] = np.sum(filtered_vals * filtered_weights) / np.sum(filtered_weights)
            else:
                x_new[i] = np.mean(filtered_vals)
            
            print(f"    New state: {x_new[i]:.4f}")
        
        # Force attacked agents to their faulty values
        for a in attck_vals:
            x_new[a] = attck_vals[a](k+1)
            print(f"  Attacked agent {a} forced to: {x_new[a]:.4f}")
        
        print(f"State at iteration {k+1}: {x_new.round(4)}")
        x_vals.append(x_new.copy())
        
        # Check convergence of honest agents
        if len(honest_agents) > 1:
            honest_vals = [x_new[i] for i in honest_agents]
            variance = np.var(honest_vals)
            convergence_info['honest_variance'].append(variance)
            
            # More robust convergence detection
            convergence_threshold = 1e-3
            if variance < convergence_threshold and not convergence_info['converged']:
                # Double-check: ensure convergence is stable for at least 2 iterations
                if (len(convergence_info['honest_variance']) >= 2 and 
                    convergence_info['honest_variance'][-2] < convergence_threshold):
                    convergence_info['converged'] = True
                    convergence_info['convergence_round'] = k + 1
                    convergence_info['final_consensus'] = np.mean(honest_vals)
                    print(f"  >>> CONVERGENCE DETECTED at iteration {k+1} <<<")
                    print(f"      Honest agents consensus: {np.mean(honest_vals):.6f}")
                    print(f"      Variance: {variance:.2e}")
            
            # Stop early if converged for several iterations (optional)
            if (convergence_info['converged'] and 
                k + 1 - convergence_info['convergence_round'] >= 5):
                print(f"  Early stopping: Stable convergence maintained for 5 iterations")
                break
    
    return x_vals, convergence_info

# ============================================================================
# ENHANCED PLOTTING FUNCTION
# ============================================================================

def plot_network_topology(neighbors, att_v, title="Network Topology", save_path=None):
    """
    Create a publication-quality network topology visualization.
    
    Parameters:
      - neighbors: List of neighbor lists for each agent
      - att_v: Dictionary of attacked agents
      - title: Plot title
      - save_path: Optional path to save the figure (e.g., 'network.pdf', 'network.png')
    """
    # Create NetworkX graph
    G = nx.Graph()
    num_agents = len(neighbors)
    
    # Add nodes
    for i in range(num_agents):
        G.add_node(i)
    
    # Add edges
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            if neighbor < num_agents:  # Ensure valid neighbor index
                G.add_edge(i, neighbor)
    
    # Set up the plot with high DPI for publication quality
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Choose layout algorithm - spring layout usually works well
    # You can also try: circular_layout, kamada_kawai_layout, shell_layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Alternative layouts to try:
    # pos = nx.circular_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    
    # Define colors and styles
    node_colors = []
    node_sizes = []
    
    for i in range(num_agents):
        if i in att_v:
            node_colors.append('#FF4444')  # Red for attackers
            node_sizes.append(800)         # Larger size for attackers
        else:
            node_colors.append('#4444FF')  # Blue for honest agents
            node_sizes.append(600)         # Normal size for honest agents
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=1.5,
                          alpha=0.7)
    
    # Add labels with better formatting
    labels = {i: f'{i}' for i in range(num_agents)}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=14,
                           font_weight='bold',
                           font_color='white')
    
    # Create custom legend
    honest_patch = plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='#4444FF', markersize=12, 
                             label='Honest Agents', markeredgecolor='black', markeredgewidth=1)
    attack_patch = plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#FF4444', markersize=15,
                             label='Attacked Agents', markeredgecolor='black', markeredgewidth=1)
    
    plt.legend(handles=[honest_patch, attack_patch], 
              loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Formatting for publication quality
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')  # Remove axes for cleaner look
    
    # Add network statistics as text
    stats_text = f"Nodes: {G.number_of_nodes()}\n"
    stats_text += f"Edges: {G.number_of_edges()}\n"
    stats_text += f"Attackers: {len(att_v)}\n"
    stats_text += f"Density: {nx.density(G):.3f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Network topology saved to: {save_path}")
    
    plt.show()
    
    # Print network analysis
    print("\n" + "="*50)
    print("NETWORK TOPOLOGY ANALYSIS")
    print("="*50)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    print(f"Diameter: {nx.diameter(G) if nx.is_connected(G) else 'Disconnected'}")
    print(f"Average clustering: {nx.average_clustering(G):.3f}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"Degree distribution: min={min(degrees)}, max={max(degrees)}, std={np.std(degrees):.2f}")
    
    # Connectivity analysis
    if nx.is_connected(G):
        print("Network is connected ✓")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.3f}")
    else:
        print("Network is disconnected ✗")
        components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(components)}")

def plot_enhanced_state(states, correct_consensus, att_v, agent_ids, 
                       title="W-MSR Algorithm", convergence_info=None):
    """Enhanced plotting with better visualization - Main state evolution only."""
    
    iterations = np.arange(states.shape[0])
    plt.figure(figsize=(12, 8))
    
    # State evolution plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
    
    for col, agent in enumerate(agent_ids):
        agent_idx = agent - 1  # Convert to 0-indexed
        
        if agent_idx in att_v:
            plt.plot(iterations, states[:, col], 
                    color='red', marker='v', linestyle='--', linewidth=2,
                    label=f'Attacked Agent {agent}', markersize=6)
        else:
            plt.plot(iterations, states[:, col], 
                    color=colors[col], marker='o', linestyle='-', 
                    label=f'Honest Agent {agent}', markersize=4, alpha=0.8)
    
    # Add consensus line
    plt.axhline(correct_consensus, color='black', linestyle=':', linewidth=2,
               label=f"True Consensus ({correct_consensus:.3f})")
    
    # Mark convergence point if available
    if convergence_info and convergence_info['converged']:
        conv_round = convergence_info['convergence_round']
        plt.axvline(conv_round, color='green', linestyle='--', alpha=0.7,
                   label=f"Convergence (round {conv_round})")
        plt.axhline(convergence_info['final_consensus'], color='green', 
                   linestyle='--', alpha=0.7,
                   label=f"Final Consensus ({convergence_info['final_consensus']:.3f})")
    
    plt.title(f"{title} - State Evolution", fontsize=16, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("State Value", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Network size: {len(agent_ids)} agents")
    print(f"Attackers: {list(att_v.keys())} ({len(att_v)} total)")
    print(f"Filter parameter: f = {f}")
    print(f"True consensus: {correct_consensus:.4f}")
    
    if convergence_info:
        if convergence_info['converged']:
            print(f"Converged: YES (round {convergence_info['convergence_round']})")
            print(f"Final consensus: {convergence_info['final_consensus']:.6f}")
            error = abs(convergence_info['final_consensus'] - correct_consensus)
            print(f"Consensus error: {error:.6f}")
            
            # Analyze post-convergence variance behavior
            conv_round = convergence_info['convergence_round'] - 1  # 0-indexed
            if conv_round < len(convergence_info['honest_variance']) - 1:
                post_conv_variance = convergence_info['honest_variance'][conv_round:]
                print(f"Post-convergence variance range: [{min(post_conv_variance):.2e}, {max(post_conv_variance):.2e}]")
                print(f"Variance still changing due to:")
                print(f"  - Floating-point precision limits")
                print(f"  - Continued MSR filtering operations")
                print(f"  - Residual influence from attacked neighbors")
        else:
            print("Converged: NO")
            
        final_variance = convergence_info['honest_variance'][-1] if convergence_info['honest_variance'] else None
        if final_variance is not None:
            print(f"Final variance: {final_variance:.2e}")
            
        # Show honest agent final values for analysis
        final_states = states[-1]
        honest_final = [final_states[i-1] for i, agent_id in enumerate(agent_ids) 
                       if (agent_id-1) not in att_v]
        print(f"Final honest agent values: {[f'{val:.6f}' for val in honest_final]}")
        print(f"Standard deviation: {np.std(honest_final):.2e}")

# ============================================================================
# RUN SIMULATION
# ============================================================================

if __name__ == "__main__":
    print("W-MSR Algorithm Simulation")
    print("="*60)
    
    # First, visualize the network topology
    print("Generating network topology visualization...")
    plot_network_topology(neighbors, att_v, 
                         title="Multi-Agent Network Topology for W-MSR Algorithm",
                         save_path="network_topology.pdf")  # Saves high-quality PDF for thesis
    
    # Run the consensus algorithm
    r, conv_info = msr_consensus_custom(x0, neighbors, max_iterations, f, attck_vals=att_v)
    
    # Convert to numpy array for plotting
    states = np.array(r)
    
    # Calculate correct consensus (average of initial honest agent values)
    honest_initial = [x0[i] for i in range(len(x0)) if i not in att_v]
    correct_consensus = np.mean(honest_initial)
    
    # Plot results with enhanced visualization
    plot_enhanced_state(states, correct_consensus, att_v, indices, 
                       title="W-MSR Algorithm", convergence_info=conv_info)
