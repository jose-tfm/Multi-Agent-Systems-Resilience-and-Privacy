import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigvals, eig
from scipy.optimize import minimize
from matplotlib import rcParams

# Set publication-quality plotting parameters with larger fonts
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 20
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 22
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

def row_normalize(M: np.ndarray) -> np.ndarray:
    """Row-stochastic normalization"""
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-12
    return M / s

def build_Ap(N: int, A: np.ndarray, optimized_augmented: bool = False) -> np.ndarray:
    """Build augmented matrix with original or optimized augmented weights"""
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    
    for i in range(N):
        inds = {
            0: i,        
            1: N + 3*i,   
            2: N + 3*i+1, 
            3: N + 3*i+2  
        }
        
        if not optimized_augmented:
            # Original augmented weights from the paper
            Ap[inds[0], inds[1]] = 1.0   # x_i -> y_i1
            Ap[inds[1], inds[0]] = 1.0   # y_i1 -> x_i
            Ap[inds[0], inds[3]] = 2.0   # x_i -> y_i3
            Ap[inds[3], inds[2]] = 1.0   # y_i3 -> y_i2
            Ap[inds[2], inds[0]] = 1.0   # y_i2 -> x_i
        else:
            # Optimized augmented weights (example values - can be further tuned)
            Ap[inds[0], inds[1]] = 0.186  # Optimized weight
            Ap[inds[1], inds[0]] = 1.0
            Ap[inds[0], inds[3]] = 0.108  # Optimized weight  
            Ap[inds[3], inds[2]] = 1.0
            Ap[inds[2], inds[0]] = 1.0
    
    return row_normalize(Ap)

def build_A_from_p(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct A matrix from parameter vector p"""
    N = mask.shape[0]
    A = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                A[i, j] = p[idx]
                idx += 1
    return row_normalize(A)

def consensus_rate_p(p: np.ndarray, mask: np.ndarray, N: int, optimized_augmented: bool = False) -> float:
    """Objective function: second largest eigenvalue magnitude"""
    A_var = build_A_from_p(p, mask)
    vals = np.abs(eigvals(build_Ap(N, A_var, optimized_augmented=optimized_augmented)))
    vals.sort()
    return vals[-2]

def create_three_stage_eigenvalue_comparison(N=20, p_edge=0.6, seed=42, save_figure=True):
    """
    Creates a three-stage comparison showing the progressive impact of optimizations:
    1. Original network + Original augmented weights
    2. Optimized network + Original augmented weights  
    3. Optimized network + Optimized augmented weights
    """
    np.random.seed(seed)
    
    # Create the test network
    while True:
        G = nx.erdos_renyi_graph(N, p_edge, seed=seed)
        if nx.is_connected(G):
            break
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.rand()
    
    # Extract and normalize adjacency matrix
    A_original = nx.to_numpy_array(G, nodelist=range(N), weight='weight', dtype=float)
    A_original = row_normalize(A_original)
    
    # Optimize the network weights (keeping augmented weights original)
    mask = A_original > 0
    p0 = np.array([A_original[i, j] for i in range(N) for j in range(N) if mask[i, j]])
    bounds = [(0.01, None)] * p0.size
    
    print("Stage 1: Optimizing network weights (keeping original augmented weights)...")
    res1 = minimize(lambda p: consensus_rate_p(p, mask, N, optimized_augmented=False),
                    p0, method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-9, 'maxiter': 500})
    
    A_net_optimized = build_A_from_p(res1.x, mask)
    
    print("Stage 2: Optimizing network weights with optimized augmented weights...")
    res2 = minimize(lambda p: consensus_rate_p(p, mask, N, optimized_augmented=True),
                    p0, method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-9, 'maxiter': 500})
    
    A_full_optimized = build_A_from_p(res2.x, mask)
    
    # Build all three augmented matrices
    P1_original = build_Ap(N, A_original, optimized_augmented=False)  # Original + Original
    P2_net_opt = build_Ap(N, A_net_optimized, optimized_augmented=False)  # Net Opt + Original Aug
    P3_full_opt = build_Ap(N, A_full_optimized, optimized_augmented=True)  # Net Opt + Aug Opt
    
    # Compute eigenvalues for all three cases
    eigs1 = eigvals(P1_original)
    eigs2 = eigvals(P2_net_opt)
    eigs3 = eigvals(P3_full_opt)
    
    # Sort eigenvalues by magnitude
    eigs1_sorted = sorted(eigs1, key=lambda x: abs(x), reverse=True)
    eigs2_sorted = sorted(eigs2, key=lambda x: abs(x), reverse=True)
    eigs3_sorted = sorted(eigs3, key=lambda x: abs(x), reverse=True)
    
    # Extract second largest eigenvalue magnitudes
    lambda2_1 = abs(eigs1_sorted[1])
    lambda2_2 = abs(eigs2_sorted[1])
    lambda2_3 = abs(eigs3_sorted[1])
    
    # Create three-panel comparison plot with larger size for better paper visibility
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot unit circle for all plots
    theta = np.linspace(0, 2*np.pi, 1000)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    
    # Colors for different stages
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    
    axes = [ax1, ax2, ax3]
    eigenvals = [eigs1, eigs2, eigs3]
    lambda2s = [lambda2_1, lambda2_2, lambda2_3]
    titles = [
        'Original Network +\nOriginal Augmented Weights',
        'Optimized Network +\nOriginal Augmented Weights', 
        'Optimized Network +\nOptimized Augmented Weights'
    ]
    
    for i, (ax, eigs, lambda2, title, color) in enumerate(zip(axes, eigenvals, lambda2s, titles, colors)):
        # Plot unit circle
        ax.plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.4, linewidth=1.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Real Part')
        if i == 0:
            ax.set_ylabel('Imaginary Part')
        
        # Sort eigenvalues for this plot
        eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
        
        # Plot all eigenvalues
        ax.scatter(np.real(eigs), np.imag(eigs), c=color, s=60, alpha=0.7, 
                  edgecolors='black', linewidth=1.2, label='All Eigenvalues', zorder=3)
        
        # Highlight dominant eigenvalue
        ax.scatter(np.real(eigs_sorted[0]), np.imag(eigs_sorted[0]), 
                  c='darkgreen', s=180, marker='*', edgecolors='black', linewidth=1.5,
                  label=f'$\\lambda_1 = {abs(eigs_sorted[0]):.3f}$', zorder=4)
        
        # Highlight second eigenvalue
        ax.scatter(np.real(eigs_sorted[1]), np.imag(eigs_sorted[1]), 
                  c='orange', s=150, marker='s', edgecolors='black', linewidth=1.5,
                  label=f'$\\lambda_2 = {lambda2:.3f}$', zorder=4)
        
        ax.set_title(f'({chr(97+i)}) {title}', fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
    
    # Calculate improvements
    improvement_stage1 = (lambda2_1 - lambda2_2) / lambda2_1 * 100
    improvement_stage2 = (lambda2_2 - lambda2_3) / lambda2_2 * 100
    improvement_total = (lambda2_1 - lambda2_3) / lambda2_1 * 100
    
    # Main title with improvement summary
    fig.suptitle(f'Progressive Optimization Impact on Eigenvalue Distribution (N={N})\n' + 
                f'Network Opt: {improvement_stage1:.1f}% improvement, ' +
                f'Augmented Opt: {improvement_stage2:.1f}% additional, ' +
                f'Total: {improvement_total:.1f}% improvement', 
                fontweight='bold', y=1.05, fontsize=18)
    
    plt.tight_layout()
    
    if save_figure:
        base_filename = f'three_stage_optimization_N{N}'
        
        plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Three-stage comparison saved: {base_filename}.png")
        
        plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"PDF version saved: {base_filename}.pdf")
    
    plt.show()
    
    # Print detailed results
    print(f"\\n=== THREE-STAGE OPTIMIZATION RESULTS ===")
    print(f"Stage 1 - Original (Network + Augmented): λ₂ = {lambda2_1:.6f}")
    print(f"Stage 2 - Network Optimized: λ₂ = {lambda2_2:.6f} ({improvement_stage1:.2f}% improvement)")
    print(f"Stage 3 - Full Optimization: λ₂ = {lambda2_3:.6f} ({improvement_stage2:.2f}% additional improvement)")
    print(f"\\nTotal improvement: {improvement_total:.2f}%")
    print(f"Convergence speedup: {lambda2_1/lambda2_3:.2f}× faster")
    
    print(f"\\nContribution breakdown:")
    print(f"  Network optimization: {improvement_stage1:.1f}% of total improvement")
    print(f"  Augmented optimization: {improvement_stage2:.1f}% additional improvement")
    print(f"  Combined effect: {improvement_total:.1f}% total improvement")
    
    return {
        'lambda2_original': lambda2_1,
        'lambda2_network_opt': lambda2_2, 
        'lambda2_full_opt': lambda2_3,
        'improvement_stage1': improvement_stage1,
        'improvement_stage2': improvement_stage2,
        'improvement_total': improvement_total,
        'A_original': A_original,
        'A_network_opt': A_net_optimized,
        'A_full_opt': A_full_optimized
    }

def create_convergence_comparison(results, N=20, save_figure=True):
    """
    Creates a bar chart showing the progressive improvements
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Lambda2 values
    stages = ['Original', 'Network\\nOptimized', 'Fully\\nOptimized']
    lambda2_values = [results['lambda2_original'], 
                      results['lambda2_network_opt'], 
                      results['lambda2_full_opt']]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    bars1 = ax1.bar(stages, lambda2_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Second Largest Eigenvalue ($\\lambda_2$)')
    ax1.set_title('(a) Eigenvalue Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, lambda2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Convergence speed (1/lambda2)
    speeds = [1/val for val in lambda2_values]
    bars2 = ax2.bar(stages, speeds, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Convergence Speed (1/$\\lambda_2$)')
    ax2.set_title('(b) Convergence Speed Improvement', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, speeds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'Progressive Optimization Benefits (N={N})', fontweight='bold')
    plt.tight_layout()
    
    if save_figure:
        filename = f'convergence_progression_N{N}'
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Convergence progression saved: {filename}.png and {filename}.pdf")
    
    plt.show()

def plot_network_topology(A, title="Network Topology", N=11, save_figure=True, filename=None):
    """
    Creates a publication-quality network topology visualization.
    Optimized for smaller networks (N=11) for clearer demonstration.
    """
    # Create directed graph from adjacency matrix
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # Remove zero-weight edges for cleaner visualization
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 1e-6]
    G.remove_edges_from(edges_to_remove)
    
    # Create figure with optimal size for N=11
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Use circular layout for N=11 - creates a cleaner, more symmetric visualization
    if N <= 12:
        pos = nx.circular_layout(G)
        # Adjust positions slightly for better spacing
        for node in pos:
            pos[node] = pos[node] * 1.2
    else:
        # For larger networks, use spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Calculate node degrees for sizing
    degrees = dict(G.degree())
    # Larger nodes for better visibility with N=11
    node_sizes = [400 + 80 * degrees[node] for node in G.nodes()]
    
    # Extract edge weights for width scaling
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        # More pronounced edge width differences for clarity
        edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
    else:
        edge_widths = [1.5] * len(G.edges())
    
    # Draw edges with prominent arrows for directed graph
    nx.draw_networkx_edges(G, pos, 
                          width=edge_widths,
                          alpha=0.8, 
                          edge_color='#1f4e79',  # Dark blue for better arrow visibility
                          arrowsize=25,  # Larger arrows
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.15',  # More curvature to see direction
                          min_source_margin=15,  # Space from source node
                          min_target_margin=15,  # Space to target node
                          ax=ax)
    
    # Draw nodes with better colors for publication
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color='lightcoral',  # Better color for publication
                          edgecolors='darkred',
                          linewidths=2,
                          alpha=0.9,
                          ax=ax)
    
    # Add node labels with better formatting
    labels = {i: str(i+1) for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=12,  # Larger font for N=11
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add a subtle grid for reference
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    
    if save_figure:
        if filename is None:
            filename = f'network_topology_N{N}'
        
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Network topology saved: {filename}.png and {filename}.pdf")
    
    plt.show()
    
    return G

def create_network_analysis_figure(results, N=11, p_edge=0.6, seed=42, save_figure=True):
    """
    Creates a comprehensive figure showing the network topology and optimization results
    Optimized for N=11 for clearer visualization
    """
    # Recreate the original network for visualization
    np.random.seed(seed)
    while True:
        G = nx.erdos_renyi_graph(N, p_edge, seed=seed)
        if nx.is_connected(G):
            break
    
    # Add random weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.rand()
    
    # Extract adjacency matrix
    A_original = nx.to_numpy_array(G, nodelist=range(N), weight='weight', dtype=float)
    A_original = row_normalize(A_original)
    
    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Original Network Topology with circular layout for N=11
    ax1 = plt.subplot(2, 2, 1)
    G_viz = nx.from_numpy_array(A_original, create_using=nx.DiGraph)
    
    # Use circular layout for better visualization with N=11
    pos = nx.circular_layout(G_viz)
    # Scale up the positions for better spacing
    for node in pos:
        pos[node] = pos[node] * 1.3
    
    # Calculate edge weights for better visualization
    edge_weights = [G_viz[u][v]['weight'] for u, v in G_viz.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 2 * (w / max_weight) for w in edge_weights]
    
    # Draw network with enhanced visibility and prominent arrows
    nx.draw_networkx_edges(G_viz, pos, alpha=0.8, edge_color='#1f4e79', 
                          arrowsize=20, arrowstyle='-|>', width=edge_widths, 
                          connectionstyle='arc3,rad=0.15',  # More curve to see direction
                          min_source_margin=12,
                          min_target_margin=12,
                          ax=ax1)
    
    # Calculate node sizes based on degree
    degrees = dict(G_viz.degree())
    node_sizes = [300 + 60 * degrees[node] for node in G_viz.nodes()]
    
    nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, node_color='lightcoral', 
                          edgecolors='darkred', linewidths=2, ax=ax1)
    
    labels = {i: str(i+1) for i in range(N)}
    nx.draw_networkx_labels(G_viz, pos, labels, font_size=10, font_weight='bold', 
                           font_color='white', ax=ax1)
    
    ax1.set_title('(a) Original Network Topology', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Plot 2: Eigenvalue comparison
    ax2 = plt.subplot(2, 2, 2)
    stages = ['Original', 'Net. Opt.', 'Full Opt.']
    lambda2_values = [results['lambda2_original'], 
                      results['lambda2_network_opt'], 
                      results['lambda2_full_opt']]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(stages, lambda2_values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Second Largest Eigenvalue ($\\lambda_2$)')
    ax2.set_title('(b) Optimization Progress', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, lambda2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Network Statistics - Better for N=11
    ax3 = plt.subplot(2, 2, 3)
    degrees_list = [G_viz.degree(i) for i in range(N)]
    unique_degrees, counts = np.unique(degrees_list, return_counts=True)
    
    bars3 = ax3.bar(unique_degrees, counts, alpha=0.7, color='skyblue', edgecolor='navy', width=0.6)
    ax3.set_xlabel('Node Degree')
    ax3.set_ylabel('Number of Nodes')
    ax3.set_title('(c) Degree Distribution', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(unique_degrees)
    
    # Add count labels on bars
    for bar, count in zip(bars3, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Improvement breakdown
    ax4 = plt.subplot(2, 2, 4)
    improvements = [results['improvement_stage1'], results['improvement_stage2']]
    improvement_labels = ['Network\\nOptimization', 'Augmented\\nOptimization']
    improvement_colors = ['#ff7f0e', '#2ca02c']
    
    bars4 = ax4.bar(improvement_labels, improvements, color=improvement_colors, 
                   alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('(d) Optimization Contributions', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars4, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add overall statistics optimized for N=11
    stats_text = (f"Network Parameters:\\n"
                 f"N = {N} nodes\\n"
                 f"p = {p_edge} edge probability\\n"
                 f"Edges: {G_viz.number_of_edges()}\\n"
                 f"Density: {nx.density(G_viz):.3f}\\n"
                 f"Diameter: {nx.diameter(G_viz)}\\n\\n"
                 f"Optimization Results:\\n"
                 f"Total Improvement: {results['improvement_total']:.1f}%\\n"
                 f"Speedup Factor: {results['lambda2_original']/results['lambda2_full_opt']:.2f}×")
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
             fontweight='bold')
    
    plt.suptitle(f'Network Topology and Optimization Analysis (N={N})', 
                fontweight='bold', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.18)
    
    if save_figure:
        filename = f'network_analysis_N{N}'
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Network analysis figure saved: {filename}.png and {filename}.pdf")
    
    plt.show()
    
    return A_original

def create_detailed_digraph_visualization(A, title="Directed Network Topology", N=11, save_figure=True, filename=None):
    """
    Creates a highly detailed directed graph visualization with prominent arrows
    and edge labels to clearly show the digraph structure.
    """
    # Create directed graph from adjacency matrix
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # Remove zero-weight edges for cleaner visualization
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 1e-6]
    G.remove_edges_from(edges_to_remove)
    
    # Create figure with optimal size
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Use circular layout for best arrow visibility
    pos = nx.circular_layout(G)
    # Scale up positions for better spacing
    for node in pos:
        pos[node] = pos[node] * 1.5
    
    # Calculate node degrees
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Node sizes based on total degree
    node_sizes = [500 + 100 * degrees[node] for node in G.nodes()]
    
    # Extract edge weights and create width mapping
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        # Normalize edge widths between 2 and 8
        edge_widths = [2 + 6 * ((w - min_weight) / (max_weight - min_weight)) for w in edge_weights]
    else:
        edge_widths = [3.0] * len(G.edges())
    
    # Draw edges with very prominent arrows
    for i, (u, v) in enumerate(G.edges()):
        # Calculate connection style based on whether there's a reverse edge
        if G.has_edge(v, u):
            # Bidirectional - use different curvatures
            connectionstyle = "arc3,rad=0.2" if u < v else "arc3,rad=-0.2"
        else:
            # Unidirectional - slight curve
            connectionstyle = "arc3,rad=0.1"
        
        nx.draw_networkx_edges(G, pos,
                              edgelist=[(u, v)],
                              width=edge_widths[i],
                              alpha=0.8,
                              edge_color='darkblue',
                              arrowsize=30,  # Very large arrows
                              arrowstyle='-|>',
                              connectionstyle=connectionstyle,
                              min_source_margin=20,
                              min_target_margin=20,
                              ax=ax)
    
    # Draw nodes with color coding based on in/out degree
    # Color nodes based on their role (more outgoing = redder, more incoming = bluer)
    node_colors = []
    for node in G.nodes():
        out_deg = out_degrees[node]
        in_deg = in_degrees[node]
        total_deg = degrees[node]
        if total_deg > 0:
            # Ratio of outgoing edges (red) vs incoming edges (blue)
            red_ratio = out_deg / total_deg
            blue_ratio = in_deg / total_deg
            # Create color mix
            color = (red_ratio, 0.3, blue_ratio)
        else:
            color = (0.5, 0.5, 0.5)  # Gray for isolated nodes
        node_colors.append(color)
    
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          edgecolors='black',
                          linewidths=3,
                          alpha=0.9,
                          ax=ax)
    
    # Add node labels
    labels = {i: str(i+1) for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=14,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Set equal aspect ratio and good limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_figure:
        if filename is None:
            filename = f'detailed_digraph_N{N}'
        
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Detailed digraph saved: {filename}.png and {filename}.pdf")
    
    plt.show()
    
    return G

if __name__ == "__main__":
    print("Creating three-stage optimization comparison...")
    
    # Use N=11 for clearer visualization and demonstration
    N = 11
    p_edge = 0.6
    
    # Create the progressive optimization analysis
    results = create_three_stage_eigenvalue_comparison(N=N, p_edge=p_edge, seed=42)
    
    # Create supplementary convergence comparison
    print("\\nCreating convergence progression charts...")
    create_convergence_comparison(results, N=N)
    
    # Create comprehensive network analysis figure
    print("\\nCreating network topology and analysis figure...")
    A_original = create_network_analysis_figure(results, N=N, p_edge=p_edge, seed=42)
    
    # Create standalone network topology plot
    print("\\nCreating standalone network topology plot...")
    plot_network_topology(A_original, 
                         title=f"{N}×{N} Erdős-Rényi Network (p={p_edge})", 
                         N=N, 
                         filename=f"network_topology_{N}x{N}")
    
    # Create detailed directed graph visualization with prominent arrows
    print("\\nCreating detailed directed graph visualization...")
    create_detailed_digraph_visualization(A_original,
                                         title=f"Detailed Directed Graph Topology (N={N})",
                                         N=N,
                                         filename=f"detailed_digraph_{N}x{N}")
    
    print("\\nThree-stage optimization analysis completed!")
    print("Generated files:")
    print(f"  - three_stage_optimization_N{N}.png/.pdf (Main eigenvalue comparison)")
    print(f"  - convergence_progression_N{N}.png/.pdf (Progress charts)")
    print(f"  - network_analysis_N{N}.png/.pdf (Comprehensive analysis)")
    print(f"  - network_topology_{N}x{N}.png/.pdf (Network topology)")
    print(f"  - detailed_digraph_{N}x{N}.png/.pdf (Detailed directed graph with arrows)")
    print("\\nThis demonstrates the individual and combined benefits of network and augmented weight optimization.")
    print(f"\\nUsing N={N} agents makes the network topology much clearer for publication figures!")
    print("\\nThe detailed digraph visualization clearly shows the directed nature with prominent arrows!")
