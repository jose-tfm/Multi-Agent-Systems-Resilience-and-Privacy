import numpy as np
import sys
import os
import time
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.utils import plot_state
from utils.enhanced_plots import plot_enhanced_msr, plot_convergence_rate, plot_network_topology


# Define the network topology with improved documentation
neighbors = [
    [9, 1, 4, 7, 6],        # Agent 0 neighbors
    [0, 9, 2, 3, 8, 5, 6],  # Agent 1 neighbors  
    [3, 9, 5, 1, 6, 4],     # Agent 2 neighbors
    [2, 9, 5, 8, 4, 6, 1],  # Agent 3 neighbors
    [0, 6, 8, 7, 9, 1],     # Agent 4 neighbors
    [3, 8, 7, 6, 1, 9, 4, 2], # Agent 5 neighbors
    [5, 8, 7, 4, 1, 3, 0],  # Agent 6 neighbors
    [8, 6, 4, 0],           # Agent 7 neighbors
    [7, 5, 3, 6, 4, 1, 9],  # Agent 8 neighbors
    [2, 0, 3, 1, 5, 8, 6, 4] # Agent 9 neighbors
]

# Define attacked agent values with more sophisticated attack patterns
att_v = {
    0: (lambda t: 0.3),  # Constant attack
    8: (lambda t: 0.8)   # Constant attack
    # You can add time-varying attacks like:
    # 0: (lambda t: 0.3 + 0.1 * np.sin(t * 0.1)),  # Sinusoidal attack
    # 8: (lambda t: min(0.9, 0.5 + t * 0.01))      # Gradually increasing attack
}

# Initial state for agents
x0 = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.5, 0.4, 0.6, 0.3]
indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Agent IDs for plotting
max_iterations = 30
f = 2  # Number of outliers to remove from each end


def analyze_network_properties(neighbors):
    """
    Analyze network properties relevant to consensus algorithms.
    """
    num_agents = len(neighbors)
    
    # Calculate degree distribution
    degrees = [len(neighbor_list) for neighbor_list in neighbors]
    avg_degree = np.mean(degrees)
    
    # Calculate connectivity
    adjacency = np.zeros((num_agents, num_agents))
    for i, neighbor_list in enumerate(neighbors):
        for j in neighbor_list:
            if j < num_agents:  # Bounds check
                adjacency[i, j] = 1
    
    # Check strong connectivity (simplified check)
    is_connected = True
    for i in range(num_agents):
        if len(neighbors[i]) == 0:
            is_connected = False
            break
    
    print(f"=== Network Analysis ===")
    print(f"Number of agents: {num_agents}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Degree distribution: {degrees}")
    print(f"Min degree: {min(degrees)}, Max degree: {max(degrees)}")
    print(f"Network appears connected: {is_connected}")
    print(f"Total edges: {sum(degrees) // 2}")  # Each edge counted twice
    print()


def msr_consensus_enhanced(x0, neighbors, max_iterations, f, attck_vals={}, verbose=True, tolerance=1e-6):
    """
    Enhanced M-MSR consensus algorithm with improved analysis and metrics.
    
    Parameters:
      - x0: Initial state values (list or numpy array)
      - neighbors: List of lists where neighbors[i] contains the indices of agents that are neighbors of agent i
      - max_iterations: Number of iterations to run
      - f: Number of outliers to remove from each agent's neighborhood (from both ends)
      - attck_vals: Dictionary with attacked agents as keys and functions that define faulty values over time
      - verbose: Whether to print detailed iteration information
      - tolerance: Convergence tolerance for early stopping
    
    Returns:
      - x_vals: List of state vectors over the iterations
      - metrics: Dictionary containing convergence metrics
    """
    num_agents = len(x0)
    x_vals = [np.array(x0, dtype=float)]
    
    # Initialize metrics tracking
    metrics = {
        'convergence_times': {},
        'consensus_errors': [],
        'inter_agent_variance': [],
        'iteration_times': [],
        'filtered_counts': defaultdict(list)
    }
    
    if verbose:
        print("=== Enhanced M-MSR Algorithm ===")
        print(f"Initial State: {x0}")
        print(f"Attacked agents: {list(attck_vals.keys())}")
        print(f"Filter parameter f: {f}")
        print()
    
    # Set initial state for attacked agents
    for a in attck_vals:
        x_vals[-1][a] = attck_vals[a](0)
    
    # Calculate target consensus (honest agents only)
    honest_agents = [i for i in range(num_agents) if i not in attck_vals]
    target_consensus = np.mean([x0[i] for i in honest_agents])
    
    for k in range(max_iterations):
        start_time = time.time()
        x_new = np.zeros(num_agents)
        
        if verbose:
            print(f"\n--- Iteration {k+1} ---")
        
        for i in range(num_agents):
            # Get neighbor indices including self
            inds = neighbors[i][:]
            if i not in inds:
                inds.append(i)
            
            # Collect values from neighbors
            vals = np.array([x_vals[-1][j] for j in inds])
            weights = np.ones(len(inds)) / len(inds)
            
            if verbose:
                print(f"  Agent {i}:")
                print(f"    Neighbors (including self): {inds}")
                print(f"    Values: {vals}")
            
            # Sort values and corresponding weights
            sort_order = np.argsort(vals)
            vals_sorted = vals[sort_order]
            weights_sorted = weights[sort_order]
            
            # Filter out the f lowest and f highest values (if possible)
            initial_count = len(vals_sorted)
            if f > 0 and len(vals_sorted) > 2 * f:
                filtered_vals = vals_sorted[f:-f]
                filtered_weights = weights_sorted[f:-f]
                filtered_count = len(filtered_vals)
                if verbose:
                    print(f"    Filtered out {2*f} values (f={f} from each end)")
            else:
                filtered_vals = vals_sorted
                filtered_weights = weights_sorted
                filtered_count = len(filtered_vals)
                if verbose:
                    print(f"    No filtering applied (insufficient values)")
            
            # Track filtering statistics
            metrics['filtered_counts'][i].append(filtered_count / initial_count)
            
            # Update state by computing the weighted average
            if np.sum(filtered_weights) > 0:
                x_new[i] = np.sum(filtered_vals * filtered_weights) / np.sum(filtered_weights)
            else:
                x_new[i] = np.mean(filtered_vals)
            
            if verbose:
                print(f"    New state: {x_new[i]:.6f}")
        
        # Force attacked agents to follow their faulty values
        for a in attck_vals:
            x_new[a] = attck_vals[a](k+1)
            if verbose:
                print(f"    Attacked agent {a} forced to: {x_new[a]}")
        
        # Record iteration time
        iteration_time = time.time() - start_time
        metrics['iteration_times'].append(iteration_time)
        
        # Calculate metrics
        honest_states = [x_new[i] for i in honest_agents]
        consensus_error = abs(np.mean(honest_states) - target_consensus)
        inter_variance = np.var(honest_states)
        
        metrics['consensus_errors'].append(consensus_error)
        metrics['inter_agent_variance'].append(inter_variance)
        
        # Check for convergence of individual agents
        for i in honest_agents:
            if i not in metrics['convergence_times']:
                if abs(x_new[i] - target_consensus) < tolerance:
                    metrics['convergence_times'][i] = k + 1
        
        if verbose:
            print(f"  Final state: {x_new}")
            print(f"  Consensus error: {consensus_error:.6f}")
            print(f"  Inter-agent variance: {inter_variance:.6f}")
        
        x_vals.append(x_new.copy())
        
        # Early stopping if all honest agents converged
        if len(metrics['convergence_times']) == len(honest_agents):
            if verbose:
                print(f"\n*** All honest agents converged at iteration {k+1} ***")
            break
    
    return x_vals, metrics


def run_comprehensive_analysis():
    """
    Run comprehensive analysis of the M-MSR algorithm.
    """
    print("="*60)
    print("COMPREHENSIVE M-MSR ALGORITHM ANALYSIS")
    print("="*60)
    
    # Analyze network properties
    analyze_network_properties(neighbors)
    
    # Run the enhanced algorithm
    print("Running enhanced M-MSR algorithm...")
    results, metrics = msr_consensus_enhanced(
        x0, neighbors, max_iterations, f, 
        attck_vals=att_v, verbose=False, tolerance=1e-6
    )
    
    # Convert results to numpy array for plotting
    states = np.array(results)
    correct_consensus = np.average([x0[i] for i in range(len(x0)) if i not in att_v])
    
    print(f"\n=== Algorithm Performance Summary ===")
    print(f"Total iterations run: {len(results) - 1}")
    print(f"Target consensus value: {correct_consensus:.6f}")
    print(f"Final consensus error: {metrics['consensus_errors'][-1]:.6f}")
    print(f"Final inter-agent variance: {metrics['inter_agent_variance'][-1]:.6f}")
    print(f"Average iteration time: {np.mean(metrics['iteration_times'])*1000:.2f} ms")
    
    if metrics['convergence_times']:
        print(f"\nConvergence times:")
        for agent, time in metrics['convergence_times'].items():
            print(f"  Agent {agent}: {time} iterations")
        print(f"  Average convergence time: {np.mean(list(metrics['convergence_times'].values())):.1f} iterations")
    
    # Create visualizations
    print(f"\nGenerating enhanced visualizations...")
    
    # 1. Enhanced main plot
    plot_enhanced_msr(states, correct_consensus, att_v, indices, 
                     title="Enhanced W-MSR Algorithm Analysis")
    
    # 2. Network topology
    plot_network_topology(neighbors, att_v)
    
    # 3. Convergence analysis
    plot_convergence_rate(states, att_v, indices, tolerance=1e-3)
    
    # 4. Original plot for comparison
    print(f"\nOriginal plot for comparison:")
    plot_state(states, correct_consensus, att_v, indices, 
              title="Original W-MSR Plot", xlabel="Iteration", ylabel="State Value")
    
    return results, metrics


def run_parameter_study():
    """
    Study the effect of different parameters on algorithm performance.
    """
    print("\n" + "="*60)
    print("PARAMETER STUDY")
    print("="*60)
    
    f_values = [1, 2, 3]
    results_dict = {}
    
    for f_val in f_values:
        print(f"\nTesting f = {f_val}...")
        results, metrics = msr_consensus_enhanced(
            x0, neighbors, max_iterations, f_val, 
            attck_vals=att_v, verbose=False
        )
        
        results_dict[f"f={f_val}"] = (np.array(results), att_v, indices)
        
        print(f"  Final consensus error: {metrics['consensus_errors'][-1]:.6f}")
        print(f"  Convergence time: {len(results)-1} iterations")
    
    # Create comparison plot
    from utils.enhanced_plots import create_comparison_plot
    create_comparison_plot(results_dict)


if __name__ == "__main__":
    # Run comprehensive analysis
    results, metrics = run_comprehensive_analysis()
    
    # Run parameter study
    run_parameter_study()
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
