import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ============================================================================
# ENHANCED REPUTATION-BASED CONSENSUS WITH DETAILED VISUALIZATION
# ============================================================================

# ============================================================================
# CONFIGURATION - Easy to modify for different tests
# ============================================================================

# CHOOSE NETWORK SIZE: Set to True for small network, False for large network
USE_SMALL_NETWORK = False

if USE_SMALL_NETWORK:
    # ==== SMALL 4-AGENT NETWORK ====7
    initial_states = [0.2, 1.0, 0.0, 0.6]
    attacked = [True, False, False, False]  # Agent 0 is attacker
    indices = [1, 2, 3, 4]  # 1-indexed for plotting
    att_v = {0: 0.8}  # Attacked agent with forced value
    
    neighbors = [
        [1, 2, 3],       # Agent 0 connects to 1,2,3
        [0, 2, 3],       # Agent 1 connects to 0,2,3   
        [0, 1, 3],       # Agent 2 connects to 0,1,3
        [0, 1, 2],       # Agent 3 connects to 0,1,2
    ]
    
    max_iter = 30
    f = 1  # For 4 agents, f=1 is appropriate (removes 1 min + 1 max, leaves 2 values)
    epsilon = 0.1
    
    print("🔬 USING SMALL 4-AGENT NETWORK FOR TESTING")
    
else:
    # ==== LARGE 10-AGENT NETWORK (ORIGINAL) ====
    initial_states = [2, 4.8, 3.7, 3.5, 2.8, 2.5, 0.5, 4.7, 1, 3.6]
    attacked = [True, False, False, False, False, False, False, False, True, False]
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1-indexed for plotting
    att_v = {0: 2, 8: 1}  # Attacked agents with forced values

    neighbors = [
        [9, 1, 4, 7, 6],      
        [0, 9, 2, 3, 8, 5, 6], 
        [3, 9, 5, 1, 6, 4],
        [2, 9, 5, 8, 4, 6, 1],
        [0, 6, 8, 7, 9, 1],
        [3, 8, 7, 6, 1, 9, 4, 2],
        [5, 8, 7, 4, 1, 3, 0],
        [8, 6, 4, 0, 5],
        [7, 5, 3, 6, 4, 1, 9],
        [2, 0, 3, 1, 5, 8, 6, 4]  
    ]
    
    max_iter = 30
    f = 2  # For 10 agents, f=2 is appropriate for Byzantine fault tolerance
    epsilon = 0.1
    
    print("🏗️ USING LARGE 10-AGENT NETWORK (ORIGINAL)")

print(f"Testing with {len([i for i, att in enumerate(attacked) if att])} attacker(s): {[i for i, att in enumerate(attacked) if att]}")

print("\n🔍 ENHANCED REPUTATION-BASED CONSENSUS DEMONSTRATION")
print("="*70)
print("📊 Network Configuration:")
print(f"   • Agents: {len(initial_states)}")
print(f"   • Attacked agents: {[i for i, att in enumerate(attacked) if att]}")
print(f"   • Initial states: {initial_states}")
print(f"   • Robustness parameter f: {f}")
print(f"   • Confidence factor ε: {epsilon}")
if USE_SMALL_NETWORK:
    print("   • Network type: Small (4 agents)")
else:
    print("   • Network type: Large (10 agents)")
print("="*70)

# ============================================================================
# ENHANCED REPUTATION-BASED CONSENSUS WITH DETAILED VISUALIZATION
# ============================================================================

# ============================================================================
# ENHANCED REPUTATION FUNCTIONS WITH DETAILED LOGGING
# ============================================================================

def min_f(values, f):
    """
    Find the f-th smallest value after removing f maximum and f minimum values.
    This implements proper Byzantine fault tolerance.
    """
    if len(values) <= 2*f:
        # Not enough values for f-robust selection, return minimum
        return min(values)
    
    sorted_vals = sorted(values)
    # Remove f smallest and f largest values (Byzantine fault tolerance)
    trimmed = sorted_vals[f:-f] if f > 0 else sorted_vals
    
    if len(trimmed) == 0:
        return min(values)
    
    # Return the minimum of the remaining values
    return min(trimmed)

def compute_raw_reputation(agent_i, neighbor_j, current_states, neighbors, verbose=False):
    """
    Compute raw reputation of neighbor j as seen by agent i.
    
    Formula: raw_rep = 1 - (1/|N_i|) * Σ_{v∈N_i} |x_j - x_v|
    
    This measures how "consistent" neighbor j is with other neighbors.
    """
    Ni = neighbors[agent_i]
    distances = []
    
    if verbose:
        print(f"    Computing raw reputation for agent {neighbor_j} as seen by agent {agent_i}")
        print(f"    Agent {agent_i}'s neighbors: {Ni}")
    
    total_distance = 0
    for neighbor_v in Ni:
        distance = abs(current_states[neighbor_j] - current_states[neighbor_v])
        distances.append(distance)
        total_distance += distance
        
        if verbose:
            print(f"      |x_{neighbor_j} - x_{neighbor_v}| = |{current_states[neighbor_j]:.3f} - {current_states[neighbor_v]:.3f}| = {distance:.3f}")
    
    avg_distance = total_distance / len(Ni)
    raw_reputation = 1 - avg_distance
    
    if verbose:
        print(f"    Average distance: {avg_distance:.3f}")
        print(f"    Raw reputation: 1 - {avg_distance:.3f} = {raw_reputation:.3f}")
    
    return raw_reputation, distances

def update_reputations(x, c_old, neighbors, iteration, f, epsilon, verbose=False):
    """
    Update reputation matrix with detailed explanations.
    
    Process:
    1. Compute raw reputations based on consistency
    2. Normalize using robust statistics (ignore f outliers)
    3. Apply confidence factor for zero reputations
    """
    N = len(x)
    c_next = np.zeros((N, N))
    
    if verbose:
        print(f"\n🔄 REPUTATION UPDATE - Iteration {iteration}")
        print("="*50)
    
    for i in range(N):
        Ni = neighbors[i]
        raw_reps = {}
        
        if verbose:
            print(f"\n👤 Agent {i} evaluating neighbors {Ni}")
            print(f"   Current states: {[f'{x[j]:.3f}' for j in Ni]}")
        
        # Step 1: Compute raw reputations
        for j in Ni:
            raw_rep, distances = compute_raw_reputation(i, j, x, neighbors, verbose=verbose)
            raw_reps[j] = raw_rep
        
        # Step 2: Robust normalization
        raw_values = list(raw_reps.values())
        max_raw = max(raw_values)
        min_raw = min_f(raw_values, f)
        
        if verbose:
            print(f"\n   📊 Normalization for agent {i}:")
            print(f"      Raw values: {[f'{v:.3f}' for v in raw_values]}")
            print(f"      Max raw: {max_raw:.3f}")
            print(f"      Min raw (after f-robust): {min_raw:.3f}")
        
        denom = max_raw - min_raw if (max_raw - min_raw) != 0 else 1e-9
        
        # Step 3: Normalize and apply confidence
        for j in Ni:
            if i == j:
                c_norm = 1.0  # Self-trust
            else:
                c_norm = (raw_reps[j] - min_raw) / denom
                c_norm = max(0, min(c_norm, 1))
            
            if c_norm > 0:
                c_next[i, j] = c_norm
                if verbose:
                    print(f"      Agent {j}: norm_rep = {c_norm:.3f} → trust = {c_norm:.3f}")
            else:
                confidence_penalty = epsilon**(iteration+1)
                c_next[i, j] = confidence_penalty
                if verbose:
                    print(f"      Agent {j}: norm_rep = 0 → trust = ε^{iteration+1} = {confidence_penalty:.4e}")
    
    return c_next

def update_consensus(x, c, neighbors, verbose=False):
    """
    Update consensus using weighted average based on trust.
    
    Formula: x_i^(k+1) = (Σ_{j∈N_i} c[i,j] * x_j) / (Σ_{j∈N_i} c[i,j])
    """
    N = len(x)
    x_next = np.zeros(N)
    
    if verbose:
        print(f"\n🎯 CONSENSUS UPDATE")
        print("="*30)
    
    for i in range(N):
        Ni = neighbors[i]
        numerator = 0.0
        denominator = 0.0
        
        if verbose:
            print(f"\n👤 Agent {i} consensus update:")
        
        for j in Ni:
            weight = c[i, j]
            contribution = weight * x[j]
            numerator += contribution
            denominator += weight
            
            if verbose:
                print(f"   Agent {j}: trust={weight:.3f}, state={x[j]:.3f}, contribution={contribution:.3f}")
        
        if denominator < 1e-9:
            x_next[i] = x[i]
            if verbose:
                print(f"   → No trust, keeping current state: {x[i]:.3f}")
        else:
            x_next[i] = numerator / denominator
            if verbose:
                print(f"   → New state: {numerator:.3f}/{denominator:.3f} = {x_next[i]:.3f}")
    
    return x_next

# ============================================================================
# STEP-BY-STEP DEMONSTRATION FUNCTIONS FOR THESIS
# ============================================================================

def demonstrate_algorithm_steps(x, neighbors, reputation_matrix, iteration, attacked_agents=None):
    """
    Demonstrate each step of the reputation-based consensus algorithm in table format.
    Perfect for thesis presentations and detailed algorithm explanation.
    
    Args:
        x: Current states of all agents
        neighbors: Adjacency matrix or neighbor relationships
        reputation_matrix: Current reputation/trust matrix
        iteration: Current iteration number
        attacked_agents: List of attacked agent indices
    """
    n_agents = len(x)
    if attacked_agents is None:
        attacked_agents = []
    
    print("\n" + "="*100)
    print(f"REPUTATION-BASED CONSENSUS ALGORITHM - ITERATION {iteration}")
    print("="*100)
    
    # Step 1: Show Initial States and Reputation Matrix
    print(f"\n🔸 STEP 1: INITIAL CONDITIONS")
    print("-" * 60)
    
    # Create states table
    states_data = []
    for i in range(n_agents):
        agent_type = "ATTACKER" if i in attacked_agents else "HONEST"
        states_data.append([f"Agent {i}", f"{x[i]:.4f}", agent_type])
    
    print("CURRENT AGENT STATES:")
    print(format_table(states_data, headers=["Agent", "State Value", "Type"]))
    
    # Show reputation matrix in readable format
    print(f"\nREPUTATION MATRIX (Trust Levels):")
    reputation_data = []
    for i in range(n_agents):
        row = [f"Agent {i}"] + [f"{reputation_matrix[i,j]:.3f}" for j in range(n_agents)]
        reputation_data.append(row)
    
    headers = ["Evaluator"] + [f"→ Agent {j}" for j in range(n_agents)]
    print(format_table(reputation_data, headers=headers))
    
    # Step 2: Calculate Weighted Contributions
    print(f"\n🔸 STEP 2: CALCULATE WEIGHTED CONTRIBUTIONS")
    print("-" * 60)
    
    weighted_data = []
    for i in range(n_agents):
        print(f"\nFor Agent {i}:")
        agent_contributions = []
        
        # Get neighbors of agent i (from neighbor list)
        for j in neighbors[i]:  # Iterate through neighbors of i
            trust = reputation_matrix[i,j]
            neighbor_state = x[j]
            weighted_contribution = trust * neighbor_state
            
            agent_contributions.append([
                f"Agent {j}", 
                f"{neighbor_state:.4f}", 
                f"{trust:.3f}", 
                f"{weighted_contribution:.4f}"
            ])
        
        if agent_contributions:
            print(f"  Contributions from neighbors:")
            print("  " + format_table(agent_contributions, 
                                   headers=["Neighbor", "State", "Trust", "Weighted Contrib"]))
            weighted_data.extend(agent_contributions)
    
    # Step 3: Normalization Process
    print(f"\n🔸 STEP 3: NORMALIZATION PROCESS")
    print("-" * 60)
    
    normalization_data = []
    for i in range(n_agents):
        numerator = 0
        denominator = 0
        
        neighbor_count = 0
        # Get neighbors of agent i (from neighbor list)
        for j in neighbors[i]:  # Iterate through neighbors of i
            trust = reputation_matrix[i,j]
            numerator += trust * x[j]
            denominator += trust
            neighbor_count += 1
        
        if denominator > 0:
            new_state = numerator / denominator
        else:
            new_state = x[i]  # Keep current state if no trusted neighbors
        
        normalization_data.append([
            f"Agent {i}",
            f"{numerator:.4f}",
            f"{denominator:.4f}",
            f"{new_state:.4f}",
            str(neighbor_count)
        ])
    
    print("NORMALIZATION CALCULATIONS:")
    print(format_table(normalization_data, 
                      headers=["Agent", "Numerator", "Denominator", "New State", "Neighbors"]))
    
    # Step 4: State Update and Consensus Progress
    print(f"\n🔸 STEP 4: STATE UPDATE & CONSENSUS PROGRESS")
    print("-" * 60)
    
    # Calculate new states
    x_new = np.zeros(n_agents)
    convergence_data = []
    
    for i in range(n_agents):
        numerator = 0
        denominator = 0
        
        # Get neighbors of agent i (from neighbor list)
        for j in neighbors[i]:  # Iterate through neighbors of i
            trust = reputation_matrix[i,j]
            numerator += trust * x[j]
            denominator += trust
        
        if denominator > 0:
            x_new[i] = numerator / denominator
        else:
            x_new[i] = x[i]
        
        state_change = abs(x_new[i] - x[i])
        convergence_data.append([
            f"Agent {i}",
            f"{x[i]:.4f}",
            f"{x_new[i]:.4f}",
            f"{state_change:.4f}",
            "✓" if state_change < 0.001 else "○"
        ])
    
    print("STATE UPDATES:")
    print(format_table(convergence_data, 
                      headers=["Agent", "Old State", "New State", "Change", "Converged"]))
    
    # Summary statistics
    print(f"\n📊 ITERATION SUMMARY:")
    print("-" * 40)
    max_change = max(abs(x_new[i] - x[i]) for i in range(n_agents))
    avg_state = np.mean(x_new)
    honest_avg = np.mean([x_new[i] for i in range(n_agents) if i not in attacked_agents])
    
    summary_data = [
        ["Maximum State Change", f"{max_change:.6f}"],
        ["Average State (All)", f"{avg_state:.4f}"],
        ["Average State (Honest)", f"{honest_avg:.4f}"],
        ["Converged?", "YES" if max_change < 0.001 else "NO"]
    ]
    
    print(format_table(summary_data, headers=["Metric", "Value"]))
    
    return x_new

def format_table(data, headers=None):
    """
    Format data as a table using simple ASCII formatting.
    This avoids the need for external tabulate library.
    """
    if not data:
        return "No data to display"
    
    # Calculate column widths
    if headers:
        all_data = [headers] + data
    else:
        all_data = data
    
    col_widths = []
    for col in range(len(all_data[0])):
        max_width = max(len(str(row[col])) for row in all_data)
        col_widths.append(max_width + 2)  # Add padding
    
    # Create table
    table_str = ""
    
    # Headers
    if headers:
        header_row = "|".join(f" {headers[i]:<{col_widths[i]-1}}" for i in range(len(headers)))
        table_str += header_row + "\n"
        table_str += "|".join("-" * col_widths[i] for i in range(len(headers))) + "\n"
    
    # Data rows
    for row in data:
        data_row = "|".join(f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(row)))
        table_str += data_row + "\n"
    
    return table_str

def demonstrate_single_step_example():
    """
    Create a simple example for thesis demonstration with fixed values.
    This creates clear, clean tables perfect for thesis presentation.
    """
    print("\n" + "="*80)
    print("REPUTATION-BASED CONSENSUS ALGORITHM - STEP-BY-STEP EXAMPLE")
    print("="*80)
    
    # Example scenario: 5 agents, agent 4 is attacker
    n_agents = 5
    x = np.array([0.1, 0.8, 0.3, 0.6, 0.9])  # Current states
    attacked_agents = [4]
    
    # Simple connected network
    neighbors = np.array([
        [0, 1, 1, 0, 1],  # Agent 0 connects to 1,2,4
        [1, 0, 1, 1, 0],  # Agent 1 connects to 0,2,3
        [1, 1, 0, 1, 1],  # Agent 2 connects to all
        [0, 1, 1, 0, 1],  # Agent 3 connects to 1,2,4
        [1, 0, 1, 1, 0]   # Agent 4 connects to 0,2,3
    ])
    
    # Reputation matrix (trust levels)
    reputation_matrix = np.array([
        [1.0, 0.8, 0.9, 0.0, 0.2],  # Agent 0 trusts: high 1,2, low 4
        [0.8, 1.0, 0.7, 0.8, 0.1],  # Agent 1 trusts: most, distrusts 4
        [0.9, 0.7, 1.0, 0.8, 0.3],  # Agent 2 trusts: mostly, some doubt about 4
        [0.0, 0.8, 0.8, 1.0, 0.1],  # Agent 3 trusts: 1,2, distrusts 0,4
        [0.5, 0.5, 0.5, 0.5, 1.0]   # Agent 4 (attacker): neutral to others
    ])
    
    print(f"\n🔸 STEP 1: INITIAL CONDITIONS")
    print("-" * 50)
    
    # Show current states
    states_data = []
    for i in range(n_agents):
        agent_type = "🔴 ATTACKER" if i in attacked_agents else "🟢 HONEST"
        states_data.append([f"Agent {i}", f"{x[i]:.3f}", agent_type])
    
    print("CURRENT AGENT STATES:")
    print(format_table(states_data, headers=["Agent", "State", "Type"]))
    
    print(f"\n🔸 STEP 2: TRUST/REPUTATION MATRIX")
    print("-" * 50)
    print("(How much each agent trusts others - Row evaluates Column)")
    
    reputation_data = []
    for i in range(n_agents):
        row = [f"Agent {i}"] + [f"{reputation_matrix[i,j]:.2f}" for j in range(n_agents)]
        reputation_data.append(row)
    
    headers = ["Evaluator"] + [f"Agent {j}" for j in range(n_agents)]
    print(format_table(reputation_data, headers=headers))
    
    print(f"\n🔸 STEP 3: WEIGHTED CONSENSUS CALCULATION")
    print("-" * 50)
    print("Example: How Agent 0 updates its state")
    
    # Detailed calculation for Agent 0
    i = 0
    calculation_data = []
    numerator = 0
    denominator = 0
    
    print(f"\nAgent {i} considers its neighbors:")
    for j in range(n_agents):
        if i != j and neighbors[i,j] > 0:
            trust = reputation_matrix[i,j]
            neighbor_state = x[j]
            weighted_contrib = trust * neighbor_state
            numerator += weighted_contrib
            denominator += trust
            
            calculation_data.append([
                f"Agent {j}",
                f"{neighbor_state:.3f}",
                f"{trust:.2f}",
                f"{weighted_contrib:.4f}"
            ])
    
    print(format_table(calculation_data, 
                      headers=["Neighbor", "State", "Trust", "Contribution"]))
    
    new_state = numerator / denominator if denominator > 0 else x[i]
    
    print(f"\nFINAL CALCULATION:")
    calc_summary = [
        ["Numerator (sum of contributions)", f"{numerator:.4f}"],
        ["Denominator (sum of trust)", f"{denominator:.2f}"],
        ["New State", f"{new_state:.4f}"],
        ["Old State", f"{x[i]:.3f}"],
        ["Change", f"{abs(new_state - x[i]):.4f}"]
    ]
    print(format_table(calc_summary, headers=["Component", "Value"]))
    
    print(f"\n🔸 STEP 4: COMPLETE STATE UPDATE")
    print("-" * 50)
    
    # Calculate all new states
    x_new = np.zeros(n_agents)
    update_data = []
    
    for i in range(n_agents):
        numerator = 0
        denominator = 0
        
        for j in range(n_agents):
            if i != j and neighbors[i,j] > 0:
                trust = reputation_matrix[i,j]
                numerator += trust * x[j]
                denominator += trust
        
        if denominator > 0:
            x_new[i] = numerator / denominator
        else:
            x_new[i] = x[i]
        
        change = abs(x_new[i] - x[i])
        update_data.append([
            f"Agent {i}",
            f"{x[i]:.3f}",
            f"{x_new[i]:.3f}",
            f"{change:.4f}",
            "🟢 HONEST" if i not in attacked_agents else "🔴 ATTACKER"
        ])
    
    print("ALL AGENT UPDATES:")
    print(format_table(update_data, 
                      headers=["Agent", "Old State", "New State", "Change", "Type"]))
    
    print(f"\n📊 CONSENSUS PROGRESS:")
    print("-" * 30)
    
    # Show consensus metrics
    honest_states = [x_new[i] for i in range(n_agents) if i not in attacked_agents]
    consensus_data = [
        ["Average (All Agents)", f"{np.mean(x_new):.4f}"],
        ["Average (Honest Only)", f"{np.mean(honest_states):.4f}"],
        ["Standard Deviation", f"{np.std(x_new):.4f}"],
        ["Max Change This Step", f"{max(abs(x_new[i] - x[i]) for i in range(n_agents)):.4f}"]
    ]
    
    print(format_table(consensus_data, headers=["Metric", "Value"]))
    
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 20)
    print(f"• Honest agents (0,1,2,3) have learned to distrust Agent 4")
    print(f"• Agent 4 (attacker) has low trust scores from others")
    print(f"• Honest agents converge towards similar values")
    print(f"• Attacker's influence is limited by low reputation")

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def plot_state_evolution_enhanced(states, reputation_data, correct_consensus, att_v, agent_ids):
    """
    Enhanced state evolution plot with reputation insights.
    """
    iterations = np.arange(states.shape[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: State Evolution
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
    
    for col, agent in enumerate(agent_ids):
        agent_idx = agent - 1  # Convert to 0-indexed
        
        if agent_idx in [i for i, att in enumerate(attacked) if att]:
            ax1.plot(iterations, states[:, col], 
                    color='red', marker='v', linestyle='--', linewidth=3,
                    label=f'⚠️ Attacked Agent {agent}', markersize=8, alpha=0.8)
        else:
            ax1.plot(iterations, states[:, col], 
                    color=colors[col], marker='o', linestyle='-', 
                    label=f'👤 Honest Agent {agent}', markersize=6, alpha=0.8)
    
    ax1.axhline(correct_consensus, color='black', linestyle=':', linewidth=3,
               label=f"🎯 True Consensus ({correct_consensus:.3f})")
    
    ax1.set_title('State Evolution with Reputation-Based Consensus', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('State Value', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Reputation of Attacked vs Honest Agents
    attacked_indices = [i for i, att in enumerate(attacked) if att]
    honest_indices = [i for i, att in enumerate(attacked) if not att]
    
    avg_rep_attacked = []
    avg_rep_honest = []
    
    for rep_matrix in reputation_data:
        # Average reputation received by attacked agents (excluding self-reputation)
        attacked_reps = []
        for attacker in attacked_indices:
            reps = [rep_matrix[i, attacker] for i in range(len(rep_matrix)) if i != attacker and rep_matrix[i, attacker] > 0]
            if reps:
                attacked_reps.extend(reps)
        
        # Average reputation received by honest agents
        honest_reps = []
        for honest in honest_indices:
            reps = [rep_matrix[i, honest] for i in range(len(rep_matrix)) if i != honest and rep_matrix[i, honest] > 0]
            if reps:
                honest_reps.extend(reps)
        
        avg_rep_attacked.append(np.mean(attacked_reps) if attacked_reps else 0)
        avg_rep_honest.append(np.mean(honest_reps) if honest_reps else 0)
    
    ax2.plot(range(len(avg_rep_attacked)), avg_rep_attacked, 
             'r-o', linewidth=3, markersize=8, label='⚠️ Average Reputation of Attacked Agents')
    ax2.plot(range(len(avg_rep_honest)), avg_rep_honest, 
             'b-o', linewidth=3, markersize=8, label='👤 Average Reputation of Honest Agents')
    
    ax2.set_title('Reputation Discrimination Over Time', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('Average Reputation Score', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()

def plot_network_and_reputation(neighbors, att_v, title="Reputation Network Topology", save_path=None):
    """
    Create a publication-quality network topology visualization matching W-MSR style.
    
    Parameters:
      - neighbors: List of neighbor lists for each agent
      - att_v: Dictionary of attacked agents
      - title: Plot title
      - save_path: Optional path to save the figure (e.g., 'network.pdf', 'network.png')
    """
    import networkx as nx
    
    # Create NetworkX DIRECTED graph
    G = nx.DiGraph()
    num_agents = len(neighbors)
    
    # Add nodes
    for i in range(num_agents):
        G.add_node(i)
    
    # Add DIRECTED edges based on neighbor lists
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            if neighbor < num_agents:  # Ensure valid neighbor index
                # Agent i receives information FROM neighbor j
                G.add_edge(neighbor, i)
    
    # Set up the plot with high DPI for publication quality
    plt.figure(figsize=(20, 12), dpi=300)
    
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
            node_sizes.append(900)         # Larger size for attackers
        else:
            node_colors.append('#4444FF')  # Blue for honest agents
            node_sizes.append(900)         # Normal size for honest agents
    
    # Draw the network with DIRECTED edges
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Draw directed edges with arrows
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=3,
                          alpha=0.7,
                          arrows=True,           # Show arrows for direction
                          arrowsize=18,          # Arrow size
                          arrowstyle='->')       # Arrow style
    
    # Add labels with better formatting
    labels = {i: f'{i}' for i in range(num_agents)}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=18,
                           font_weight='bold',
                           font_color='white')
    
    # Create custom legend
    honest_patch = plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='#4444FF', markersize=18, 
                             label='Honest Agents', markeredgecolor='black', markeredgewidth=1)
    attack_patch = plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#FF4444', markersize=18,
                             label='Attacked Agents', markeredgecolor='black', markeredgewidth=1)
    
    plt.legend(handles=[honest_patch, attack_patch], 
              loc='upper right', fontsize=18, frameon=True, fancybox=True, shadow=True)
    
    # Formatting for publication quality
    plt.title(title + " (Directed Graph)", fontsize=24, fontweight='bold', pad=20)
    plt.axis('off')  # Remove axes for cleaner look
    
    # Add network statistics as text
    stats_text = f"Nodes: {G.number_of_nodes()}\n"
    stats_text += f"Edges: {G.number_of_edges()}\n"
    stats_text += f"Attackers: {len(att_v)}\n"
    stats_text += f"Density: {nx.density(G):.3f}\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Directed network topology saved to: {save_path}")
    
    plt.show()
    
    # Print network analysis for directed graph
    print("\n" + "="*50)
    print("REPUTATION NETWORK TOPOLOGY ANALYSIS")
    print("="*50)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    print(f"Attacked agents: {list(att_v.keys())}")
    print("="*50)

# ============================================================================
# ENHANCED SIMULATION WITH DETAILED TRACKING
# ============================================================================

def run_enhanced_reputation_simulation(initial_states, attacked, max_iter, f, epsilon, att_v, neighbors, verbose=True):
    """
    Enhanced simulation with detailed tracking and explanations.
    """
    N = len(initial_states)
    x = np.array(initial_states, dtype=float)
    
    # Initialize reputation matrix
    c = np.zeros((N, N))
    for i in range(N):
        for j in neighbors[i]:
            c[i, j] = 1.0
    
    all_states = [x.copy()]
    all_reputations = [c.copy()]
    attacker_indices = [i for i, att in enumerate(attacked) if att]
    
    if verbose:
        print(f"\n🚀 Starting Enhanced Reputation Simulation")
        print(f"Initial states: {[f'{s:.3f}' for s in x]}")
        print(f"Attackers will be forced to: {att_v}")
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n" + "="*70)
            print(f"🔄 ITERATION {iteration + 1}")
            print("="*70)
        
        # Update reputations
        c_new = update_reputations(x, c, neighbors, iteration, f=f, epsilon=epsilon, verbose=verbose)
        
        # Update consensus
        x_next = update_consensus(x, c_new, neighbors, verbose=verbose)
        
        # Demonstrate step-by-step process for first few iterations (for thesis)
        if iteration < 2:  # Show detailed steps for first 2 iterations
            demonstrate_algorithm_steps(x, neighbors, c_new, iteration, 
                                      attacked_agents=attacker_indices)
        
        # Force attacked agents
        if att_v is not None:
            for attacker in attacker_indices:
                if attacker in att_v:
                    old_value = x_next[attacker]
                    x_next[attacker] = att_v[attacker]
                    if verbose:
                        print(f"\n🚨 Forcing attacker {attacker}: {old_value:.3f} → {att_v[attacker]:.3f}")
        
        all_states.append(x_next.copy())
        all_reputations.append(c_new.copy())
        x = x_next.copy()
        c = c_new.copy()
        
        if verbose:
            print(f"\n📊 End of iteration {iteration + 1}:")
            print(f"   States: {[f'{s:.3f}' for s in x]}")
            honest_states = [x[i] for i in range(N) if not attacked[i]]
            print(f"   Honest consensus: {np.mean(honest_states):.3f}")
    
    return np.array(all_states), all_reputations

# ============================================================================
# RUN ENHANCED SIMULATION
# ============================================================================

if __name__ == "__main__":
    # Run the enhanced simulation
    states, reputation_data = run_enhanced_reputation_simulation(
        initial_states, attacked, max_iter, f, epsilon, att_v, neighbors, verbose=True
    )
    
    correct_consensus = np.mean([initial_states[i] for i in range(len(initial_states)) if not attacked[i]])
    
    print(f"\n" + "="*70)
    print(f"📈 SIMULATION COMPLETE")
    print("="*70)
    print(f"✅ Correct consensus (honest agents): {correct_consensus:.3f}")
    print(f"🎯 Final honest consensus: {np.mean([states[-1][i] for i in range(len(states[-1])) if not attacked[i]]):.3f}")
    print(f"⚠️  Final attacked values: {[f'{states[-1][i]:.3f}' for i in range(len(states[-1])) if attacked[i]]}")
    
    # Generate all visualizations
    print(f"\n🎨 Generating enhanced visualizations...")
    
    # For thesis demonstration: Show step-by-step algorithm breakdown
    print(f"\n📚 GENERATING THESIS DEMONSTRATION...")
    demonstrate_single_step_example()
    
    # 1. State evolution with reputation insights
    plot_state_evolution_enhanced(states, reputation_data, correct_consensus, att_v, indices)
    
    # 2. Network topology visualization (matching W-MSR style)
    plot_network_and_reputation(neighbors, att_v, 
                                title="Reputation-Based Consensus Network", 
                                save_path="reputation_network.pdf")
    
    print(f"✨ All visualizations generated successfully!")
