#!/usr/bin/env python3
"""
Enhanced General Algorithm 1 with Detection Diagnostics for PLOS ONE
===================================================================

This script enhances the basic General Algorithm 1 implementation with:
1. Detailed detection statistics tracking
2. Mechanism diagnostic plots
3. Publication-quality visualization
4. Statistical analysis of detection performance

Authors: [Your name]
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys
from typing import Dict, List, Tuple, Any

# Publication-quality matplotlib configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0
})

class GeneralAlgorithmDiagnostics:
    """Enhanced General Algorithm 1 with comprehensive diagnostics."""
    
    def __init__(self, agents: List[int], attackers: List[int], f: int, epsilon: float, T: int):
        self.agents = agents
        self.attackers = attackers
        self.honest_agents = [a for a in agents if a not in attackers]
        self.f = f
        self.epsilon = epsilon
        self.T = T
        
        # Generate candidate subsets F
        self.F_subsets = []
        for size in range(f + 1):
            for subset in itertools.combinations(agents, size):
                self.F_subsets.append(frozenset(subset))
        self.F_subsets = sorted(self.F_subsets, key=lambda s: len(s))
        self.subset_index = {S: i for i, S in enumerate(self.F_subsets)}
        
        # Initialize data structures
        self.c = {u: {} for u in agents}  # Candidate values
        self.x_values = {u: [] for u in agents}  # Selected states
        self.detection_stats = {u: [] for u in agents}  # Detection statistics
        self.candidate_diffs = {u: [] for u in agents}  # Candidate differences
        self.detection_events = []  # (time, agent, detected_subset)
        
        # Network topology (fully connected for simplicity)
        self.adjacency = {u: agents.copy() for u in agents}
    
    def attacked_value(self, agent: int, t: int) -> float:
        """Return attack value for compromised agents."""
        if agent in self.attackers:
            return 0.1  # Constant attack
        return None
    
    def consensus_candidate(self, u: int, t: int, S: frozenset) -> float:
        """Compute consensus candidate c_u^(t+1)[S]."""
        if u in self.attackers:
            return self.attacked_value(u, t)
        
        # Average over neighbors not in excluded set S
        included_agents = [v for v in self.adjacency[u] if v not in S]
        if not included_agents:
            return self.c[u][t][self.subset_index[S]]
        
        vals = [self.c[v][t][self.subset_index[S]] for v in included_agents]
        return np.mean(vals)
    
    def select_state_with_diagnostics(self, u: int, t: int) -> Tuple[float, Dict]:
        """Enhanced state selection with detailed diagnostics."""
        if u in self.attackers:
            return self.attacked_value(u, t), {}
        
        # Get full candidate (S = ∅)
        val_full = self.c[u][t][0]
        
        # Compute diagnostics for all candidates
        diagnostics = {
            'candidate_values': [],
            'candidate_diffs': [],
            'valid_candidates': [],
            'detection_score': 0,
            'selected_subset': frozenset(),
            'threshold_exceeded': False
        }
        
        # Analyze all candidates
        for idx, S in enumerate(self.F_subsets):
            cand_val = self.c[u][t][idx]
            diff = abs(cand_val - val_full)
            
            diagnostics['candidate_values'].append((S, cand_val))
            diagnostics['candidate_diffs'].append((S, diff))
            
            # Check if this candidate indicates potential attack
            if diff > self.epsilon:
                diagnostics['valid_candidates'].append((S, cand_val, diff))
                diagnostics['threshold_exceeded'] = True
        
        # Detection logic: exactly one valid candidate
        if len(diagnostics['valid_candidates']) == 1:
            selected_subset, selected_val, detection_score = diagnostics['valid_candidates'][0]
            diagnostics['selected_subset'] = selected_subset
            diagnostics['detection_score'] = detection_score
            
            # Record detection event
            self.detection_events.append((t, u, selected_subset, detection_score))
            
            return selected_val, diagnostics
        else:
            # No unique detection, use full candidate
            diagnostics['selected_subset'] = frozenset()
            return val_full, diagnostics
    
    def run_simulation(self, x_init: Dict[int, float]) -> Dict[str, Any]:
        """Run complete simulation with diagnostics."""
        # Initialize
        for u in self.agents:
            self.c[u][0] = [x_init[u]] * len(self.F_subsets)
            self.x_values[u] = [x_init[u]]
        
        # Simulation loop
        for t in range(self.T):
            # Update candidate values
            for u in self.agents:
                self.c[u][t+1] = [None] * len(self.F_subsets)
            
            for u in self.agents:
                for S in self.F_subsets:
                    idx = self.subset_index[S]
                    self.c[u][t+1][idx] = self.consensus_candidate(u, t, S)
            
            # State selection with diagnostics
            for u in self.agents:
                selected_val, diagnostics = self.select_state_with_diagnostics(u, t+1)
                self.x_values[u].append(selected_val)
                self.detection_stats[u].append(diagnostics)
                
                # Store candidate differences for analysis
                max_diff = max([diff for _, diff in diagnostics['candidate_diffs']])
                self.candidate_diffs[u].append(max_diff)
        
        # Compile results
        states = np.array([self.x_values[u] for u in self.agents]).T
        correct_consensus = np.mean([x_init[u] for u in self.honest_agents])
        
        return {
            'states': states,
            'correct_consensus': correct_consensus,
            'detection_events': self.detection_events,
            'detection_stats': self.detection_stats,
            'candidate_diffs': self.candidate_diffs,
            'x_values': self.x_values,
            'c_values': self.c
        }
    
    def create_diagnostic_plots(self, results: Dict[str, Any], save_figures: bool = True):
        """Create comprehensive diagnostic plots."""
        states = results['states']
        detection_events = results['detection_events']
        candidate_diffs = results['candidate_diffs']
        
        # Figure 1: Agent trajectories with detection events
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        iterations = np.arange(states.shape[0])
        
        # Plot agent trajectories
        honest_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
        attack_color = '#e74c3c'
        
        for i, agent in enumerate(self.agents):
            if agent in self.attackers:
                ax1.plot(iterations, states[:, i], '^--', color=attack_color, 
                        linewidth=3, markersize=8, label=f'Attacker {agent}', alpha=0.9)
            else:
                color_idx = self.honest_agents.index(agent) % len(honest_colors)
                ax1.plot(iterations, states[:, i], 'o-', color=honest_colors[color_idx],
                        linewidth=2.5, markersize=6, label=f'Honest Agent {agent}', alpha=0.9)
        
        # True consensus
        ax1.axhline(results['correct_consensus'], color='#2c3e50', linestyle=':', 
                   linewidth=2.5, label=f"True Consensus = {results['correct_consensus']:.2f}")
        
        # Mark detection events
        for t, agent, subset, score in detection_events:
            ax1.axvline(t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.annotate(f'Detect\\nt={t}', xy=(t, states[t, self.agents.index(agent)]), 
                        xytext=(t+0.5, states[t, self.agents.index(agent)]+0.1),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, color='red', weight='bold')
        
        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Agent State Value', fontweight='bold') 
        ax1.set_title('(a) Agent Trajectory Evolution with Attack Detection Events', 
                     fontweight='bold', pad=20)
        ax1.legend(loc='best', framealpha=1.0)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, len(iterations)-0.5)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig('general_alg1_trajectory_diagnostic.png', dpi=300, bbox_inches='tight')
            plt.savefig('general_alg1_trajectory_diagnostic.pdf', bbox_inches='tight')
            print("Saved: general_alg1_trajectory_diagnostic.png/.pdf")
        plt.show()
        
        # Figure 2: Detection statistics evolution
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top panel: Candidate differences vs threshold
        for i, agent in enumerate(self.agents):
            if agent in self.attackers:
                continue  # Skip attackers for detection analysis
            
            diffs = candidate_diffs[agent][1:]  # Skip initial value
            iterations_subset = np.arange(1, len(diffs) + 1)
            
            color_idx = self.honest_agents.index(agent) % len(honest_colors)
            ax2a.plot(iterations_subset, diffs, 'o-', color=honest_colors[color_idx],
                     linewidth=2, markersize=5, label=f'Agent {agent} Max Diff', alpha=0.8)
        
        # Detection threshold
        ax2a.axhline(self.epsilon, color='black', linestyle='--', linewidth=2, 
                    label=f'Detection Threshold ε = {self.epsilon:.2f}')
        
        ax2a.set_ylabel('Max Candidate Difference', fontweight='bold')
        ax2a.set_title('(b) Detection Statistics: Candidate Differences vs Threshold', 
                      fontweight='bold', pad=20)
        ax2a.legend(loc='best', framealpha=1.0)
        ax2a.grid(True, alpha=0.3)
        
        # Bottom panel: Detection score evolution
        detection_scores = np.zeros(self.T)
        for t, agent, subset, score in detection_events:
            if t <= self.T:
                detection_scores[t-1] = max(detection_scores[t-1], score)
        
        iterations_det = np.arange(1, self.T + 1)
        ax2b.plot(iterations_det, detection_scores, 'o-', color='#d62728', 
                 linewidth=2.5, markersize=6, label='Detection Score', alpha=0.9)
        ax2b.fill_between(iterations_det, 0, detection_scores, alpha=0.3, color='red')
        
        # Mark first detection
        if detection_events:
            first_detection_time = detection_events[0][0]
            ax2b.axvline(first_detection_time, color='red', linestyle='--', alpha=0.7, 
                        linewidth=2, label=f'First Detection (t={first_detection_time})')
        
        ax2b.set_xlabel('Iteration', fontweight='bold')
        ax2b.set_ylabel('Detection Score\\n(Candidate Diff - ε)', fontweight='bold')
        ax2b.legend(loc='best', framealpha=1.0)
        ax2b.grid(True, alpha=0.3)
        ax2b.set_xlim(0.5, self.T + 0.5)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig('general_alg1_detection_diagnostic.png', dpi=300, bbox_inches='tight')
            plt.savefig('general_alg1_detection_diagnostic.pdf', bbox_inches='tight')
            print("Saved: general_alg1_detection_diagnostic.png/.pdf")
        plt.show()
    
    def print_detection_analysis(self, results: Dict[str, Any]):
        """Print detailed detection analysis for publication."""
        detection_events = results['detection_events']
        
        print("\\n" + "="*70)
        print("DETECTION MECHANISM ANALYSIS")
        print("="*70)
        
        print(f"Algorithm parameters:")
        print(f"  - Agents: {self.agents}")
        print(f"  - Attackers: {self.attackers}")
        print(f"  - Byzantine tolerance: f = {self.f}")
        print(f"  - Detection threshold: ε = {self.epsilon:.3f}")
        print(f"  - Simulation time: T = {self.T}")
        
        print(f"\\nDetection events:")
        if detection_events:
            for i, (t, agent, subset, score) in enumerate(detection_events):
                print(f"  {i+1}. Time t={t}, Agent {agent} detected subset {set(subset)} "
                      f"(score={score:.3f})")
            
            first_detection = detection_events[0][0]
            print(f"\\nFirst detection occurred at iteration {first_detection}")
            print(f"Total detection events: {len(detection_events)}")
        else:
            print("  No detection events occurred")
        
        # Final convergence analysis
        final_states = [results['x_values'][u][-1] for u in self.honest_agents]
        consensus_error = np.std(final_states)
        
        print(f"\\nConvergence analysis:")
        print(f"  - True consensus: {results['correct_consensus']:.3f}")
        print(f"  - Final honest states: {[f'{s:.3f}' for s in final_states]}")
        print(f"  - Consensus error (std): {consensus_error:.6f}")
        
        max_error = max([abs(s - results['correct_consensus']) for s in final_states])
        print(f"  - Maximum error from true consensus: {max_error:.6f}")
        
        if consensus_error < 1e-3:
            print("  ✓ Algorithm achieved successful consensus")
        else:
            print("  ✗ Algorithm did not fully converge")

def main():
    """Main function demonstrating enhanced General Algorithm 1."""
    
    print("Enhanced General Algorithm 1 with Detection Diagnostics")
    print("=" * 60)
    
    # Experimental setup
    agents = [1, 2, 3]
    attackers = [3]
    f = 1
    epsilon = 0.05
    T = 8
    x_init = {1: 0.0, 2: 1.0, 3: 0.1}
    
    # Create algorithm instance
    alg = GeneralAlgorithmDiagnostics(agents, attackers, f, epsilon, T)
    
    print(f"\\nRunning simulation...")
    print(f"  - Agents: {agents}")
    print(f"  - Attackers: {attackers}")
    print(f"  - Initial states: {x_init}")
    
    # Run simulation
    results = alg.run_simulation(x_init)
    
    # Create diagnostic plots
    print("\\nGenerating diagnostic plots...")
    alg.create_diagnostic_plots(results, save_figures=True)
    
    # Print analysis
    alg.print_detection_analysis(results)
    
    print("\\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  - general_alg1_trajectory_diagnostic.png/.pdf")
    print("  - general_alg1_detection_diagnostic.png/.pdf")
    print("\\nThese figures demonstrate the detection mechanism")
    print("and show why attackers are flagged by the algorithm.")

if __name__ == "__main__":
    main()
