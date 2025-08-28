"""
Configuration file for M-MSR algorithm experiments.
This allows easy modification of parameters without changing the main code.
"""

import numpy as np

# ============================================================================
# NETWORK CONFIGURATIONS
# ============================================================================

# Default 10-agent network
DEFAULT_NETWORK = [
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

# Smaller test network (4 agents)
SMALL_NETWORK = [
    [0, 1, 2, 3],       
    [0, 1, 2, 3],   
    [0, 1, 2, 3],
    [0, 1, 2, 3],     
]

# Ring network (each agent connected to 2 neighbors)
def create_ring_network(n):
    """Create a ring network topology."""
    return [[(i-1) % n, (i+1) % n] for i in range(n)]

# Complete network (all agents connected to all)
def create_complete_network(n):
    """Create a complete network topology."""
    return [list(range(n)) for _ in range(n)]

# ============================================================================
# ATTACK CONFIGURATIONS
# ============================================================================

# Constant attacks
CONSTANT_ATTACKS = {
    0: (lambda t: 0.3),
    8: (lambda t: 0.8)
}

# Time-varying attacks
DYNAMIC_ATTACKS = {
    0: (lambda t: 0.3 + 0.1 * np.sin(t * 0.2)),  # Sinusoidal
    8: (lambda t: min(0.9, 0.5 + t * 0.01))      # Gradual increase
}

# Single attacker scenarios
SINGLE_ATTACK_LOW = {0: (lambda t: 0.1)}
SINGLE_ATTACK_HIGH = {8: (lambda t: 0.9)}

# Multiple attackers
MULTIPLE_ATTACKS = {
    0: (lambda t: 0.2),
    2: (lambda t: 0.8),
    7: (lambda t: 0.1)
}

# ============================================================================
# ALGORITHM PARAMETERS
# ============================================================================

class MSRConfig:
    """Configuration class for M-MSR algorithm parameters."""
    
    def __init__(self):
        # Network topology
        self.neighbors = DEFAULT_NETWORK
        
        # Initial states
        self.x0 = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.5, 0.4, 0.6, 0.3]
        
        # Algorithm parameters
        self.max_iterations = 30
        self.f = 2  # Filter parameter
        self.tolerance = 1e-6
        
        # Attack configuration
        self.attacks = CONSTANT_ATTACKS
        
        # Visualization parameters
        self.agent_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.plot_title = "W-MSR Algorithm"
        self.save_plots = False
        self.plot_directory = "./plots/"
        
        # Analysis parameters
        self.verbose = True
        self.run_parameter_study = True
        self.convergence_tolerance = 1e-3

# ============================================================================
# EXPERIMENTAL SCENARIOS
# ============================================================================

def get_scenario_config(scenario_name):
    """
    Get predefined experimental scenarios.
    
    Args:
        scenario_name: str, one of:
            - "default": Standard 10-agent network with 2 attackers
            - "small": 4-agent complete network with 1 attacker  
            - "ring": 8-agent ring network with 1 attacker
            - "heavy_attack": 10-agent network with 3 attackers
            - "dynamic": 10-agent network with time-varying attacks
    """
    config = MSRConfig()
    
    if scenario_name == "default":
        # Already set by default
        pass
        
    elif scenario_name == "small":
        config.neighbors = SMALL_NETWORK
        config.x0 = [0.2, 1.0, 0.0, 0.6]
        config.attacks = {0: (lambda t: 0.2)}
        config.agent_ids = [1, 2, 3, 4]
        config.f = 1
        
    elif scenario_name == "ring":
        config.neighbors = create_ring_network(8)
        config.x0 = [0.1, 0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6]
        config.attacks = {2: (lambda t: 0.9)}
        config.agent_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        config.f = 1
        
    elif scenario_name == "heavy_attack":
        config.attacks = MULTIPLE_ATTACKS
        config.f = 3
        
    elif scenario_name == "dynamic":
        config.attacks = DYNAMIC_ATTACKS
        config.max_iterations = 50
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    return config

# ============================================================================
# PARAMETER STUDY CONFIGURATIONS
# ============================================================================

PARAMETER_STUDIES = {
    "filter_values": {
        "parameter": "f",
        "values": [1, 2, 3, 4],
        "description": "Effect of filter parameter f"
    },
    
    "attack_strength": {
        "parameter": "attack_value",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
        "description": "Effect of attack strength"
    },
    
    "network_size": {
        "parameter": "network_size", 
        "values": [6, 8, 10, 12],
        "description": "Effect of network size"
    },
    
    "attack_count": {
        "parameter": "num_attackers",
        "values": [1, 2, 3, 4],
        "description": "Effect of number of attackers"
    }
}
