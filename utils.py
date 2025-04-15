import numpy as np
import matplotlib.pyplot as plt

def plot_state(states, correct_consensus, att_v, agent_ids, title, xlabel, ylabel):
    """
    Parameters:
      - states: A numpy array of shape (num_iterations, num_agents) containing the state evolution.
      - correct_consensus: The consensus value to mark).
      - att_v: Dictionary of attacked agents. Keys are agent IDs, values are functions returning the attack value.
      - agent_ids: List of agent IDs corresponding to the columns in states.
    """
    iterations = np.arange(states.shape[0])
    plt.figure(figsize=(10,6))

    for col, agent in enumerate(agent_ids):
        if agent in att_v:
            plt.plot(iterations, states[:, col],
                     marker='v', linestyle='--',
                     label=f'Attacked Agent {agent}')
        else:
            plt.plot(iterations, states[:, col],
                     marker='o', label=f'Agent {agent}')

    plt.axhline(correct_consensus, color='gray', linestyle='--', label="Correct Consensus")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
