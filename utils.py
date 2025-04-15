import numpy as np
import matplotlib.pyplot as plt

def plot_state(states, correct_consensus, att_v, title, xlabel, ylabel):

    """
    Parameters:
      - states: A numpy array of shape (num_iterations, num_agents) containing the state evolution.
      - title: Title of the plot.
      - xlabel: Label for the x-axis.
      - ylabel: Label for the y-axis.
    """
    iterations = np.arange(states.shape[0])
    plt.figure(figsize=(10,6))

    # Plot each agent's state evolution.
    for i in range(states.shape[1]):
        if i in att_v:
            attack_val = att_v  
            plt.plot(iterations, states[:, i], marker='v', linestyle='--', label=f'Attacked Agent {i}')
        else:
            plt.plot(iterations, states[:, i], marker='o', label=f'Agent {i}')

    plt.axhline(correct_consensus, color='gray', linestyle='--', label="Correct Consensus")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()