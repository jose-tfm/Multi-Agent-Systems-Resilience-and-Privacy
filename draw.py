import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ── 1) define the 15×15 augmented matrix A^P ─────────────────────────────
Ap = np.array([
    [0,       1/4,     1/4,     1/12,    1/8,     1/4,     1/24,    0,       0,       0,       0,       0,       0,       0,       0      ],
    [1/4,     0,       1/4,     0,       0,       0,       0,       1/12,    1/8,     1/4,     1/24,    0,       0,       0,       0      ],
    [1/4,     1/4,     0,       0,       0,       0,       0,       0,       0,       0,       0,       1/12,    1/8,     1/4,     1/24   ],
    [1/11,    0,       0,       0,       3/22,    1/11,    15/22,   0,       0,       0,       0,       0,       0,       0,       0      ],
    [1/2,     0,       0,       1/2,     0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0      ],
    [3/4,     0,       0,       1/4,     0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0      ],
    [1/16,    0,       0,       15/16,   0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0      ],
    [0,       1/11,    0,       0,       0,       0,       0,       0,       3/22,    1/11,    15/22,   0,       0,       0,       0      ],
    [0,       1/2,     0,       0,       0,       0,       0,       1/2,     0,       0,       0,       0,       0,       0,       0      ],
    [0,       3/4,     0,       0,       0,       0,       0,       1/4,     0,       0,       0,       0,       0,       0,       0      ],
    [0,       1/16,    0,       0,       0,       0,       0,       15/16,   0,       0,       0,       0,       0,       0,       0      ],
    [0,       0,       1/11,    0,       0,       0,       0,       0,       0,       0,       0,       0,       3/22,    1/11,    15/22   ],
    [0,       0,       1/2,     0,       0,       0,       0,       0,       0,       0,       0,       1/2,     0,       0,       0      ],
    [0,       0,       3/4,     0,       0,       0,       0,       0,       0,       0,       0,       1/4,     0,       0,       0      ],
    [0,       0,       1/16,    0,       0,       0,       0,       0,       0,       0,       0,       15/16,   0,       0,       0      ],
], dtype=float)

# ── 2) build directed graph from A^P ───────────────────────────────────────
G = nx.from_numpy_array(Ap, create_using=nx.DiGraph)

# ── 3) designate centers and their privacy leaves ─────────────────────────
centers = [0, 1, 2]
leaves = {
    0: [3, 4, 5, 6],
    1: [7, 8, 9, 10],
    2: [11, 12, 13, 14]
}

# ── 4) hand‐specify positions for a nice layout ────────────────────────────
pos = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}
angles = [-2.5, -3.9, -5.2]
r = 0.8
for i, θ in zip(centers, angles):
    cx, cy = pos[i]
    for j, ℓ in enumerate(leaves[i]):
        a = θ + j*0.4
        pos[ℓ] = (cx + r*np.cos(a), cy + r*np.sin(a))

# ── 5) split edges into reciprocals vs singletons ─────────────────────────
recips = set()
singles = []
for u, v in G.edges():
    if G.has_edge(v, u):
        if (v, u) not in recips:
            recips.add((u, v))
    else:
        singles.append((u, v))

# ── 6) draw the graph ─────────────────────────────────────────────────────
plt.figure(figsize=(6, 6))
ax = plt.gca()

# 6a) draw singleton edges
nx.draw_networkx_edges(
    G, pos,
    edgelist=singles,
    edge_color='lightgray',
    arrowsize=8,
    arrowstyle='-|>',
    width=1,
    ax=ax
)

# 6b) draw reciprocal edges as curved arrows
for u, v in recips:
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v)],
        connectionstyle='arc3,rad=0.2',
        edge_color='gray',
        arrowsize=10,
        arrowstyle='-|>',
        width=1.5,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(v, u)],
        connectionstyle='arc3,rad=-0.2',
        edge_color='gray',
        arrowsize=10,
        arrowstyle='-|>',
        width=1.5,
        ax=ax
    )

# 6c) highlight the center cycle in thick black
for u, v in recips:
    if u in centers and v in centers:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle='arc3,rad=0.2',
            edge_color='black',
            arrowsize=12,
            arrowstyle='-|>',
            width=3,
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(v, u)],
            connectionstyle='arc3,rad=-0.2',
            edge_color='black',
            arrowsize=12,
            arrowstyle='-|>',
            width=3,
            ax=ax
        )

# ── 7) draw nodes ─────────────────────────────────────────────────────────
nx.draw_networkx_nodes(
    G, pos,
    nodelist=centers,
    node_color='lightgreen',
    node_size=700,
    ax=ax
)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=sum(leaves.values(), []),
    node_color='gold',
    node_size=400,
    ax=ax
)

# ── 8) add labels and show ───────────────────────────────────────────────
labels = {i: str(i+1) for i in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)

plt.axis('off')
plt.tight_layout()
plt.show()
