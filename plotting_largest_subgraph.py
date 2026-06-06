import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------
# FILES
# --------------------------------------------------

solution_file = "/Users/omikawadhwa/Documents/GitHub/Image Analysis/network.csv"

edge_files = {
    "MCNS": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/mcns_0.9_edge_list.csv",
    "BANC": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/banc_626_edge_list.csv",
    "FAFB": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/fafb_783_edge_list.csv",
}

SOURCE_COL = "source neuron id"
TARGET_COL = "target neuron id"

DATASET_TO_PLOT = "FAFB"  # "MCNS", "BANC", or "FAFB"

# --------------------------------------------------
# LOAD SOLUTION TABLE
# --------------------------------------------------

solution = pd.read_csv(solution_file)

nodes = solution[DATASET_TO_PLOT].astype(str).tolist()

print("Number of solution nodes:", len(nodes))

# --------------------------------------------------
# LOAD EDGE LIST
# --------------------------------------------------

edges = pd.read_csv(edge_files[DATASET_TO_PLOT])

edges[SOURCE_COL] = edges[SOURCE_COL].astype(str)
edges[TARGET_COL] = edges[TARGET_COL].astype(str)

# --------------------------------------------------
# KEEP ONLY EDGES BETWEEN SOLUTION NODES
# This gives the induced directed subgraph
# --------------------------------------------------

node_set = set(nodes)

sub_edges = edges[
    edges[SOURCE_COL].isin(node_set) &
    edges[TARGET_COL].isin(node_set)
].copy()

print("Number of induced edges:", len(sub_edges))

# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

G = nx.DiGraph()

G.add_nodes_from(nodes)

G.add_edges_from(
    zip(
        sub_edges[SOURCE_COL],
        sub_edges[TARGET_COL]
    )
)

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(
    G,
    seed=1,
    k=0.8,
    iterations=200
)

node_sizes = [
    300 + 100 * (G.in_degree(n) + G.out_degree(n))
    for n in G.nodes()
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    alpha=0.85
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=18,
    width=1.5,
    alpha=0.6,
    connectionstyle="arc3,rad=0.08"
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7
)

plt.title(f"{DATASET_TO_PLOT} induced directed circuit")
plt.axis("off")
plt.tight_layout()

out_png = f"/Users/omikawadhwa/Documents/GitHub/Image Analysis/{DATASET_TO_PLOT}_shared_circuit_graph.png"

plt.savefig(out_png, dpi=300)
plt.show()

print("Saved:", out_png)