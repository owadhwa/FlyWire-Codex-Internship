import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict
import random

# ============================================================
# FILES
# ============================================================

files = {
    "MCNS": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/mcns_0.9_edge_list.csv",
    "BANC": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/banc_626_edge_list.csv",
    "FAFB": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/fafb_783_edge_list.csv",
    "MAOL": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/maol_1.1_edge_list.csv",
    "MANC": "/Users/omikawadhwa/Documents/GitHub/Image Analysis/Edge Lists/manc_1.2.1_edge_list.csv",
}

SOURCE_COL = "source neuron id"
TARGET_COL = "target neuron id"
OUT_DIR = "/Users/omikawadhwa/Documents/GitHub/Image Analysis"

random.seed(1)

# ============================================================
# LOAD GRAPHS
# ============================================================

def load_graph(path):
    df = pd.read_csv(path)

    G = nx.DiGraph()
    G.add_edges_from(
        zip(df[SOURCE_COL].astype(str), df[TARGET_COL].astype(str))
    )

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


graphs = {name: load_graph(path) for name, path in files.items()}

for name, G in graphs.items():
    print(name, G.number_of_nodes(), G.number_of_edges())

# ============================================================
# USE GIANT SCC AS SEARCH SPACE
# ============================================================

def largest_scc(G):
    nodes = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(nodes).copy()


search_graphs = {
    name: largest_scc(G)
    for name, G in graphs.items()
}

print("\nLargest SCCs:")
for name, G in search_graphs.items():
    print(name, G.number_of_nodes(), G.number_of_edges())

# ============================================================
# FAST CANONICAL-LIKE SIGNATURE FOR SMALL INDUCED DIGRAPHS
# ============================================================

def canonical_signature(S):
    """
    Hashable signature for small directed induced subgraphs.
    Converts all nested lists to tuples so it can be used as a dictionary key.
    """

    nodes = list(S.nodes())

    node_features = {}

    for n in nodes:
        pred_degrees = tuple(
            sorted(
                S.in_degree(p) + S.out_degree(p)
                for p in S.predecessors(n)
            )
        )

        succ_degrees = tuple(
            sorted(
                S.in_degree(s) + S.out_degree(s)
                for s in S.successors(n)
            )
        )

        node_features[n] = (
            S.in_degree(n),
            S.out_degree(n),
            pred_degrees,
            succ_degrees
        )

    ordered = sorted(
        nodes,
        key=lambda n: node_features[n]
    )

    idx = {
        n: i
        for i, n in enumerate(ordered)
    }

    edges = tuple(
        sorted(
            (idx[u], idx[v])
            for u, v in S.edges()
        )
    )

    return (
        S.number_of_nodes(),
        S.number_of_edges(),
        tuple(node_features[n] for n in ordered),
        edges
    )

# ============================================================
# GROW CONNECTED INDUCED CANDIDATES
# ============================================================

def grow_candidate(G, seed, size):
    selected = {seed}
    frontier = {seed}

    while len(selected) < size and frontier:
        neighbors = set()

        for n in frontier:
            neighbors.update(G.predecessors(n))
            neighbors.update(G.successors(n))

        neighbors -= selected

        if not neighbors:
            break

        # mostly high-degree, occasionally random
        if random.random() < 0.85:
            nxt = max(
                neighbors,
                key=lambda x: G.in_degree(x) + G.out_degree(x)
            )
        else:
            nxt = random.choice(list(neighbors))

        selected.add(nxt)
        frontier = {nxt}

    if len(selected) != size:
        return None

    return G.subgraph(selected).copy()


def generate_candidates(G, size, max_candidates=5000):
    candidates = []
    seen_node_sets = set()

    high_degree_nodes = sorted(
        G.nodes(),
        key=lambda n: G.in_degree(n) + G.out_degree(n),
        reverse=True
    )

    seeds = high_degree_nodes[:max_candidates]

    if G.number_of_nodes() > max_candidates:
        seeds += random.sample(
            list(G.nodes()),
            max_candidates
        )

    for seed in seeds:
        S = grow_candidate(G, seed, size)

        if S is None:
            continue

        key = tuple(sorted(S.nodes()))

        if key in seen_node_sets:
            continue

        seen_node_sets.add(key)
        candidates.append(S)

        if len(candidates) >= max_candidates:
            break

    return candidates

# ============================================================
# SIGNATURE SEARCH
# ============================================================

def search_by_signature(
    search_graphs,
    min_size=3,
    max_size=20,
    min_datasets=3,
    max_candidates=10000
):

    best = None
    all_hits = []

    for size in range(max_size, min_size - 1, -1):

        print("\nSearching size", size)

        buckets = defaultdict(list)

        for dataset, G in search_graphs.items():

            print("Generating", dataset)

            candidates = generate_candidates(
                G,
                size=size,
                max_candidates=max_candidates
            )

            print(dataset, "candidates:", len(candidates))

            for S in candidates:

                sig = canonical_signature(S)

                buckets[sig].append({
                    "dataset": dataset,
                    "graph": S,
                    "nodes": list(S.nodes())
                })

        size_hits = []

        for sig, items in buckets.items():

            datasets_present = sorted(
                set(x["dataset"] for x in items)
            )

            if len(datasets_present) >= min_datasets:

                chosen = []
                used = set()

                for item in items:
                    if item["dataset"] not in used:
                        chosen.append(item)
                        used.add(item["dataset"])

                    if len(chosen) >= min_datasets:
                        break

                hit = {
                    "size": size,
                    "signature": sig,
                    "items": chosen,
                    "datasets": [x["dataset"] for x in chosen],
                    "n_datasets": len(chosen)
                }

                size_hits.append(hit)
                all_hits.append(hit)

        print(
            "Hits at size",
            size,
            ":",
            len(size_hits)
        )

        # Since we search from large to small,
        # if we found any hit at this size, this is the largest N.
        # Choose best among hits of the same size.
        if len(size_hits) > 0:

            best = max(
                size_hits,
                key=lambda h: (
                    h["size"],
                    h["n_datasets"]
                )
            )

            print("\nBEST HIT FOUND")
            print("size:", best["size"])
            print("datasets:", best["datasets"])

            return best

    return best

# ============================================================
# RUN
# ============================================================

hit = search_by_signature(
    search_graphs,
    min_size=3,
    max_size=20,
    min_datasets=3,
    max_candidates=1000
)

# ============================================================
# SAVE SOLUTION
# ============================================================

if hit is None:
    print("No match found.")
else:
    # The ordering inside canonical signature is not directly stored,
    # so save node sets first.
    # Then verify manually/with VF2 on these tiny graphs.

    items = hit["items"]

    solution = pd.DataFrame({
        item["dataset"]: item["nodes"]
        for item in items
    })

    out = f"{OUT_DIR}/network.csv"
    solution.to_csv(out, index=False)

    print("Saved:", out)
    print(solution.head())