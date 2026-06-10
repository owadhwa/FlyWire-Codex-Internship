import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict
from networkx.algorithms import isomorphism as iso
import random
import time

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

# ============================================================
# PARAMETERS
# ============================================================

MIN_SIZE = 3
MAX_SIZE = 20
MIN_DATASETS = 3
MAX_CANDIDATES = 2000

RANDOM_SEED = 1
EXPANSION_WIDTH = 3

USE_WEAKLY_CONNECTED_COMPONENT = True

random.seed(RANDOM_SEED)

# ============================================================
# LOAD GRAPHS
# ============================================================

def load_graph(path):
    df = pd.read_csv(path)

    G = nx.DiGraph()

    G.add_edges_from(
        zip(
            df[SOURCE_COL].astype(str),
            df[TARGET_COL].astype(str)
        )
    )

    G.remove_edges_from(nx.selfloop_edges(G))

    return G


graphs = {
    name: load_graph(path)
    for name, path in files.items()
}

print("\nFull graphs:")
for name, G in graphs.items():
    print(
        name,
        "nodes:",
        G.number_of_nodes(),
        "edges:",
        G.number_of_edges()
    )

# ============================================================
# SEARCH SPACE: LARGEST WCC OR SCC
# ============================================================

def largest_wcc(G):
    nodes = max(
        nx.weakly_connected_components(G),
        key=len
    )
    return G.subgraph(nodes).copy()


def largest_scc(G):
    nodes = max(
        nx.strongly_connected_components(G),
        key=len
    )
    return G.subgraph(nodes).copy()


if USE_WEAKLY_CONNECTED_COMPONENT:

    search_graphs = {
        name: largest_wcc(G)
        for name, G in graphs.items()
    }

    print("\nUsing largest weakly connected components as search space:")

else:

    search_graphs = {
        name: largest_scc(G)
        for name, G in graphs.items()
    }

    print("\nUsing largest strongly connected components as search space:")


for name, G in search_graphs.items():
    print(
        name,
        "nodes:",
        G.number_of_nodes(),
        "edges:",
        G.number_of_edges()
    )

# ============================================================
# SIGNATURE FUNCTION
# ============================================================

def canonical_signature(S):
    """
    Hashable structural signature for a directed induced subgraph.
    IDs are ignored; only directed connectivity structure contributes.
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
# GROW CONNECTED DIRECTED INDUCED CANDIDATE
# ============================================================

def grow_candidate(
    G,
    seed,
    size,
    expansion_width=3
):
    """
    Builds a connected local candidate circuit.

    selected = nodes already in candidate
    frontier = subset of selected nodes used for next expansion
    """

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

        neighbors = list(neighbors)

        # Mostly choose high-degree nodes, sometimes random.
        if random.random() < 0.85:
            nxt = max(
                neighbors,
                key=lambda x: G.in_degree(x) + G.out_degree(x)
            )
        else:
            nxt = random.choice(neighbors)

        selected.add(nxt)

        # Broader than a single-node walk, cheaper than frontier = selected.
        selected_sorted = sorted(
            selected,
            key=lambda x: G.in_degree(x) + G.out_degree(x),
            reverse=True
        )

        frontier = set(
            selected_sorted[:min(expansion_width, len(selected_sorted))]
        )

    if len(selected) != size:
        return None

    return G.subgraph(selected).copy()

# ============================================================
# GENERATE CANDIDATES WITHOUT OVERLAP FILTERING
# ============================================================

def generate_candidates(
    G,
    size,
    max_candidates=5000,
    expansion_width=3
):
    """
    Generates connected directed induced candidates.
    No overlap filtering.
    Only exact duplicate node sets are removed.
    """

    candidates = []

    high_degree_nodes = sorted(
        G.nodes(),
        key=lambda n: G.in_degree(n) + G.out_degree(n),
        reverse=True
    )

    seeds = []

    # high-degree seeds
    seeds.extend(
        high_degree_nodes[:min(max_candidates, len(high_degree_nodes))]
    )

    # random seeds
    if G.number_of_nodes() > max_candidates:
        seeds.extend(
            random.sample(
                list(G.nodes()),
                k=max_candidates
            )
        )

    seen_exact = set()

    for seed in seeds:

        S = grow_candidate(
            G,
            seed=seed,
            size=size,
            expansion_width=expansion_width
        )

        if S is None:
            continue

        node_key = tuple(sorted(S.nodes()))

        # remove only exact duplicate node sets
        if node_key in seen_exact:
            continue

        seen_exact.add(node_key)
        candidates.append(S)

        if len(candidates) >= max_candidates:
            break

    return candidates

# ============================================================
# EXACT VERIFICATION
# ============================================================

def exact_isomorphic(S1, S2):
    """
    Exact directed graph isomorphism between two induced subgraphs.
    """

    if S1.number_of_nodes() != S2.number_of_nodes():
        return False, None

    if S1.number_of_edges() != S2.number_of_edges():
        return False, None

    deg1 = sorted(
        (S1.in_degree(n), S1.out_degree(n))
        for n in S1.nodes()
    )

    deg2 = sorted(
        (S2.in_degree(n), S2.out_degree(n))
        for n in S2.nodes()
    )

    if deg1 != deg2:
        return False, None

    GM = iso.DiGraphMatcher(S1, S2)
    ok = GM.is_isomorphic()

    return ok, GM.mapping if ok else None

# ============================================================
# SEARCH BY SIGNATURE
# ============================================================

def search_by_signature(
    search_graphs,
    min_size=3,
    max_size=20,
    min_datasets=3,
    max_candidates=5000,
    expansion_width=3
):
    """
    Size-descending heuristic search.

    Objective:
        maximize number of nodes N

    Constraint:
        same directed induced structure appears in >= min_datasets datasets
    """

    best = None

    for size in range(max_size, min_size - 1, -1):

        size_start = time.time()

        print("\n================================================")
        print("Searching size:", size)
        print("================================================")

        buckets = defaultdict(list)

        for dataset, G in search_graphs.items():

            print("Generating candidates for", dataset)

            candidates = generate_candidates(
                G,
                size=size,
                max_candidates=max_candidates,
                expansion_width=expansion_width
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
                set(item["dataset"] for item in items)
            )

            if len(datasets_present) < min_datasets:
                continue

            chosen = []
            used = set()

            for item in items:

                if item["dataset"] not in used:
                    chosen.append(item)
                    used.add(item["dataset"])

                if len(chosen) >= min_datasets:
                    break

            # Exact verification against first chosen graph.
            ref = chosen[0]
            verified = [ref]
            mappings = {}

            for item in chosen[1:]:

                ok, mapping = exact_isomorphic(
                    ref["graph"],
                    item["graph"]
                )

                if ok:
                    verified.append(item)
                    mappings[item["dataset"]] = mapping

            if len(verified) >= min_datasets:

                hit = {
                    "size": size,
                    "signature": sig,
                    "items": verified,
                    "datasets": [x["dataset"] for x in verified],
                    "n_datasets": len(verified),
                    "mappings": mappings
                }

                size_hits.append(hit)

        print("Hits at size", size, ":", len(size_hits))
        print("Elapsed for size", size, ":", time.time() - size_start, "seconds")

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
# SAVE SOLUTION
# ============================================================

def save_solution(hit, out_dir):

    if hit is None:
        print("\nNo shared motif found.")
        return None

    items = hit["items"]

    # Use first verified item as reference order.
    ref_item = items[0]
    ref_dataset = ref_item["dataset"]
    ref_nodes = list(ref_item["graph"].nodes())

    solution = pd.DataFrame()
    solution[ref_dataset] = ref_nodes

    for item in items[1:]:

        dataset = item["dataset"]

        ok, mapping = exact_isomorphic(
            ref_item["graph"],
            item["graph"]
        )

        if not ok:
            raise RuntimeError(
                f"Unexpected verification failure for {dataset}"
            )

        # mapping is ref_node -> item_node
        solution[dataset] = [
            mapping[n]
            for n in ref_nodes
        ]

    out_path = (
        f"{out_dir}/network.csv"
    )

    solution.to_csv(
        out_path,
        index=False
    )

    print("\nSaved solution:")
    print(out_path)

    print("\nSolution shape:", solution.shape)
    print(solution.head())

    return solution

# ============================================================
# FINAL VERIFICATION
# ============================================================

def verify_solution(solution, full_graphs):

    print("\nFinal verification:")

    subgraphs = {}

    for dataset in solution.columns:

        nodes = solution[dataset].astype(str).tolist()

        S = full_graphs[dataset].subgraph(nodes).copy()

        subgraphs[dataset] = S

        print(
            dataset,
            "nodes:",
            S.number_of_nodes(),
            "edges:",
            S.number_of_edges()
        )

    for d1, d2 in combinations(solution.columns, 2):

        ok, _ = exact_isomorphic(
            subgraphs[d1],
            subgraphs[d2]
        )

        print(
            d1,
            "vs",
            d2,
            "isomorphic:",
            ok
        )

# ============================================================
# RUN
# ============================================================

start = time.time()

hit = search_by_signature(
    search_graphs,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    min_datasets=MIN_DATASETS,
    max_candidates=MAX_CANDIDATES,
    expansion_width=EXPANSION_WIDTH
)

print("\nTotal elapsed seconds:", time.time() - start)

solution = save_solution(
    hit,
    OUT_DIR
)

if solution is not None:
    verify_solution(
        solution,
        graphs
    )